"""Runs the agentic (LLM) path end to end with a scripted fake model. No network, no API key.

The fake model calls each tool bound to the current agent once, in order, then replies with text.
That exercises the real Supervisor, ReAct loops, tool closures, output verification, traces and the
approval gates. It does not test model quality. Real-model behaviour needs a live run with a key.
"""

import json
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents import orchestrator
from core import config

SAMPLE = config.SAMPLE_DATA_DIR
RAW_FILES = [str(SAMPLE / f) for f in ("customers_raw.csv", "products_raw.csv", "sales_raw.csv")]


class ScriptedChatModel(BaseChatModel):
    tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-test-model"

    def bind_tools(self, tools, **kwargs: Any):
        return self.model_copy(update={"tool_names": [t.name for t in tools]})

    def _args_for(self, name: str) -> dict:
        if name == "execute_query_tool":
            return {"sql": "SELECT month, total_revenue FROM sales_monthly LIMIT 5"}
        if name.startswith("run_"):
            return {"goal": f"execute {name[4:]} for this phase"}
        return {}

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        calls_made = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
        if calls_made < len(self.tool_names):
            name = self.tool_names[calls_made]
            message = AIMessage(content="", tool_calls=[{"name": name, "args": self._args_for(name), "id": f"call_{calls_made}"}])
        else:
            message = AIMessage(content="Phase finished. All tools succeeded.")
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_full_agentic_run_with_scripted_model_and_both_gates(tmp_path):
    llm = ScriptedChatModel()
    state = orchestrator.run_pipeline(RAW_FILES, "Which region drives revenue?", llm=llm)
    assert state.status == "awaiting_bronze_approval", state.errors
    assert state.sttm_bronze_path and Path(state.sttm_bronze_path).exists()

    state = orchestrator.approve(state, "bronze")
    state = orchestrator.resume_pipeline(state, llm=llm, approve_fn=lambda s, layer: None)
    assert state.status == "complete", state.errors
    assert Path(state.report_path).exists()
    assert state.summary  # the reporter's final text (or the deterministic fallback)


def test_llm_runs_write_agent_traces_with_tool_calls():
    state = orchestrator.run_pipeline(RAW_FILES, "trace test", llm=ScriptedChatModel(), approve_fn=None)
    assert state.status == "awaiting_bronze_approval", state.errors

    profiler_traces = list(config.TRACE_DIR.glob(f"trace_profiler_agent_{state.run_id[:8]}.json"))
    supervisor_traces = list(config.TRACE_DIR.glob(f"trace_supervisor_phase_1_{state.run_id[:8]}.json"))
    assert profiler_traces and supervisor_traces

    trace = json.loads(profiler_traces[0].read_text(encoding="utf-8"))[-1]
    assert trace["status"] == "success"
    assert [c["name"] for c in trace["tool_calls"]] == ["inspect_files_tool", "profiler_tool"]
    assert trace["duration_seconds"] >= 0 and trace["reasoning_steps"]


def test_inspect_tool_never_sends_pii_values_to_the_model():
    """The profiler's preview tool is the only place raw sample rows are built. Emails and names must not appear."""
    from agents import profiler
    from agents.base import AgentContext
    from core.audit import AuditLogger

    ctx = AgentContext(run_id="11111111-1111-1111-1111-111111111111",
                       audit=AuditLogger("11111111-1111-1111-1111-111111111111"), inputs={"files": RAW_FILES})
    inspect_tool = profiler.tools(ctx)[0]
    output = inspect_tool.invoke({})
    assert "@example.com" not in output
    assert "***masked***" in output
    assert "order_id" in output  # business columns still visible for planning


def test_supervisor_verifies_outputs_even_when_the_model_claims_success():
    """If a tool reports success but writes nothing, the Supervisor must refuse to advance the phase."""
    steps = orchestrator.PHASE_STEPS[1]
    original = steps[0]

    def lying_profiler(ctx, goal):  # claims success, writes nothing
        return "Profiled everything."

    steps[0] = ("profiler", lying_profiler)
    try:
        state = orchestrator.run_pipeline(RAW_FILES, "verify test", llm=ScriptedChatModel(), approve_fn=None)
    finally:
        steps[0] = original
    assert state.status == "created"
    assert state.errors and "verification failed" in state.errors[0]
