"""Checks that the LLM path is wired correctly, without calling any model.

Building a ReAct graph and converting tools to schemas happen offline, so a bad
tool signature or prompt wiring fails here, not in the live run.
"""

from agents import bronze_agent, gold_agent, profiler, reporter, silver_agent, sttm_generator
from agents.base import AgentContext
from agents.orchestrator import PHASE_STEPS, _step_tool
from core.audit import AuditLogger


def _ctx(llm=None):
    return AgentContext(run_id="00000000-0000-0000-0000-000000000000", audit=AuditLogger("00000000-0000-0000-0000-000000000000"), llm=llm)


def test_every_agent_exposes_named_tools_with_schemas():
    ctx = _ctx()
    con_tools = reporter.tools(ctx, reporter.load_gold([]))
    tool_sets = {
        "profiler": profiler.tools(ctx),
        "sttm_bronze": sttm_generator.tools(ctx, "bronze"),
        "bronze": bronze_agent.tools(ctx),
        "silver": silver_agent.tools(ctx),
        "gold": gold_agent.tools(ctx),
        "reporter": con_tools,
    }
    for agent, tools in tool_sets.items():
        assert tools, agent
        for tool in tools:
            assert tool.name and tool.description, f"{agent}:{tool.name}"
            assert tool.args_schema is not None


def test_supervisor_step_tools_are_named_for_the_phase_steps():
    ctx = _ctx()
    for phase, steps in PHASE_STEPS.items():
        tools = [_step_tool(ctx, name, step) for name, step in steps]
        assert [t.name for t in tools] == [f"run_{name}" for name, _ in steps], phase


def test_react_agent_graph_builds_with_a_groq_client_without_network():
    from langchain.agents import create_agent
    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key="test-key-not-used")
    graph = create_agent(model=llm, tools=profiler.tools(_ctx(llm)), system_prompt="test", name="profiler_agent")
    assert graph is not None


def test_llm_mode_requires_a_key_with_a_clear_message(monkeypatch):
    from core import config
    from core.llm import get_llm

    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    try:
        get_llm()
    except RuntimeError as exc:
        assert "GROQ_API_KEY" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when the key is missing")
