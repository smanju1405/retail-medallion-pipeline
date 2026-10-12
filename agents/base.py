"""Shared plumbing for every agent: run context and the ReAct runner.

Each agent module exposes:
    run(ctx, task)  -> str   runs the agent (LLM ReAct loop, or plain code when ctx.llm is None)
    tools(ctx)      -> list  LangChain tools that close over ctx (the deck's tool-factory pattern)

ctx.scratchpad is the within-phase dict that tools share. The orchestrator copies it
into PipelineState when the phase finishes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import config
from core.audit import AuditLogger
from core.observability import AgentTrace, _as_text

REACT_PREAMBLE = """Work in this loop for every task:
THINK about the goal. INSPECT with the inspect tool first. PLAN the steps in one sentence.
ACT by calling the execution tool. VERIFY the tool output, then stop when it is correct.

Rules:
- Never invent file paths. Use only paths returned by tools.
- Never repeat raw customer values in your answer. Refer to columns and counts instead.
- If a tool returns a string starting with ERROR, read it, fix the cause if you can, and retry at most once.
"""


@dataclass
class AgentContext:
    run_id: str
    audit: AuditLogger
    llm: Any = None  # None means deterministic mode: plain code, no LLM calls
    inputs: dict = field(default_factory=dict)
    scratchpad: dict = field(default_factory=dict)


def _trace_inputs(inputs: dict) -> dict:
    """Keep file names only in traces, not full paths or data."""
    out = {}
    for key, value in inputs.items():
        if isinstance(value, (str, Path)) and Path(str(value)).suffix:
            out[key] = Path(str(value)).name
        elif isinstance(value, list) and value and all(isinstance(v, (str, Path)) for v in value):
            out[key] = [Path(str(v)).name for v in value]
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out


def run_react_agent(ctx: AgentContext, name: str, tools: list, system_prompt: str, task: str) -> str:
    """Run one LangChain ReAct agent to completion and return its final message text."""
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage

    agent = create_agent(
        model=ctx.llm,
        tools=tools,
        system_prompt=f"{system_prompt}\n\n{REACT_PREAMBLE}",
        name=name,
    )
    trace = AgentTrace(agent=name, run_id=ctx.run_id, inputs=_trace_inputs(ctx.inputs))
    ctx.audit.log("agent_start", name, "running", task=task[:500])
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"recursion_limit": config.MAX_AGENT_STEPS},
        )
        messages = result["messages"]
        trace.extract_from_messages(messages)
        final = _as_text(messages[-1].content).strip()
        trace.finish(output=final[:4000])
        ctx.audit.log("agent_complete", name, "success", tool_calls=len(trace.tool_calls))
        return final
    except Exception as exc:
        trace.finish(error=f"{type(exc).__name__}: {exc}")
        ctx.audit.log("agent_fail", name, "failed", error=str(exc)[:500])
        raise
    finally:
        trace.save()
