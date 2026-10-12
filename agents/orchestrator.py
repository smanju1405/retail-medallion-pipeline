"""Supervisor / Orchestrator.

Orchestrator (plain code, always runs):
  run_id, PipelineState, phase order, human approval gates, copying the scratchpad
  into PipelineState, verifying outputs, audit logging, error capture.

Supervisor (LLM, optional):
  Decides which sub-agent tools to call within a phase and writes a goal for each.
  With no LLM configured, the same steps run in fixed order as plain code.

Phases:
  1  Profiler -> Bronze STTM             -> awaiting_bronze_approval
  2  Bronze ingest -> Silver STTM        -> awaiting_silver_approval
  3  Silver cleanse -> Gold STTM         -> awaiting_gold_approval
  4  Gold materialise -> Reporter        -> complete
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable

import pandas as pd

from agents import bronze_agent, gold_agent, profiler, reporter, silver_agent, sttm_generator
from agents.base import AgentContext, run_react_agent
from core import config
from core.audit import AuditLogger
from core.memory import store_document
from core.sttm import read_sttm, write_sttm

PHASE_BY_STATUS = {"created": 1, "bronze_approved": 2, "silver_approved": 3, "gold_approved": 4}
NEXT_STATUS = {1: "awaiting_bronze_approval", 2: "awaiting_silver_approval", 3: "awaiting_gold_approval", 4: "complete"}
APPROVAL_STATUS = {"bronze": "awaiting_bronze_approval", "silver": "awaiting_silver_approval", "gold": "awaiting_gold_approval"}
APPROVED_STATUS = {"bronze": "bronze_approved", "silver": "silver_approved", "gold": "gold_approved"}
STTM_FIELD = {"bronze": "sttm_bronze_path", "silver": "sttm_silver_path", "gold": "sttm_gold_path"}

# Ordered (name, step) pairs per phase. A step is called as step(ctx, goal) -> str.
PHASE_STEPS = {
    1: [("profiler", profiler.run), ("sttm_bronze", partial(sttm_generator.run, layer="bronze"))],
    2: [("bronze", bronze_agent.run), ("sttm_silver", partial(sttm_generator.run, layer="silver"))],
    3: [("silver", silver_agent.run), ("sttm_gold", partial(sttm_generator.run, layer="gold"))],
    4: [("gold", gold_agent.run), ("reporter", reporter.run)],
}
# Scratchpad keys each phase must produce. Checked on disk after every phase.
PHASE_OUTPUTS = {
    1: ["profile_path", "sttm_bronze_path"],
    2: ["bronze_output_paths", "sttm_silver_path"],
    3: ["silver_output_paths", "sttm_gold_path"],
    4: ["gold_output_paths", "report_path", "report_json_path"],
}

SUPERVISOR_PROMPT = """You are the Supervisor agent in a retail sales Medallion pipeline.
You do not transform data yourself. You decide which sub-agent tools to call and write a short goal for each.
Call the tools listed in the task, in the order given, exactly once each.
If a tool returns ERROR, read it and retry that same tool at most once. If it still fails, stop and explain why.
When every tool has succeeded, reply with one line: the phase number and the outputs produced."""


@dataclass
class PipelineState:
    run_id: str
    business_intent: str
    uploaded_files: list[str]
    status: str = "created"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    profile_path: str | None = None
    sttm_bronze_path: str | None = None
    bronze_output_paths: list[str] = field(default_factory=list)
    sttm_silver_path: str | None = None
    silver_output_paths: list[str] = field(default_factory=list)
    sttm_gold_path: str | None = None
    gold_output_paths: list[str] = field(default_factory=list)
    report_path: str | None = None
    report_json_path: str | None = None
    summary: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _save(state: PipelineState) -> None:
    config.RUN_DIR.mkdir(parents=True, exist_ok=True)
    state.updated_at = datetime.now(timezone.utc).isoformat()
    (config.RUN_DIR / f"{state.run_id}.json").write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")


def load_state(run_id: str) -> PipelineState:
    data = json.loads((config.RUN_DIR / f"{run_id}.json").read_text(encoding="utf-8"))
    return PipelineState(**data)


def create_run(files: list[str], business_intent: str) -> PipelineState:
    """Start a run: new run_id, PipelineState, and the business intent saved to memory."""
    config.ensure_dirs()
    run_id = str(uuid.uuid4())
    state = PipelineState(run_id=run_id, business_intent=business_intent.strip(), uploaded_files=[str(f) for f in files])
    store_document(run_id, "business_intent", state.business_intent, {"files": [Path(f).name for f in files]})
    AuditLogger(run_id).log("run_created", "orchestrator", "success", files=[Path(f).name for f in files])
    _save(state)
    return state


def _inputs(state: PipelineState) -> dict:
    return {
        "files": list(state.uploaded_files),
        "business_intent": state.business_intent,
        "profile_path": state.profile_path,
        "sttm_bronze_path": state.sttm_bronze_path,
        "bronze_output_paths": list(state.bronze_output_paths),
        "sttm_silver_path": state.sttm_silver_path,
        "silver_output_paths": list(state.silver_output_paths),
        "sttm_gold_path": state.sttm_gold_path,
        "gold_output_paths": list(state.gold_output_paths),
    }


def _run_step(ctx: AgentContext, step: Callable, goal: str) -> str:
    """Run one sub-agent, then make its outputs visible to the next step in the same phase."""
    result = step(ctx, goal)
    ctx.inputs.update(ctx.scratchpad)
    return result


def _verify(scratchpad: dict, keys: list[str]) -> None:
    """Supervisor check: every promised output exists on disk. Never trust a tool's success message alone."""
    problems = []
    for key in keys:
        value = scratchpad.get(key)
        if not value:
            problems.append(f"{key} was not produced")
            continue
        for path in value if isinstance(value, list) else [value]:
            if not Path(path).exists():
                problems.append(f"{key}: file not found ({Path(path).name})")
    if problems:
        raise RuntimeError("Output verification failed: " + "; ".join(problems))


def _step_tool(ctx: AgentContext, name: str, step: Callable):
    from langchain_core.tools import StructuredTool

    def call(goal: str) -> str:
        """Run a sub-agent. The argument is a short goal describing what it should do."""
        try:
            return _run_step(ctx, step, goal)
        except Exception as exc:
            return f"ERROR: {exc}"

    return StructuredTool.from_function(func=call, name=f"run_{name}", description=f"Run the {name} sub-agent.")


def run_next_phase(state: PipelineState, llm=None) -> PipelineState:
    """Run the phase that matches the current status. Returns the updated state; errors are recorded, not raised."""
    phase = PHASE_BY_STATUS.get(state.status)
    if phase is None:
        raise ValueError(f"Run {state.run_id[:8]} is '{state.status}'. Approve the STTM before the next phase.")

    audit = AuditLogger(state.run_id)
    ctx = AgentContext(run_id=state.run_id, audit=audit, llm=llm, inputs=_inputs(state))
    steps = PHASE_STEPS[phase]
    audit.log("phase_start", "supervisor", "running", phase=phase, mode="llm" if llm else "deterministic")
    try:
        if llm is None:
            for name, step in steps:
                _run_step(ctx, step, f"{name} for phase {phase}")
        else:
            tools = [_step_tool(ctx, name, step) for name, step in steps]
            task = (f"Phase {phase}. Business intent: {state.business_intent or 'not provided'}. "
                    f"Call these tools in order: {', '.join(t.name for t in tools)}.")
            run_react_agent(ctx, f"supervisor_phase_{phase}", tools, SUPERVISOR_PROMPT, task)

        _verify(ctx.scratchpad, PHASE_OUTPUTS[phase])
    except Exception as exc:
        state.errors = [f"Phase {phase}: {exc}"]
        audit.log("phase_fail", "supervisor", "failed", phase=phase, error=str(exc)[:500])
        _save(state)
        return state

    for key, value in ctx.scratchpad.items():
        setattr(state, key, value)  # copy scratchpad into PipelineState, then the scratchpad is discarded
    state.status = NEXT_STATUS[phase]
    state.errors = []
    audit.log("phase_complete", "supervisor", "success", phase=phase, next_status=state.status)
    _save(state)
    return state


def approve(state: PipelineState, layer: str, edited_sttm: pd.DataFrame | None = None) -> PipelineState:
    """Human approval gate. Optionally saves the edited STTM first, then validates it."""
    if layer not in APPROVAL_STATUS:
        raise ValueError(f"Unknown layer '{layer}'")
    if state.status != APPROVAL_STATUS[layer]:
        raise ValueError(f"Run is '{state.status}', not awaiting {layer} approval")

    path = Path(getattr(state, STTM_FIELD[layer]))
    if edited_sttm is not None:
        write_sttm(edited_sttm, path)
    sttm = read_sttm(path)  # raises ValueError if the edited rules are invalid
    approved = int(((sttm["layer"] == layer) & (sttm["include"] == "Y")).sum())

    state.status = APPROVED_STATUS[layer]
    state.errors = []
    AuditLogger(state.run_id).log("human_approval", "human", "success", layer=layer, approved_rules=approved)
    _save(state)
    return state


def layer_awaiting_approval(state: PipelineState) -> str | None:
    for layer, status in APPROVAL_STATUS.items():
        if state.status == status:
            return layer
    return None


def run_pipeline(
    files: list[str],
    business_intent: str,
    llm=None,
    approve_fn: Callable[[PipelineState, str], pd.DataFrame | None] | None = None,
) -> PipelineState:
    """Start a new run and drive it through all four phases. See resume_pipeline."""
    return resume_pipeline(create_run(files, business_intent), llm=llm, approve_fn=approve_fn)


def resume_pipeline(
    state: PipelineState,
    llm=None,
    approve_fn: Callable[[PipelineState, str], pd.DataFrame | None] | None = None,
) -> PipelineState:
    """Drive an existing run forward. approve_fn(state, layer) is called at each gate.

    It returns an edited STTM DataFrame, or None to approve the file as generated.
    Without approve_fn the run stops at the first gate and returns the waiting state.
    """
    while state.status != "complete":
        layer = layer_awaiting_approval(state)
        if layer:
            if approve_fn is None:
                break
            state = approve(state, layer, approve_fn(state, layer))
            continue
        state = run_next_phase(state, llm)
        if state.errors:
            break
    return state
