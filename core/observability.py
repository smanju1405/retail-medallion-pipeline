"""AgentTrace: records what each agent planned, called and produced.

One trace file per agent per run, data/traces/trace_{agent}_{run_id[:8]}.json.
Each agent invocation appends one entry to that file.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core import config


def _as_text(content: Any) -> str:
    """LangChain message content can be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content)


@dataclass
class AgentTrace:
    agent: str
    run_id: str
    inputs: dict = field(default_factory=dict)
    plan: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    reasoning_steps: list[dict] = field(default_factory=list)
    output: Any = None
    status: str = "running"
    error: str | None = None
    duration_seconds: float = 0.0
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    _t0: float = field(default_factory=time.monotonic, repr=False)

    def extract_from_messages(self, messages: list) -> None:
        """Pull plan, tool calls and the full message chain out of a ReAct run."""
        from langchain_core.messages import AIMessage

        for message in messages:
            text = _as_text(message.content)
            self.reasoning_steps.append({"type": type(message).__name__, "content": text[:2000]})
            if isinstance(message, AIMessage):
                if text and self.plan is None:
                    self.plan = text[:1000]
                for call in message.tool_calls or []:
                    self.tool_calls.append({"name": call["name"], "args": call["args"]})

    def finish(self, output: Any = None, error: str | None = None) -> None:
        self.output = output
        self.error = error
        self.status = "failed" if error else "success"
        self.duration_seconds = round(time.monotonic() - self._t0, 3)

    def save(self) -> None:
        config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = config.TRACE_DIR / f"trace_{self.agent}_{self.run_id[:8]}.json"
        entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        entry = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        entries.append(entry)
        path.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")
