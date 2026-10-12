"""Append-only JSON-lines audit log, one file per pipeline run.

Log file names, row counts and decisions only. Never raw data values.
"""

import json
from datetime import datetime, timezone

from core import config


class AuditLogger:
    def __init__(self, run_id: str):
        self.run_id = run_id
        config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        self.path = config.AUDIT_DIR / f"audit_{run_id}.jsonl"

    def log(self, event: str, agent: str, status: str, **details) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "event": event,
            "status": status,
            **details,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
