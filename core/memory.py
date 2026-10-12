"""Document store for business intent and final reports.

Stored as JSON files under data/memory/. The deck's stack lists ChromaDB; a
file store keeps the first version dependency-free. Swap the body of
store_document / retrieve_documents for a vector store later without
changing callers.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from core import config


def store_document(run_id: str, kind: str, text: str, metadata: dict | None = None) -> Path:
    """Persist one document (e.g. kind='business_intent' or 'final_report')."""
    config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MEMORY_DIR / f"{kind}_{run_id}.json"
    record = {
        "run_id": run_id,
        "kind": kind,
        "text": text,
        "metadata": metadata or {},
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def retrieve_documents(kind: str | None = None, run_id: str | None = None) -> list[dict]:
    """Return stored documents, optionally filtered by kind and/or run."""
    if not config.MEMORY_DIR.exists():
        return []
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(config.MEMORY_DIR.glob("*.json"))]
    return [d for d in docs if (kind is None or d["kind"] == kind) and (run_id is None or d["run_id"] == run_id)]
