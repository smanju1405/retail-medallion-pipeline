"""Paths, LLM provider settings and tunables for the retail medallion pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
SAMPLE_DATA_DIR = ROOT_DIR / "sample_data"

UPLOAD_DIR = DATA_DIR / "uploads"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
STTM_DIR = DATA_DIR / "sttm"
PROFILE_DIR = DATA_DIR / "profiles"
REPORT_DIR = DATA_DIR / "reports"
TRACE_DIR = DATA_DIR / "traces"
AUDIT_DIR = DATA_DIR / "audit"
RUN_DIR = DATA_DIR / "runs"
MEMORY_DIR = DATA_DIR / "memory"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()  # "groq" or "google"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE = 0.0  # data work must be repeatable

MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "25"))  # LangGraph recursion limit
MAX_QUERY_ROWS = 200  # rows returned to the Reporter agent per SQL call


def run_dir_for(layer_dir: Path, run_id: str) -> Path:
    """Per-run output folder, e.g. data/silver/<run_id>/, so runs never overwrite each other."""
    folder = layer_dir / run_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def ensure_dirs() -> None:
    """Create every output folder the pipeline writes to."""
    for folder in (
        UPLOAD_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, STTM_DIR, PROFILE_DIR,
        REPORT_DIR, TRACE_DIR, AUDIT_DIR, RUN_DIR, MEMORY_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)
