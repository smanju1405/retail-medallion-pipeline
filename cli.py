"""Command-line runner for the retail medallion pipeline.

Examples:
    python cli.py --deterministic --auto-approve          # offline, no API key needed
    python cli.py                                         # LLM agents, pauses at each approval gate
    python cli.py --files my_sales_raw.csv --intent "Which region grew fastest?"
"""

from __future__ import annotations

import argparse
import glob
import sys

from agents.orchestrator import load_state, resume_pipeline, run_pipeline
from core import config

DEFAULT_INTENT = "Which customer region and product category drive the most revenue, and how does revenue trend by month?"


def _pause_for_review(state, layer: str):
    """Stop at the gate and wait for a person to review the STTM file."""
    print(f"\nApproval needed: {layer} STTM -> {getattr(state, f'sttm_{layer}_path')}")
    print("Review the rules (include Y/N column). Press Enter to approve, or Ctrl+C to stop.")
    input()
    return None  # keep the file as generated or as edited on disk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the agentic retail medallion pipeline.")
    parser.add_argument("--files", nargs="+", help="raw CSV files (default: sample_data/*_raw.csv)")
    parser.add_argument("--intent", default=DEFAULT_INTENT, help="business question the report should answer")
    parser.add_argument("--deterministic", action="store_true", help="run the same steps as plain code, without an LLM")
    parser.add_argument("--auto-approve", action="store_true", help="approve every gate without pausing")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue a paused run instead of starting a new one")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    llm = None
    if not args.deterministic:
        from core.llm import get_llm

        llm = get_llm()
    approve_fn = (lambda state, layer: None) if args.auto_approve else _pause_for_review

    if args.resume:
        state = resume_pipeline(load_state(args.resume), llm=llm, approve_fn=approve_fn)
    else:
        files = args.files or sorted(glob.glob(str(config.SAMPLE_DATA_DIR / "*_raw.csv")))
        if not files:
            print("No input CSV files found.", file=sys.stderr)
            return 2
        state = run_pipeline(files, args.intent, llm=llm, approve_fn=approve_fn)

    print(f"\nRun {state.run_id} status: {state.status}")
    if state.errors:
        print("Failed:", "; ".join(state.errors), file=sys.stderr)
        return 1
    if state.status == "complete":
        print(f"Report: {state.report_path}")
        print(f"Summary: {state.summary}")
    else:
        print(f"Paused. Resume with: python cli.py --resume {state.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
