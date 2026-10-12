"""Agent 3 - Bronze Agent: ingests raw CSVs into Parquet, keeping the source values as-is.

Only columns approved in the Bronze STTM are kept. Metadata columns (_source_file,
_run_id, _ingested_at) are added for lineage.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agents.base import AgentContext, run_react_agent
from agents.profiler import dataset_name
from core import config
from core.sttm import approved_rows, read_sttm

NAME = "bronze_agent"
SYSTEM_PROMPT = """You are the Bronze agent in a retail sales medallion pipeline.
You ingest raw CSV files into Bronze Parquet exactly as the human-approved Bronze STTM says.
Call inspect_task_tool first to read the file previews and the approved rules, then call
bronze_ingestion_tool. Do not change values; Bronze keeps raw data. Report the output files."""


def ingest_bronze(files: list[str], sttm_path: str, run_id: str) -> list[Path]:
    sttm = read_sttm(sttm_path)
    rules = approved_rows(sttm, "bronze")
    out_dir = config.run_dir_for(config.BRONZE_DIR, run_id)
    outputs = []
    for dataset, table_rules in rules.groupby("target_table", sort=False):
        source = next((f for f in files if dataset_name(f) == dataset), None)
        if source is None:
            raise FileNotFoundError(f"No uploaded file matches dataset '{dataset}'")
        raw = pd.read_csv(source, dtype=str, keep_default_na=False)
        missing = [c for c in table_rules["source_column"] if c not in raw.columns]
        if missing:
            raise KeyError(f"'{Path(source).name}' is missing approved columns: {missing}")

        df = raw[list(table_rules["source_column"])].rename(
            columns=dict(zip(table_rules["source_column"], table_rules["target_column"]))
        )
        df["_source_file"] = Path(source).name
        df["_run_id"] = run_id
        df["_ingested_at"] = datetime.now(timezone.utc).isoformat()
        out = out_dir / f"{dataset}_bronze.parquet"
        df.to_parquet(out, index=False)
        outputs.append(out)
    return outputs


def deterministic(ctx: AgentContext) -> str:
    outputs = ingest_bronze(ctx.inputs["files"], ctx.inputs["sttm_bronze_path"], ctx.run_id)
    ctx.scratchpad["bronze_output_paths"] = [str(p) for p in outputs]
    ctx.audit.log("bronze_written", NAME, "success", files=[p.name for p in outputs])
    return f"Ingested {len(outputs)} table(s) to Bronze."


def tools(ctx: AgentContext) -> list:
    from langchain_core.tools import tool

    @tool
    def inspect_task_tool() -> str:
        """Preview each raw CSV (columns and row count) and list the approved Bronze rules."""
        sttm = read_sttm(ctx.inputs["sttm_bronze_path"])
        rules = approved_rows(sttm, "bronze")
        previews = {}
        for path in ctx.inputs["files"]:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=3)
            previews[dataset_name(path)] = {"columns": list(df.columns)}
        return json.dumps({
            "previews": previews,
            "rules": rules[["target_table", "source_column", "target_column"]].to_dict(orient="records"),
        })

    @tool
    def bronze_ingestion_tool() -> str:
        """Execute the approved Bronze STTM: write one Bronze Parquet file per table."""
        try:
            return deterministic(ctx)
        except Exception as exc:
            return f"ERROR: {exc}"

    return [inspect_task_tool, bronze_ingestion_tool]


def run(ctx: AgentContext, task: str) -> str:
    if ctx.llm is None:
        return deterministic(ctx)
    return run_react_agent(ctx, NAME, tools(ctx), SYSTEM_PROMPT, task)
