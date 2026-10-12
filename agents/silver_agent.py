"""Agent 4 - Silver Agent: cleanses Bronze Parquet using the approved Silver STTM.

Operations are whitelisted rule names (see core/sttm.py), never free-form code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agents.base import AgentContext, run_react_agent
from core import config
from core.sttm import apply_silver_rules, approved_rows, read_sttm

NAME = "silver_agent"
SYSTEM_PROMPT = """You are the Silver agent in a retail sales medallion pipeline.
You cleanse Bronze data into Silver using the human-approved Silver STTM: type casting, null handling,
de-duplication, date standardisation and text normalisation. Call inspect_task_tool to read the Bronze
schemas and the approved rules, then call silver_ingestion_tool. Report the rows kept and dropped per table."""


def cleanse_silver(bronze_paths: list[str], sttm_path: str, run_id: str) -> list[Path]:
    sttm = read_sttm(sttm_path)
    out_dir = config.run_dir_for(config.SILVER_DIR, run_id)
    outputs = []
    for path in bronze_paths:
        dataset = Path(path).stem.rsplit("_", 1)[0]
        rules = approved_rows(sttm, "silver", dataset)
        if rules.empty:
            raise ValueError(f"No approved Silver rules for table '{dataset}'")
        bronze = pd.read_parquet(path)
        missing = [c for c in rules["source_column"] if c not in bronze.columns]
        if missing:
            raise KeyError(f"Bronze table '{dataset}' is missing columns: {missing}")
        silver = apply_silver_rules(bronze, rules, dataset)
        out = out_dir / f"{dataset}_silver.parquet"
        silver.to_parquet(out, index=False)
        outputs.append(out)
    return outputs


def deterministic(ctx: AgentContext) -> str:
    outputs = cleanse_silver(ctx.inputs["bronze_output_paths"], ctx.inputs["sttm_silver_path"], ctx.run_id)
    ctx.scratchpad["silver_output_paths"] = [str(p) for p in outputs]
    ctx.audit.log("silver_written", NAME, "success", files=[p.name for p in outputs])
    return f"Cleansed {len(outputs)} table(s) to Silver."


def tools(ctx: AgentContext) -> list:
    from langchain_core.tools import tool

    @tool
    def inspect_task_tool() -> str:
        """Show Bronze schemas, null counts and row counts, plus the approved Silver rules per column."""
        sttm = read_sttm(ctx.inputs["sttm_silver_path"])
        schemas = {}
        for path in ctx.inputs["bronze_output_paths"]:
            df = pd.read_parquet(path)
            schemas[Path(path).stem.rsplit("_", 1)[0]] = {
                "rows": len(df),
                "null_counts": {c: int(v) for c, v in df.isna().sum().items()},
            }
        rules = approved_rows(sttm, "silver")[["target_table", "source_column", "transformation"]]
        return json.dumps({"bronze": schemas, "rules": rules.to_dict(orient="records")})

    @tool
    def silver_ingestion_tool() -> str:
        """Execute the approved Silver STTM: cleanse each Bronze table and write Silver Parquet."""
        try:
            return deterministic(ctx)
        except Exception as exc:
            return f"ERROR: {exc}"

    return [inspect_task_tool, silver_ingestion_tool]


def run(ctx: AgentContext, task: str) -> str:
    if ctx.llm is None:
        return deterministic(ctx)
    return run_react_agent(ctx, NAME, tools(ctx), SYSTEM_PROMPT, task)
