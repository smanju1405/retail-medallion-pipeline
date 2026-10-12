"""Agent 5 - Gold Agent: materialises analytics tables from Silver using the approved Gold STTM.

Joins, grouping keys and aggregates come from the approved rules (see build_gold_table).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agents.base import AgentContext, run_react_agent
from core import config
from core.sttm import approved_rows, build_gold_table, read_sttm

NAME = "gold_agent"
SYSTEM_PROMPT = """You are the Gold agent in a retail sales medallion pipeline.
You materialise business-ready aggregate tables from Silver using the human-approved Gold STTM.
Call inspect_task_tool to read the Silver schemas and the approved rules, then call gold_ingestion_tool.
Report each Gold table you produced and its row count."""


def materialise_gold(silver_paths: list[str], sttm_path: str, run_id: str) -> list[Path]:
    sttm = read_sttm(sttm_path)
    silver = {Path(p).stem.rsplit("_", 1)[0]: pd.read_parquet(p) for p in silver_paths}
    out_dir = config.run_dir_for(config.GOLD_DIR, run_id)
    outputs = []
    for target, rules in approved_rows(sttm, "gold").groupby("target_table", sort=False):
        table = build_gold_table(target, rules, silver)
        out = out_dir / f"{target}.parquet"
        table.to_parquet(out, index=False)
        outputs.append(out)
    return outputs


def deterministic(ctx: AgentContext) -> str:
    outputs = materialise_gold(ctx.inputs["silver_output_paths"], ctx.inputs["sttm_gold_path"], ctx.run_id)
    ctx.scratchpad["gold_output_paths"] = [str(p) for p in outputs]
    ctx.audit.log("gold_written", NAME, "success", files=[p.name for p in outputs])
    return f"Materialised {len(outputs)} Gold table(s)."


def tools(ctx: AgentContext) -> list:
    from langchain_core.tools import tool

    @tool
    def inspect_task_tool() -> str:
        """Show Silver schemas and row counts, plus the approved Gold rules."""
        sttm = read_sttm(ctx.inputs["sttm_gold_path"])
        schemas = {}
        for path in ctx.inputs["silver_output_paths"]:
            df = pd.read_parquet(path)
            schemas[Path(path).stem.rsplit("_", 1)[0]] = {"rows": len(df), "columns": list(df.columns)}
        rules = approved_rows(sttm, "gold")[
            ["target_table", "target_column", "source_table", "source_column", "transformation", "join_on"]
        ]
        return json.dumps({"silver": schemas, "rules": rules.to_dict(orient="records")})

    @tool
    def gold_ingestion_tool() -> str:
        """Execute the approved Gold STTM: join, group and aggregate Silver into Gold Parquet tables."""
        try:
            return deterministic(ctx)
        except Exception as exc:
            return f"ERROR: {exc}"

    return [inspect_task_tool, gold_ingestion_tool]


def run(ctx: AgentContext, task: str) -> str:
    if ctx.llm is None:
        return deterministic(ctx)
    return run_react_agent(ctx, NAME, tools(ctx), SYSTEM_PROMPT, task)
