"""Agent 2 - STTM Agent: proposes Source-to-Target Mapping rules for one layer.

Bronze STTM comes from the profile, Silver STTM from bronze parquet, and Gold STTM
from silver parquet. The generators write a CSV that a human reviews before any
layer executes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agents.base import AgentContext, run_react_agent
from agents.profiler import dataset_name, profile_dataframe
from core import config
from core.sttm import STTM_COLUMNS, write_sttm

SYSTEM_PROMPT = """You are the STTM agent in a retail sales medallion pipeline.
You turn the data profile and the business intent into source-to-target mapping rules for one layer.
Call inspect_context_tool to read the context for the requested layer. Then call the generate tool for that
layer. Summarise the rules in plain language and flag anything that needs a human decision, such as
dropped rows, imputed values or joins."""

SALES, CUSTOMERS, PRODUCTS = "sales", "customers", "products"


def _row(layer, target_table, target_column, source_table, source_column, transformation, join_on=""):
    return {
        "layer": layer, "target_table": target_table, "target_column": target_column,
        "source_table": source_table, "source_column": source_column,
        "transformation": transformation, "join_on": join_on, "include": "Y",
    }


def _bronze_rules(profile: dict) -> pd.DataFrame:
    rows = []
    for dataset, meta in profile["datasets"].items():
        for col in meta["columns"]:
            rows.append(_row("bronze", dataset, col, dataset, col, "as_string"))
    return pd.DataFrame(rows, columns=STTM_COLUMNS)


def _silver_transformation(stats: dict, name: str) -> str:
    semantic = stats["semantic"]
    if semantic == "identifier":
        ops = ["trim", "dropna"]
    elif semantic == "date":
        ops = ["trim", "to_date"]
    elif semantic == "monetary":
        ops = ["strip_currency", "to_float", "fillna:median"]
    elif semantic == "quantity":
        ops = ["to_int", "nonnegative", "fillna:median"]
    elif semantic == "measure":
        ops = ["to_float", "fillna:median"]
    elif semantic == "pii":
        ops = ["trim", "lower"] if "email" in name.lower() else ["trim", "fillna:unknown"]
    else:  # category or free text
        ops = ["trim", "title", "fillna:unknown"]
    return "|".join(ops)


def _silver_rules(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, df in frames.items():
        profile = profile_dataframe(df)
        has_duplicates = any("duplicate rows" in n for n in profile["quality_notes"])
        business_cols = [(c, s) for c, s in profile["columns"].items() if not c.startswith("_")]  # skip lineage metadata
        for i, (col, stats) in enumerate(business_cols):
            transformation = _silver_transformation(stats, col)
            if i == 0 and has_duplicates:
                # 'dedupe' is table-level: the engine drops exact duplicate rows once per table
                transformation = f"{transformation}|dedupe"
            rows.append(_row("silver", dataset, col, dataset, col, transformation))
    return pd.DataFrame(rows, columns=STTM_COLUMNS)


def _gold_template(name, base, keys, aggs):
    """Rows for one gold table. keys: (target_col, source_table, source_col, transformation, join_on)."""
    rows = []
    for target_col, source_table, source_col, transformation, join_on in keys:
        rows.append(_row("gold", name, target_col, source_table, source_col, transformation, join_on))
    for target_col, source_col, transformation in aggs:
        rows.append(_row("gold", name, target_col, base, source_col, transformation))
    return rows


def _gold_rules(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Gold templates for the retail dataset. Only templates whose columns exist are emitted."""
    if SALES not in frames:
        return pd.DataFrame(columns=STTM_COLUMNS)

    def cols(table):
        return set(frames[table].columns) if table in frames else set()

    sales = cols(SALES)
    rows = []
    if {"order_date", "line_total"} <= sales:
        rows += _gold_template(
            "sales_monthly", SALES,
            keys=[("month", SALES, "order_date", "group_by:month", "")],
            aggs=[("total_revenue", "line_total", "sum"),
                  ("order_count", "order_id", "count"),
                  ("avg_line_value", "line_total", "avg")],
        )
    if {"order_date", "store_region", "line_total"} <= sales:
        rows += _gold_template(
            "revenue_by_region_month", SALES,
            keys=[("month", SALES, "order_date", "group_by:month", ""),
                  ("region", SALES, "store_region", "group_by", "")],
            aggs=[("total_revenue", "line_total", "sum")],
        )
    if {"product_id", "line_total", "quantity"} <= sales and "category" in cols(PRODUCTS):
        rows += _gold_template(
            "revenue_by_category", SALES,
            keys=[("category", PRODUCTS, "category", "group_by", "product_id")],
            aggs=[("total_revenue", "line_total", "sum"), ("units_sold", "quantity", "sum")],
        )
    if {"customer_id", "line_total"} <= sales and "region" in cols(CUSTOMERS):
        rows += _gold_template(
            "revenue_by_customer_region", SALES,
            keys=[("customer_region", CUSTOMERS, "region", "group_by", "customer_id")],
            aggs=[("total_revenue", "line_total", "sum")],
        )
    return pd.DataFrame(rows, columns=STTM_COLUMNS)


def _load_context(ctx: AgentContext, layer: str):
    if layer == "bronze":
        with open(ctx.inputs["profile_path"], encoding="utf-8") as fh:
            return json.load(fh)
    paths = ctx.inputs["bronze_output_paths"] if layer == "silver" else ctx.inputs["silver_output_paths"]
    # file names are '<dataset>_bronze.parquet' or '<dataset>_silver.parquet'
    return {Path(p).stem.rsplit("_", 1)[0]: pd.read_parquet(p) for p in paths}


def generate(ctx: AgentContext, layer: str) -> Path:
    context = _load_context(ctx, layer)
    if layer == "bronze":
        rules = _bronze_rules(context)
    elif layer == "silver":
        rules = _silver_rules(context)
    else:
        rules = _gold_rules(context)
    if rules.empty:
        raise ValueError(f"No {layer} STTM rules could be generated from the available data")

    path = write_sttm(rules, config.STTM_DIR / f"sttm_{layer}_{ctx.run_id}.csv")
    ctx.scratchpad[f"sttm_{layer}_path"] = str(path)
    ctx.audit.log("sttm_written", f"sttm_agent_{layer}", "success", file=path.name, rules=len(rules))
    return path


def deterministic(ctx: AgentContext, layer: str) -> str:
    path = generate(ctx, layer)
    return f"Generated {layer} STTM: {path.name}"


def tools(ctx: AgentContext, layer: str) -> list:
    from langchain_core.tools import tool

    @tool
    def inspect_context_tool() -> str:
        """Show the context for this layer: profile datasets and columns (bronze), or bronze/silver schemas (silver, gold)."""
        context = _load_context(ctx, layer)
        if layer == "bronze":
            summary = {d: {"columns": list(m["columns"]), "rows": m["rows"]} for d, m in context["datasets"].items()}
            return json.dumps({"join_keys": context["join_keys"], "datasets": summary}, default=str)
        return json.dumps({t: {"columns": list(df.columns), "rows": len(df)} for t, df in context.items()}, default=str)

    @tool
    def generate_sttm_tool() -> str:
        """Generate the STTM CSV for this layer and save it for human review. Returns the file name."""
        try:
            return deterministic(ctx, layer)
        except Exception as exc:
            return f"ERROR: {exc}"

    return [inspect_context_tool, generate_sttm_tool]


def run(ctx: AgentContext, task: str, layer: str) -> str:
    if ctx.llm is None:
        return deterministic(ctx, layer)
    return run_react_agent(ctx, f"sttm_agent_{layer}", tools(ctx, layer), SYSTEM_PROMPT, task)
