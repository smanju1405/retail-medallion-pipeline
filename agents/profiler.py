"""Agent 1 - Data Profiler: builds a machine-readable profile of the raw CSVs.

Profile stats are computed locally. Only the profile reaches the LLM, never
full rows. PII columns (name, email, phone, address) are masked in samples.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agents.base import AgentContext, run_react_agent
from core import config
from core.sttm import to_date

NAME = "profiler_agent"
PII_PATTERN = re.compile(r"(name|email|phone|address)", re.I)
MONEY_PATTERN = re.compile(r"(price|amount|total|revenue|cost|value|discount)", re.I)
QTY_PATTERN = re.compile(r"(quantity|qty|units)", re.I)
CATEGORY_PATTERN = re.compile(r"(region|category|segment|status|type|channel)", re.I)

SYSTEM_PROMPT = """You are the Data Profiler agent in a retail sales medallion pipeline.
Your job is to understand the structure and quality of raw CSV files before any transformation.
Call inspect_files_tool first, then profiler_tool. Report what you found: tables, key columns,
join keys, and quality issues. Do not write transformation rules; that is the STTM agent's job."""


def dataset_name(path: str | Path) -> str:
    """sales_raw.csv -> 'sales'. Sanitised so it is safe as a table name."""
    stem = re.sub(r"\W+", "_", Path(str(path)).stem).strip("_").lower()
    return stem[:-4] if stem.endswith("_raw") else stem


def _read_raw(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _numeric_ratio(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    cleaned = values.str.replace(r"[$,\s]", "", regex=True)
    return float(pd.to_numeric(cleaned, errors="coerce").notna().mean())


def _semantic(name: str, numeric: float, dates: float, unique: int, rows: int) -> str:
    if name.lower() == "id" or name.lower().endswith("_id"):
        return "identifier"
    if PII_PATTERN.search(name):
        return "pii"
    if dates >= 0.9:
        return "date"
    if numeric >= 0.9 and MONEY_PATTERN.search(name):
        return "monetary"
    if numeric >= 0.9 and QTY_PATTERN.search(name):
        return "quantity"
    if numeric >= 0.9:
        return "measure"
    if CATEGORY_PATTERN.search(name) or (rows and unique <= 20):
        return "category"
    return "text"


def _column_stats(name: str, series: pd.Series) -> tuple[dict, list[str]]:
    """Stats for one column plus any quality notes it raised."""
    rows = len(series)
    missing = series.str.strip().eq("") | series.isna()
    present = series[~missing].str.strip()
    numeric = _numeric_ratio(present)
    date_parsed = to_date(present) if not present.empty else pd.Series(dtype="datetime64[ns]")
    dates = float(date_parsed.notna().mean()) if not present.empty else 0.0
    unique = int(present.nunique())
    semantic = _semantic(name, numeric, dates, unique, rows)

    samples = list(dict.fromkeys(present.head(20)))[:3]
    if semantic == "pii":
        samples = ["***masked***"] * len(samples)

    stats = {
        "semantic": semantic,
        "null_count": int(missing.sum()),
        "null_pct": round(100 * missing.mean(), 2) if rows else 0.0,
        "unique_count": unique,
        "sample_values": samples,
    }
    if numeric >= 0.9:
        nums = pd.to_numeric(present.str.replace(r"[$,\s]", "", regex=True), errors="coerce").dropna()
        if not nums.empty:
            stats.update(min=float(nums.min()), max=float(nums.max()), mean=round(float(nums.mean()), 2))
    if dates >= 0.9:
        stats["dtype_guess"] = "date"
    elif numeric >= 0.9:
        stats["dtype_guess"] = "numeric"
    else:
        stats["dtype_guess"] = "text"

    notes = []
    if stats["null_pct"] > 20:
        notes.append(f"{name}: {stats['null_pct']}% missing")
    if stats["dtype_guess"] == "numeric" and present.str.contains(r"[$,]", regex=True).any():
        notes.append(f"{name}: currency symbols or thousands separators in text")
    if stats["dtype_guess"] == "numeric" and "min" in stats and stats["min"] < 0:
        notes.append(f"{name}: negative values present")
    if 0 < dates < 1 and dates >= 0.5:
        notes.append(f"{name}: mixed or unparseable date formats")
    if semantic == "category":
        normalised = present.str.lower().str.strip().nunique()
        if normalised < unique:
            notes.append(f"{name}: inconsistent casing or whitespace ({unique} raw vs {normalised} normalised values)")
    return stats, notes


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Profile one table. Works on raw strings and on bronze parquet alike."""
    columns, notes = {}, []
    for col in df.columns:
        columns[col], col_notes = _column_stats(col, df[col].astype(str))
        notes.extend(col_notes)
    duplicates = int(df.duplicated().sum())
    if duplicates:
        notes.append(f"{duplicates} exact duplicate rows")
    return {"rows": int(len(df)), "columns": columns, "quality_notes": notes}


def profile_files(paths: list[str]) -> dict:
    datasets, all_columns = {}, {}
    for path in paths:
        name = dataset_name(path)
        datasets[name] = {"source_file": Path(path).name, **profile_dataframe(_read_raw(path))}
        for col in datasets[name]["columns"]:
            all_columns.setdefault(col, set()).add(name)
    join_keys = sorted(c for c, tables in all_columns.items() if c.endswith("_id") and len(tables) > 1)
    quality = [f"{name}: {note}" for name, d in datasets.items() for note in d["quality_notes"]]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "join_keys": join_keys,
        "quality_notes": quality,
    }


def write_profile(profile: dict, run_id: str) -> Path:
    config.PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = config.PROFILE_DIR / f"profile_combined_{stamp}_{run_id[:8]}.json"
    path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    return path


def _profile_now(ctx: AgentContext) -> str:
    profile = profile_files(ctx.inputs["files"])
    path = write_profile(profile, ctx.run_id)
    ctx.scratchpad["profile_path"] = str(path)
    ctx.audit.log("profile_written", NAME, "success", file=path.name, datasets=list(profile["datasets"]))
    return json.dumps(profile, default=str)


def deterministic(ctx: AgentContext) -> str:
    _profile_now(ctx)
    return f"Profiled {len(ctx.inputs['files'])} file(s)."


def tools(ctx: AgentContext) -> list:
    from langchain_core.tools import tool

    @tool
    def inspect_files_tool() -> str:
        """Preview each raw CSV: shape, column names, and 3 sample values per column (PII masked)."""
        preview = {}
        for path in ctx.inputs["files"]:
            df = _read_raw(path)
            sample = df.head(3).to_dict(orient="list")
            for col in sample:
                if PII_PATTERN.search(col):  # never send PII values to the model
                    sample[col] = ["***masked***"] * len(sample[col])
            preview[dataset_name(path)] = {"rows": len(df), "columns": list(df.columns), "sample_rows": sample}
        return json.dumps(preview, default=str)

    @tool
    def profiler_tool() -> str:
        """Compute full column statistics for all raw CSVs and save the profile. Returns the profile JSON."""
        try:
            return _profile_now(ctx)
        except Exception as exc:
            return f"ERROR: {exc}"

    return [inspect_files_tool, profiler_tool]


def run(ctx: AgentContext, task: str) -> str:
    if ctx.llm is None:
        return deterministic(ctx)
    return run_react_agent(ctx, NAME, tools(ctx), SYSTEM_PROMPT, task)
