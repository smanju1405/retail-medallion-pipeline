"""Source-to-Target Mapping (STTM): schema, validation and the rule engine.

An STTM is a CSV with one row per target column. The `transformation` cell is
a pipe-separated list of named operations, for example:

    strip_currency|to_float|nonnegative|fillna:median

Operations are looked up in a fixed whitelist. Nothing in an STTM is ever
passed to eval/exec, so a generated or edited rule cannot run arbitrary code.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

STTM_COLUMNS = [
    "layer", "target_table", "target_column", "source_table",
    "source_column", "transformation", "join_on", "include",
]
LAYERS = ("bronze", "silver", "gold")

BRONZE_OPS = {"as_string", "as_is"}
SILVER_OPS = {
    "trim", "lower", "upper", "title", "strip_currency", "to_int", "to_float",
    "to_date", "nonnegative", "fillna", "dropna", "dedupe",
}
GOLD_KEY_OPS = {"group_by"}
GOLD_AGG_OPS = {"sum", "avg", "count", "max", "min"}
GOLD_OPS = GOLD_KEY_OPS | GOLD_AGG_OPS | {"as_is"}
OPS_BY_LAYER = {"bronze": BRONZE_OPS, "silver": SILVER_OPS, "gold": GOLD_OPS}

DATE_FORMATS = ("%Y-%m-%d", "%d-%b-%Y", "%m/%d/%Y", "%Y/%m/%d")
_GOLD_AGG_FUNCS = {"sum": "sum", "avg": "mean", "count": "count", "max": "max", "min": "min"}


def parse_ops(transformation: str) -> list[tuple[str, str | None]]:
    """Split 'fillna:median|to_float' into [('fillna', 'median'), ('to_float', None)]."""
    ops = []
    for token in (transformation or "").split("|"):
        token = token.strip()
        if not token:
            continue
        name, _, arg = token.partition(":")
        ops.append((name.strip(), arg.strip() or None))
    return ops


def read_sttm(path: str | Path) -> pd.DataFrame:
    """Load and validate an STTM CSV. Raises ValueError listing every problem found."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in STTM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"STTM is missing columns: {missing}")

    df = df[STTM_COLUMNS].copy()
    df["include"] = df["include"].str.strip().str.upper().replace("", "Y")

    problems = []
    for idx, row in df.iterrows():
        line = idx + 2  # header is line 1
        if row["layer"] not in LAYERS:
            problems.append(f"line {line}: unknown layer '{row['layer']}'")
            continue
        if not row["target_table"] or not row["source_table"] or not row["source_column"]:
            problems.append(f"line {line}: target_table, source_table and source_column are required")
        if row["include"] not in {"Y", "N"}:
            problems.append(f"line {line}: include must be Y or N")
        for op, _ in parse_ops(row["transformation"]):
            if op not in OPS_BY_LAYER[row["layer"]]:
                problems.append(f"line {line}: '{op}' is not an allowed {row['layer']} operation")
    if problems:
        raise ValueError("Invalid STTM: " + "; ".join(problems))
    return df.reset_index(drop=True)


def write_sttm(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[STTM_COLUMNS].to_csv(path, index=False)
    return path


def approved_rows(sttm: pd.DataFrame, layer: str, table: str | None = None) -> pd.DataFrame:
    """Rows a human has approved (include == Y) for one layer, optionally one target table."""
    mask = (sttm["layer"] == layer) & (sttm["include"] == "Y")
    if table is not None:
        mask &= sttm["target_table"] == table
    return sttm[mask]


# --- Silver rule engine -------------------------------------------------------

def to_date(series: pd.Series) -> pd.Series:
    """Parse dates in any of DATE_FORMATS. Unparseable values become NaT."""
    cleaned = series.map(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() else None)
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        result = result.fillna(pd.to_datetime(cleaned, format=fmt, errors="coerce"))
    return result


def _map_str(series: pd.Series, fn) -> pd.Series:
    return series.map(lambda v: fn(v) if isinstance(v, str) else v)


def _fill_value(series: pd.Series, strategy: str | None):
    numeric = pd.to_numeric(series, errors="coerce")
    if strategy == "mean":
        return numeric.mean()
    if strategy == "median":
        return numeric.median()
    if strategy == "zero":
        return 0
    if strategy == "mode":
        modes = series.dropna().mode()
        return modes.iloc[0] if not modes.empty else "unknown"
    return strategy or "unknown"


def _apply_column_ops(df: pd.DataFrame, col: str, ops: list[tuple[str, str | None]]) -> pd.DataFrame:
    for op, arg in ops:
        if op == "trim":
            df[col] = _map_str(df[col], str.strip)
        elif op == "lower":
            df[col] = _map_str(df[col], str.lower)
        elif op == "upper":
            df[col] = _map_str(df[col], str.upper)
        elif op == "title":
            df[col] = _map_str(df[col], str.title)
        elif op == "strip_currency":
            df[col] = _map_str(df[col], lambda v: v.replace("$", "").replace(",", "").strip())
        elif op == "to_float":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        elif op == "to_int":
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
        elif op == "to_date":
            df[col] = to_date(df[col])
        elif op == "nonnegative":
            numeric = pd.to_numeric(df[col], errors="coerce")
            keep = numeric.isna() | (numeric >= 0).fillna(False).astype(bool)
            df[col] = numeric.where(keep)
        elif op == "fillna":
            df[col] = df[col].fillna(_fill_value(df[col], arg))
        elif op == "dropna":
            df = df[df[col].notna()].copy()
        # 'dedupe' is table-level and handled in apply_silver_rules
    return df


def apply_silver_rules(bronze: pd.DataFrame, rules: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Cleanse one bronze table according to its approved silver rules.

    Output keeps only the approved columns (renamed to target_column), drops
    exact duplicates if any rule asks for 'dedupe', and injects a surrogate key
    pk_<dataset>_silver_id as the first column.
    """
    df = bronze.copy()
    for col in rules["source_column"].unique():
        # blank and whitespace-only cells are missing values, so dropna/fillna can see them
        df[col] = df[col].map(lambda v: None if isinstance(v, str) and not v.strip() else v)
    dedupe = False
    for _, rule in rules.iterrows():
        ops = parse_ops(rule["transformation"])
        dedupe = dedupe or any(op == "dedupe" for op, _ in ops)
        df = _apply_column_ops(df, rule["source_column"], [o for o in ops if o[0] != "dedupe"])

    targets = list(dict.fromkeys(rules["target_column"]))
    df = df.rename(columns=dict(zip(rules["source_column"], rules["target_column"])))
    df = df[targets]
    if dedupe:
        df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df.insert(0, f"pk_{dataset}_silver_id", range(1, len(df) + 1))
    return df


# --- Gold rule engine ---------------------------------------------------------

def build_gold_table(target: str, rules: pd.DataFrame, silver: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Materialise one Gold table.

    The first source_table is the base. Any other source_table is left-joined on
    the join_on column. group_by rows define grouping keys (group_by:month turns
    a date into YYYY-MM). Aggregate rows (sum/avg/count/max/min) produce
    metrics. With no aggregates the table is a pass-through projection.
    """
    # Base rows are the ones without a join key; join rows always carry join_on.
    base_name = rules[rules["join_on"] == ""].iloc[0]["source_table"]
    base = silver[base_name].copy()

    for table in dict.fromkeys(rules["source_table"]):
        if table == base_name:
            continue
        table_rules = rules[rules["source_table"] == table]
        join_on = table_rules.iloc[0]["join_on"]
        if not join_on or join_on not in base.columns or join_on not in silver[table].columns:
            raise ValueError(f"Gold table '{target}': cannot join '{table}' to '{base_name}' on '{join_on}'")
        needed = list(dict.fromkeys(list(table_rules["source_column"]) + [join_on]))
        right = silver[table][needed].drop_duplicates(subset=[join_on])
        base = base.merge(right, on=join_on, how="left")

    keys, aggs, passthrough = [], [], []
    for _, rule in rules.iterrows():
        ops = parse_ops(rule["transformation"])
        op, arg = ops[0]
        if op == "group_by":
            values = base[rule["source_column"]]
            if arg == "month":
                values = pd.to_datetime(values).dt.strftime("%Y-%m")
            base[rule["target_column"]] = values.fillna("unknown") if values.dtype == object else values
            keys.append(rule["target_column"])
        elif op in GOLD_AGG_OPS:
            aggs.append((rule["target_column"], rule["source_column"], _GOLD_AGG_FUNCS[op]))
        else:
            passthrough.append((rule["target_column"], rule["source_column"]))

    if aggs:
        if keys:
            grouped = base.groupby(keys, dropna=False).agg(**{t: (s, f) for t, s, f in aggs}).reset_index()
        else:
            grouped = pd.DataFrame(
                [{t: base[s].agg(f) for t, s, f in aggs}]
            )
    else:
        grouped = base[[s for _, s in passthrough]].rename(columns=dict(passthrough))

    grouped = grouped.reset_index(drop=True)
    grouped.insert(0, f"pk_{target}_gold_id", range(1, len(grouped) + 1))
    return grouped
