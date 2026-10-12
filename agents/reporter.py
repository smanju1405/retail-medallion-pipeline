"""Agent 6 - Reporter Agent: answers the business question from Gold tables.

Two parts, kept separate on purpose:
  * Charts come from fixed, reviewed SQL in CHARTS. The LLM never chooses them.
  * The narrative comes from the LLM (or from a deterministic summary in offline mode).
    The LLM may run read-only SQL against Gold through execute_query_tool.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
from plotly.offline import get_plotlyjs

from agents.base import AgentContext, run_react_agent
from core import config
from core.memory import store_document

NAME = "reporter_agent"
SYSTEM_PROMPT = """You are the Reporter agent in a retail sales medallion pipeline.
Answer the business question using only the Gold tables. Call inspect_gold_tables_tool, then
execute_query_tool with read-only SELECT statements to check the numbers you will quote.
Finish with an executive summary of 3 to 5 sentences. Quote figures exactly as the queries return them.
Note any data-quality caveat that could change the conclusion, such as unknown or unmatched values."""

# Fixed, reviewed chart queries. Every chart name must match a Gold table name.
CHARTS = [
    {"title": "Monthly revenue trend", "type": "line", "table": "sales_monthly",
     "sql": "SELECT month, total_revenue FROM sales_monthly ORDER BY month", "x": "month", "y": "total_revenue"},
    {"title": "Revenue by customer region", "type": "bar", "table": "revenue_by_customer_region",
     "sql": "SELECT customer_region, SUM(total_revenue) AS total_revenue FROM revenue_by_customer_region "
            "GROUP BY 1 ORDER BY 2 DESC", "x": "customer_region", "y": "total_revenue"},
    {"title": "Revenue share by product category", "type": "pie", "table": "revenue_by_category",
     "sql": "SELECT category, SUM(total_revenue) AS total_revenue FROM revenue_by_category "
            "GROUP BY 1 ORDER BY 2 DESC", "x": "category", "y": "total_revenue"},
    {"title": "Orders vs revenue by month", "type": "scatter", "table": "sales_monthly",
     "sql": "SELECT month, order_count, total_revenue FROM sales_monthly ORDER BY month",
     "x": "order_count", "y": "total_revenue"},
]

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|pragma|install|load|export|"
    r"import|call|grant|truncate|replace|set)\b",
    re.I,
)
_TABLE_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")


def validate_readonly_sql(sql: str) -> str:
    """Allow one SELECT or WITH statement only. Raises ValueError otherwise."""
    statement = sql.strip().rstrip(";").strip()
    if ";" in statement:
        raise ValueError("Only one SQL statement is allowed")
    if not re.match(r"^(select|with)\b", statement, re.I):
        raise ValueError("Only SELECT queries are allowed")
    if _FORBIDDEN.search(statement):
        raise ValueError("Query contains a non-read-only keyword")
    return statement


def load_gold(paths: list[str]) -> duckdb.DuckDBPyConnection:
    """Register each Gold Parquet file as an in-memory DuckDB table named after the file."""
    con = duckdb.connect(":memory:")
    for path in paths:
        name = Path(path).stem
        if not _TABLE_NAME.match(name):
            raise ValueError(f"Gold table name '{name}' is not a valid identifier")
        escaped = str(path).replace("'", "''")
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{escaped}')")
    return con


def _tables(con) -> list[str]:
    return [row[0] for row in con.execute("SHOW TABLES").fetchall()]


def catalog(con) -> dict:
    out = {}
    for table in _tables(con):
        cols = con.execute(f"DESCRIBE {table}").fetchdf()
        rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        out[table] = {"rows": int(rows), "columns": {r["column_name"]: r["column_type"] for _, r in cols.iterrows()}}
    return out


def run_query(con, sql: str, limit: int = config.MAX_QUERY_ROWS) -> pd.DataFrame:
    return con.execute(sql).fetchdf().head(limit)


def deterministic_summary(con) -> str:
    """Offline summary built from fixed numbers. Used when no LLM is configured."""
    sentences = []
    tables = _tables(con)
    if "sales_monthly" in tables:
        totals = run_query(con, "SELECT SUM(total_revenue) AS revenue, SUM(order_count) AS orders FROM sales_monthly")
        peak = run_query(con, "SELECT month, total_revenue FROM sales_monthly ORDER BY total_revenue DESC LIMIT 1")
        sentences.append(
            f"Total revenue across the period is {totals.iloc[0]['revenue']:,.2f} from "
            f"{int(totals.iloc[0]['orders'])} order lines."
        )
        if not peak.empty:
            sentences.append(
                f"The strongest month is {peak.iloc[0]['month']} with revenue of {peak.iloc[0]['total_revenue']:,.2f}."
            )
    if "revenue_by_customer_region" in tables:
        top = run_query(con, CHARTS[1]["sql"]).head(1)
        if not top.empty:
            sentences.append(f"{top.iloc[0]['customer_region']} is the largest customer region by revenue.")
    if "revenue_by_category" in tables:
        top = run_query(con, CHARTS[2]["sql"]).head(1)
        if not top.empty:
            sentences.append(f"{top.iloc[0]['category']} is the largest product category by revenue.")
    # Surface revenue that could not be attributed, instead of always printing a caveat
    for table, column, label in (("revenue_by_customer_region", "customer_region", "customer region"),
                                 ("revenue_by_category", "category", "product category")):
        if table in tables:
            unknown = run_query(
                con, f"SELECT COALESCE(SUM(total_revenue), 0) AS v FROM {table} WHERE {column} = 'unknown'"
            ).iloc[0]["v"]
            if unknown > 0:
                sentences.append(
                    f"{unknown:,.2f} of revenue has no matching {label} and is shown as 'unknown'; review before use."
                )
    return " ".join(sentences)


def _chart_html(con, spec: dict) -> str | None:
    if spec["table"] not in _tables(con):
        return None
    df = run_query(con, spec["sql"])
    if df.empty:
        return None
    if spec["type"] == "line":
        fig = px.line(df, x=spec["x"], y=spec["y"], title=spec["title"], markers=True)
    elif spec["type"] == "bar":
        fig = px.bar(df, x=spec["x"], y=spec["y"], title=spec["title"])
    elif spec["type"] == "pie":
        fig = px.pie(df, names=spec["x"], values=spec["y"], title=spec["title"])
    else:
        fig = px.scatter(df, x=spec["x"], y=spec["y"], title=spec["title"])
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_report(con, business_intent: str, summary: str, run_id: str) -> tuple[Path, Path]:
    """Write report_<run_id>.html (charts inline, works offline) and report_<run_id>.json."""
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    charts, chart_meta = [], []
    for spec in CHARTS:
        fragment = _chart_html(con, spec)
        if fragment:
            charts.append(f"<section class='card'>{fragment}</section>")
            chart_meta.append({"title": spec["title"], "type": spec["type"], "source_table": spec["table"]})

    paragraphs = "".join(f"<p>{html.escape(p)}</p>" for p in summary.split("\n") if p.strip())
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retail sales report {html.escape(run_id[:8])}</title>
<script>{get_plotlyjs()}</script>
<style>
  body {{ font-family: Segoe UI, Arial, sans-serif; margin: 0; padding: 16px; background: #f5f7f5; color: #1d2b20; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 8px; }}
  .question {{ color: #3b5b43; font-style: italic; margin-bottom: 16px; }}
  .card {{ background: #fff; border-radius: 8px; padding: 12px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 12px; }}
</style></head>
<body>
<h1>Retail sales report</h1>
<div class="question">Business question: {html.escape(business_intent or 'not provided')}</div>
<section class="card"><h2>Executive summary</h2>{paragraphs}</section>
<div class="grid">{''.join(charts)}</div>
</body></html>"""

    html_path = config.REPORT_DIR / f"report_{run_id}.html"
    json_path = config.REPORT_DIR / f"report_{run_id}.json"
    html_path.write_text(page, encoding="utf-8")
    json_path.write_text(
        json.dumps({"run_id": run_id, "business_intent": business_intent, "summary": summary,
                    "charts": chart_meta, "gold_tables": catalog(con)}, indent=2, default=str),
        encoding="utf-8",
    )
    return html_path, json_path


def tools(ctx: AgentContext, con) -> list:
    from langchain_core.tools import tool

    @tool
    def inspect_gold_tables_tool() -> str:
        """Preview Gold tables: schema, row count and 3 sample rows each."""
        preview = {}
        for table in _tables(con):
            sample = run_query(con, f"SELECT * FROM {table} LIMIT 3").to_dict(orient="records")
            preview[table] = {"catalog": catalog(con)[table], "sample": sample}
        return json.dumps(preview, default=str)

    @tool
    def load_gold_data_tool() -> str:
        """Return the full Gold catalog with exact column names, types and row counts."""
        return json.dumps(catalog(con), default=str)

    @tool
    def execute_query_tool(sql: str) -> str:
        """Run one read-only DuckDB SELECT against the Gold tables. Returns up to 50 rows as JSON."""
        try:
            df = run_query(con, validate_readonly_sql(sql), limit=50)
            return df.to_json(orient="records", default_handler=str)
        except Exception as exc:
            return f"ERROR: {exc}"

    return [inspect_gold_tables_tool, load_gold_data_tool, execute_query_tool]


def run(ctx: AgentContext, task: str) -> str:
    paths = ctx.inputs["gold_output_paths"]
    con = load_gold(paths)
    intent = ctx.inputs.get("business_intent", "")
    if ctx.llm is None:
        summary = deterministic_summary(con)
    else:
        summary = run_react_agent(ctx, NAME, tools(ctx, con), SYSTEM_PROMPT, task)
        if not summary:
            summary = deterministic_summary(con)

    html_path, json_path = build_report(con, intent, summary, ctx.run_id)
    store_document(ctx.run_id, "final_report", summary, {"report": html_path.name})
    ctx.scratchpad.update(report_path=str(html_path), report_json_path=str(json_path), summary=summary)
    ctx.audit.log("report_written", NAME, "success", file=html_path.name)
    return f"Report written: {html_path.name}"
