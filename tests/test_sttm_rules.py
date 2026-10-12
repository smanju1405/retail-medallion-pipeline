"""Unit tests for the STTM rule engine and the read-only SQL guard."""

import pandas as pd
import pytest

from agents.reporter import validate_readonly_sql
from core.sttm import STTM_COLUMNS, apply_silver_rules, parse_ops, read_sttm, to_date


def _rule(target, source, transformation, layer="silver", table="sales", join_on=""):
    return {"layer": layer, "target_table": table, "target_column": target, "source_table": table,
            "source_column": source, "transformation": transformation, "join_on": join_on, "include": "Y"}


def _write_sttm(tmp_path, rows):
    path = tmp_path / "sttm.csv"
    pd.DataFrame(rows, columns=STTM_COLUMNS).to_csv(path, index=False)
    return path


def test_parse_ops_splits_names_and_arguments():
    assert parse_ops("strip_currency|to_float|fillna:median") == [
        ("strip_currency", None), ("to_float", None), ("fillna", "median"),
    ]


def test_read_sttm_rejects_operations_outside_the_whitelist(tmp_path):
    path = _write_sttm(tmp_path, [_rule("x", "x", "__import__('os').system('echo hi')")])
    with pytest.raises(ValueError, match="not an allowed silver operation"):
        read_sttm(path)


def test_read_sttm_rejects_unknown_layer_and_bad_include(tmp_path):
    bad_layer = _rule("x", "x", "trim", layer="platinum")
    with pytest.raises(ValueError, match="unknown layer"):
        read_sttm(_write_sttm(tmp_path, [bad_layer]))

    bad_include = {**_rule("x", "x", "trim"), "include": "maybe"}
    with pytest.raises(ValueError, match="include must be Y or N"):
        read_sttm(_write_sttm(tmp_path, [bad_include]))


def test_to_date_reads_each_supported_format():
    parsed = to_date(pd.Series(["2026-03-05", "05-Mar-2026", "03/05/2026", "not a date", None]))
    assert parsed.iloc[0] == pd.Timestamp("2026-03-05")
    assert parsed.iloc[1] == pd.Timestamp("2026-03-05")
    assert parsed.iloc[2] == pd.Timestamp("2026-03-05")
    assert pd.isna(parsed.iloc[3]) and pd.isna(parsed.iloc[4])


def test_silver_treats_blank_cells_as_missing_and_drops_them():
    bronze = pd.DataFrame({"order_id": ["O1", "O2", "O3"], "customer_id": ["C1", "", "C3"]})
    rules = pd.DataFrame([_rule("order_id", "order_id", "trim"), _rule("customer_id", "customer_id", "trim|dropna")])
    out = apply_silver_rules(bronze, rules, "sales")
    assert out["order_id"].tolist() == ["O1", "O3"]
    assert out.columns[0] == "pk_sales_silver_id"


def test_silver_dedupe_removes_exact_duplicates_and_keeps_only_approved_columns():
    bronze = pd.DataFrame({"order_id": ["O1", "O1"], "secret": ["x", "x"], "_run_id": ["r", "r"]})
    rules = pd.DataFrame([_rule("order_id", "order_id", "trim|dedupe")])
    out = apply_silver_rules(bronze, rules, "sales")
    assert len(out) == 1
    assert "secret" not in out.columns and "_run_id" not in out.columns


def test_silver_nonnegative_turns_negative_quantities_into_missing_values():
    bronze = pd.DataFrame({"quantity": ["3", "-2", "two", ""]})
    rules = pd.DataFrame([_rule("quantity", "quantity", "to_int|nonnegative|fillna:zero")])
    out = apply_silver_rules(bronze, rules, "sales")
    assert out["quantity"].tolist() == [3, 0, 0, 0]


def test_silver_strip_currency_then_float():
    bronze = pd.DataFrame({"line_total": ["$1,234.50", "12.00", ""]})
    rules = pd.DataFrame([_rule("line_total", "line_total", "strip_currency|to_float|fillna:median")])
    out = apply_silver_rules(bronze, rules, "sales")
    assert out["line_total"].tolist() == pytest.approx([1234.50, 12.00, 623.25])


@pytest.mark.parametrize("sql", [
    "SELECT 1; DROP TABLE sales_monthly",
    "DELETE FROM sales_monthly",
    "WITH x AS (SELECT 1) INSERT INTO sales_monthly SELECT * FROM x",
    "COPY sales_monthly TO '/tmp/out.csv'",
    "PRAGMA database_list",
])
def test_reporter_sql_guard_blocks_writes_and_multiple_statements(sql):
    with pytest.raises(ValueError):
        validate_readonly_sql(sql)


def test_reporter_sql_guard_allows_select_and_with():
    assert validate_readonly_sql("  SELECT month, total_revenue FROM sales_monthly;  ").startswith("SELECT")
    assert validate_readonly_sql("WITH t AS (SELECT 1 AS a) SELECT a FROM t")
