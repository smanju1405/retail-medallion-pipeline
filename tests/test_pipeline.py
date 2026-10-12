"""End-to-end tests for the deterministic pipeline, approval gates and resume.

These run without an LLM. The LLM path is covered by the wiring tests in test_agent_wiring.py.
"""

from pathlib import Path

import pandas as pd
import pytest

from agents import orchestrator
from agents.profiler import profile_files
from core import config
from core.sttm import approved_rows, read_sttm, write_sttm

SAMPLE = config.SAMPLE_DATA_DIR
RAW_FILES = [str(SAMPLE / f) for f in ("customers_raw.csv", "products_raw.csv", "sales_raw.csv")]


@pytest.fixture(scope="module")
def completed_run():
    return orchestrator.run_pipeline(RAW_FILES, "Which region drives revenue?", approve_fn=lambda s, l: None)


def test_sample_data_is_present():
    for name in ("customers_raw.csv", "products_raw.csv", "sales_raw.csv"):
        assert (SAMPLE / name).exists(), f"run python scripts/generate_sample_data.py ({name} missing)"


def test_profile_finds_join_keys_quality_issues_and_masks_pii():
    profile = profile_files(RAW_FILES)
    assert set(profile["datasets"]) == {"customers", "products", "sales"}
    assert "customer_id" in profile["join_keys"] and "product_id" in profile["join_keys"]
    assert profile["datasets"]["customers"]["columns"]["email"]["sample_values"] == ["***masked***"] * 3
    notes = " ".join(profile["quality_notes"])
    assert "negative values" in notes and "exact duplicate rows" in notes


def test_full_run_completes_and_writes_the_report(completed_run):
    assert completed_run.status == "complete"
    assert completed_run.errors == []
    assert Path(completed_run.report_path).exists()
    assert Path(completed_run.report_json_path).exists()
    html = Path(completed_run.report_path).read_text(encoding="utf-8")
    assert "Executive summary" in html and "Which region drives revenue?" in html


def test_silver_sales_matches_independent_cleansing(completed_run):
    """Expected rows = raw rows minus exact duplicates minus rows with a blank customer_id."""
    raw = pd.read_csv(SAMPLE / "sales_raw.csv", dtype=str, keep_default_na=False)
    expected = raw.drop_duplicates()
    expected = expected[expected["customer_id"].str.strip() != ""]

    silver = pd.read_parquet(next(Path(completed_run.silver_output_paths[0]).parent.glob("sales_silver.parquet")))
    assert len(silver) == len(expected)
    assert silver["customer_id"].notna().all()
    assert (silver["quantity"].dropna() >= 0).all()


def test_silver_has_no_lineage_columns_and_has_surrogate_key(completed_run):
    for path in completed_run.silver_output_paths:
        df = pd.read_parquet(path)
        assert not [c for c in df.columns if c.startswith("_")]
        assert df.columns[0].startswith("pk_") and df.columns[0].endswith("_silver_id")


def test_gold_totals_agree_across_tables(completed_run):
    """A left join to a dimension must not multiply revenue. Every gold revenue table sums to the same total."""
    gold = {Path(p).stem: pd.read_parquet(p) for p in completed_run.gold_output_paths}
    monthly = gold["sales_monthly"]["total_revenue"].sum()
    for table in ("revenue_by_region_month", "revenue_by_category", "revenue_by_customer_region"):
        assert gold[table]["total_revenue"].sum() == pytest.approx(monthly, rel=1e-9), table
    assert gold["sales_monthly"]["order_count"].sum() == len(
        pd.read_parquet(next(p for p in completed_run.silver_output_paths if "sales_silver" in p))
    )


def test_audit_log_records_every_phase(completed_run):
    lines = (config.AUDIT_DIR / f"audit_{completed_run.run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    events = [line for line in lines if '"phase_complete"' in line]
    assert len(events) == 4
    assert any('"human_approval"' in line for line in lines)


def test_run_pauses_at_first_gate_without_approve_fn():
    state = orchestrator.run_pipeline(RAW_FILES, "pause test")
    assert state.status == "awaiting_bronze_approval"
    assert state.sttm_bronze_path and Path(state.sttm_bronze_path).exists()


def test_cannot_run_a_phase_while_waiting_for_approval():
    state = orchestrator.run_pipeline(RAW_FILES, "gate test")
    with pytest.raises(ValueError, match="Approve the STTM"):
        orchestrator.run_next_phase(state)


def test_resume_continues_a_paused_run():
    paused = orchestrator.run_pipeline(RAW_FILES, "resume test")
    loaded = orchestrator.load_state(paused.run_id)
    assert loaded.status == "awaiting_bronze_approval"
    finished = orchestrator.resume_pipeline(loaded, approve_fn=lambda s, l: None)
    assert finished.status == "complete"


def test_approval_rejects_an_invalid_edit_and_keeps_the_gate_closed():
    state = orchestrator.run_pipeline(RAW_FILES, "invalid edit test")
    sttm = read_sttm(state.sttm_bronze_path)
    sttm.loc[0, "transformation"] = "__import__('os').system('x')"
    with pytest.raises(ValueError):
        orchestrator.approve(state, "bronze", edited_sttm=sttm)
    assert orchestrator.load_state(state.run_id).status == "awaiting_bronze_approval"


def test_human_can_exclude_a_column_and_the_pipeline_respects_it():
    state = orchestrator.run_pipeline(RAW_FILES, "exclude column test")
    sttm = read_sttm(state.sttm_bronze_path)
    sttm.loc[(sttm.target_table == "customers") & (sttm.source_column == "email"), "include"] = "N"
    state = orchestrator.approve(state, "bronze", edited_sttm=sttm)
    state = orchestrator.run_next_phase(state)  # phase 2: bronze ingest + silver STTM
    bronze = pd.read_parquet(next(p for p in state.bronze_output_paths if "customers" in p))
    assert "email" not in bronze.columns
    assert approved_rows(read_sttm(state.sttm_bronze_path), "bronze", "customers")["source_column"].tolist().count("email") == 0


def test_missing_gold_join_key_raises_a_clear_error(tmp_path):
    rules = pd.DataFrame([
        {"layer": "gold", "target_table": "t", "target_column": "x", "source_table": "sales",
         "source_column": "order_id", "transformation": "count", "join_on": "", "include": "Y"},
        {"layer": "gold", "target_table": "t", "target_column": "c", "source_table": "products",
         "source_column": "category", "transformation": "group_by", "join_on": "no_such_key", "include": "Y"},
    ])
    from core.sttm import build_gold_table

    silver = {"sales": pd.DataFrame({"order_id": ["O1"]}), "products": pd.DataFrame({"category": ["A"]})}
    with pytest.raises(ValueError, match="cannot join"):
        build_gold_table("t", rules, silver)
