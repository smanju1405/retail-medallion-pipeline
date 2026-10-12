"""Streamlit UI: upload data, review each STTM at its gate, watch phases run, read the report."""

import glob
import json
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from agents.orchestrator import PipelineState, approve, create_run, layer_awaiting_approval, load_state, run_next_phase
from core import config
from core.sttm import read_sttm

DEFAULT_INTENT = "Which customer region and product category drive the most revenue, and how does revenue trend by month?"
LAYER_LABEL = {"bronze": "Bronze", "silver": "Silver", "gold": "Gold"}

st.set_page_config(page_title="Retail medallion pipeline", layout="wide")
config.ensure_dirs()


def _llm(agentic: bool):
    if not agentic:
        return None
    from core.llm import get_llm

    return get_llm()


def _advance(state: PipelineState, agentic: bool) -> PipelineState:
    """Run the next phase, showing progress. Errors are shown, not raised."""
    with st.spinner("Running phase. Agents are working..." if agentic else "Running phase..."):
        return run_next_phase(state, _llm(agentic))


def _save_uploads(uploads) -> list[str]:
    folder = config.UPLOAD_DIR / str(uuid.uuid4())
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for upload in uploads:
        target = folder / Path(upload.name).name
        target.write_bytes(upload.getbuffer())
        paths.append(str(target))
    return paths


st.title("Retail medallion pipeline")
st.caption("Agentic Bronze, Silver and Gold cleansing with human approval at every STTM gate.")

with st.sidebar:
    st.header("Settings")
    agentic = st.radio("Execution mode", ["Agentic (LLM agents)", "Deterministic (no LLM)"], index=0) == "Agentic (LLM agents)"
    st.caption(f"LLM provider: {config.LLM_PROVIDER}. Keys are read from .env.")
    st.caption("Only profile statistics and schema reach the LLM, never full rows. PII is masked.")
    open_id = st.text_input("Open an existing run by ID")
    if st.button("Open run") and open_id.strip():
        try:
            st.session_state.run_id = load_state(open_id.strip()).run_id
        except FileNotFoundError:
            st.error("No run with that ID.")
    if st.button("Start a new run"):
        st.session_state.run_id = None
        st.rerun()

if "run_id" not in st.session_state:
    st.session_state.run_id = None

# ---- Start a run ----------------------------------------------------------------
if st.session_state.run_id is None:
    st.subheader("1. Start a run")
    uploads = st.file_uploader("Raw CSV files", type="csv", accept_multiple_files=True)
    use_sample = st.checkbox("Use the bundled sample data", value=not uploads)
    intent = st.text_area("Business question for the report", value=DEFAULT_INTENT, height=80)
    if st.button("Start run", type="primary"):
        files = sorted(glob.glob(str(config.SAMPLE_DATA_DIR / "*_raw.csv"))) if use_sample or not uploads else _save_uploads(uploads)
        if not files:
            st.error("Upload at least one CSV, or use the sample data.")
        else:
            try:
                state = create_run(files, intent)
                st.session_state.run_id = state.run_id
                _advance(state, agentic)
                st.rerun()
            except RuntimeError as exc:  # missing API key
                st.error(str(exc))
    st.stop()

# ---- Show the current run -------------------------------------------------------
state = load_state(st.session_state.run_id)
st.subheader(f"Run {state.run_id[:8]}")
st.write(f"**Status:** `{state.status}`")
st.write(f"**Business question:** {state.business_intent or 'not provided'}")

if state.errors:
    st.error(" ".join(state.errors))
    if st.button("Retry this phase"):
        try:
            _advance(state, agentic)
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))

layer = layer_awaiting_approval(state)
if layer:
    st.subheader(f"2. Review the {LAYER_LABEL[layer]} STTM")
    st.info("Edit the include column (Y/N) to drop a column or rule. Transformations must use the allowed "
            "operation names. Approving validates the file, then runs the next phase.")
    if layer == "bronze" and state.profile_path:
        with st.expander("Data profile (what the profiler found)"):
            st.json(json.loads(Path(state.profile_path).read_text(encoding="utf-8")))
    sttm_path = getattr(state, f"sttm_{layer}_path")
    sttm = read_sttm(sttm_path)
    edited = st.data_editor(
        sttm,
        hide_index=True,
        key=f"editor_{layer}_{state.run_id}",
        column_config={"include": st.column_config.SelectboxColumn("include", options=["Y", "N"], required=True)},
        disabled=["layer", "target_table", "source_table", "source_column"],
    )
    if st.button(f"Approve {LAYER_LABEL[layer]} STTM and continue", type="primary"):
        try:
            state = approve(state, layer, edited)
            _advance(state, agentic)
            st.rerun()
        except ValueError as exc:
            st.error(f"Not approved: {exc}")

elif state.status == "complete":
    st.subheader("2. Report")
    st.success("Pipeline complete.")
    st.write(state.summary)
    report = Path(state.report_path).read_text(encoding="utf-8")
    st.download_button("Download report (HTML)", report, file_name=Path(state.report_path).name, mime="text/html")
    components.html(report, height=1100, scrolling=True)

elif not state.errors:
    st.info("Ready to run the next phase.")
    if st.button("Run next phase", type="primary"):
        _advance(state, agentic)
        st.rerun()
