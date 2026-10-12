# CLAUDE.md: retail medallion pipeline

Project context for Claude Code (including Claude in VS Code). Read this before changing code.

## What this is
An agentic Medallion pipeline (Bronze, Silver, Gold, Report) for messy retail sales CSVs.
A Supervisor orchestrates six sub-agents. A human approves each STTM (source-to-target mapping) before the layer runs.
Full design is in README.md. Architecture follows the "Accelerate with AI" training deck.

## Layout
- `agents/orchestrator.py`: Supervisor, PipelineState, phases, approval gates, output verification. Start here.
- `agents/base.py`: AgentContext and `run_react_agent` (LangChain `create_agent`, ReAct loop, traces).
- `agents/*.py`: one module per agent. Each exposes `run(ctx, task)`, `tools(ctx)` and `deterministic(ctx)`.
- `core/sttm.py`: STTM schema, validation, and the Silver/Gold rule engine. Operations are a whitelist.
- `core/config.py`: paths and settings. `core/llm.py`: chat model factory (Groq or Google).
- `core/audit.py`, `core/observability.py`, `core/memory.py`: audit log, agent traces, document store.
- `streamlit_app.py`: UI with approval gates. `cli.py`: command-line runner.
- `tests/`: run with `python -m pytest -q`. Tests need no API key and make no network calls.

## Commands
- Install: `pip install -r requirements.txt`
- Regenerate synthetic sample data: `python scripts/generate_sample_data.py`
- Offline run (no LLM): `python cli.py --deterministic --auto-approve`
- Agentic run: `python cli.py` (needs `.env` with a key; pauses at each gate)
- UI: `streamlit run streamlit_app.py`

## Rules when editing
- Never use `eval`, `exec` or `pickle` on STTM content or LLM output. Add new transformations as named ops in
  `core/sttm.py`, add them to the whitelist, and add a test.
- Pass tool outputs through the Supervisor verification step (`_verify`). Do not trust a success message alone.
- The LLM sees profile statistics and schema only. Do not send full rows or unmasked PII to a model.
- Reporter charts come from the fixed `CHARTS` list. Do not let the LLM choose chart SQL.
- SQL the LLM runs goes through `validate_readonly_sql`. Keep it read-only.
- Audit logs and traces record file names and counts only, never data values.
- Keep keys in `.env`. Do not hard-code them or commit `.env`.
- Run `python -m pytest -q` after any change to `core/` or `agents/`.

## Known limits
- Outputs go under `data/<layer>/<run_id>/`. Rerunning never overwrites a previous run.
- The document store is JSON files, not ChromaDB. Callers use `core/memory.py` functions, so swapping is local.
- Sample data is synthetic. Do not commit real Retail Co or client data.
