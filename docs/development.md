# Development Guide

This guide walks through setting up the Author AI fact-checking app locally: a Flask backend (`backend/`) and a React + Vite frontend (`frontend/`).

## Prerequisites

Required:

| Tool | Version | Used for |
|------|---------|----------|
| Python | 3.9+ | Backend (Flask, pipelines). The code uses `list[...]` generics, which require 3.9. |
| Node.js + npm | 18+ | Frontend (Vite 5 requires Node 18+) |

Optional external tools — each **degrades gracefully** when absent (the related feature is skipped with a warning in the log; ingestion continues):

| Tool | Feature | Behavior when missing |
|------|---------|----------------------|
| `ocrmypdf` binary (needs Tesseract + Ghostscript installed) | OCR of scanned/low-text PDFs | OCR step fails inside a try/except; the original PDF text is used as-is |
| Java + Tabula jar (`tabula.jar`) | Table extraction from PDFs | Logs `Tabula jar not found ... skipping table extraction`; documents ingest with zero tables |
| `tesseract` (via `pytesseract` + OpenCV) | Chart OCR during chart ingestion | Chart extraction falls back to a skeleton / is skipped |

You also need two paid API accounts: **Anthropic** (chat, reranking, verdicts, explanations) and **OpenAI** (embeddings only). See [configuration.md](configuration.md) for every knob.

## Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # or use a Conda env if you prefer
pip install -r requirements.txt
cp .env.example .env             # then edit .env and fill in your keys
```

At minimum, set these in `backend/.env` (names only — never commit values):

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude models (`claude-sonnet-4-6` for chat/explanations/verdicts, `claude-haiku-4-5` for rerank/claim classification) |
| `OPENAI_API_KEY` | `text-embedding-3-large` embeddings |
| `API_KEY` | Shared secret for the `X-API-Key` header on all `/api/*` routes |

> **Warning:** if `API_KEY` is left unset, the `require_api_key` decorator in `app.py` becomes a no-op and authentication is **disabled entirely**. Fine for local dev, dangerous anywhere else.

### Running the backend

Run from inside `backend/` — relative storage paths (`./data`, `./logs`) resolve against the working directory:

```bash
cd backend
flask run --port 5001
# or equivalently:
python app.py                    # app.run(debug=True, port=5001)
```

Both paths load `backend/.env` automatically: `flask run` picks it up via python-dotenv, and `app.py` calls `load_dotenv` on its own `.env` at import time. No manual `export` needed (the old manual's `export FLASK_APP=...` steps are obsolete — `FLASK_APP=app.py` is already in `.env.example`).

Verify it is up:

```bash
curl http://localhost:5001/health                                  # no auth required
curl -H "X-API-Key: dev-api-key" http://localhost:5001/api/claims  # auth check
```

> **Note:** the old manual suggested `GET /api/dashboard` as a smoke test, but that endpoint returns `404 {"error": "No completed reports"}` on a fresh database. Use `/health` or `/api/claims` instead.

Storage is created automatically on first run: SQLite at `backend/data/accuracy.db` plus FAISS index directories under `backend/data/indexes/`. See [architecture.md](architecture.md).

## Frontend setup

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

The dev server runs at `http://localhost:5173` (pinned in `vite.config.ts`). Frontend environment variables:

| Variable | Default | Notes |
|----------|---------|-------|
| `VITE_API_BASE_URL` | `http://localhost:5001` | Where the Flask backend lives |
| `VITE_API_KEY` | `dev-api-key` | **Must match** the backend `API_KEY`, or every API call returns 401 |

Other scripts: `npm run build` (runs `tsc` then `vite build` — also your type-check, since there is no separate lint/test setup) and `npm run preview`.

## Trying it out

Sample PDFs live in `example_sources/` at the repo root:

| File | Role |
|------|------|
| `2024 Global Wolrd Hunger Index.pdf` | Source |
| `2025_world_hunger.pdf` | Source |
| `HH_Nov24-May25_FINAL.pdf` | Source |
| `disruptions_in_the_food_supply_chain.pdf` | Source |
| `World_Hunger_Fake.pdf` (also `.docx`) | **Report** — deliberately contains claims to fact-check |

In the UI: upload the source PDFs as sources, upload `World_Hunger_Fake.pdf` as the report, then run the pipeline. The run executes in a background thread; the UI polls `GET /api/jobs/<job_id>` for progress. Full endpoint reference: [api.md](api.md).

> **Gotcha:** the app is single-report, last-run-wins. Each pipeline run calls `reset_environment()` and **wipes all prior pipeline data** (claims, indexes, chat history) before starting. Don't expect to compare two runs side by side. See [history.md](history.md) for why.

> **Cost note:** a real pipeline run makes paid Anthropic and OpenAI API calls proportional to document size.

## Running tests

```bash
cd backend
python -m pytest
```

Test files in `backend/tests/`:

| File | Covers |
|------|--------|
| `test_api.py` | `/health`, and that `/api/*` requires `X-API-Key` |
| `test_accuracy_pipeline.py` | Claim extraction heuristics (heading filtering, numeric claims) and end-to-end retrieval with a faked ingestion step |
| `test_chart_pipeline.py` | Chart-to-chunk conversion, chart persistence, chart-aware verdict prompts (all with mocked LLM clients) |
| `test_chunking.py` | The `chunk_text` splitter (this file replaced the old `test_semantic_chunking.py`) |

`conftest.py` provides a `settings` fixture that redirects `DATA_ROOT`, `SQLITE_PATH`, `FAISS_INDEX_DIR`, `CACHE_DIR`, and the vector paths into pytest temp directories and sets `API_KEY=test-key`, so tests never touch your real `backend/data/`.

> **Warning:** `conftest.py` also loads `backend/.env` before tests run. The suite mocks LLM calls in most places, but if real `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` values are present, code paths that construct live clients can hit paid APIs. For a guaranteed-offline run, unset those variables or point the tests at a bare `.env`:

```bash
cd backend
env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY python -m pytest
```

There is no frontend test suite; `npm run build` is the closest thing to a frontend CI check.

## Troubleshooting

**Constant "Tabula jar not found" warnings / no tables extracted.** `TABULA_JAR_PATH` in `backend/.env` must be on a **single line**. A value wrapped across lines (easy to do when pasting) makes the path check in `table_extraction.py` fail on every ingest, silently disabling table extraction. Confirm the jar exists at the configured path, and that `java` is on your `PATH`.

**OCR never runs / OCR errors.** The `ocrmypdf` binary path comes from `OCR_MY_PDF_PATH` (default `/usr/local/bin/ocrmypdf`). Check `which ocrmypdf` and update the variable — Homebrew on Apple Silicon installs to `/opt/homebrew/bin/ocrmypdf`. `ocrmypdf` itself needs Tesseract and Ghostscript installed. OCR only triggers for PDFs with little or non-ASCII text (`should_run_ocr` in `services/ocr.py`), so most digital PDFs skip it by design. Set `PDF_OCR_ENABLED=false` to turn it off entirely.

**Port conflicts.** The backend defaults to 5001 (avoid 5000 — macOS AirPlay Receiver squats on it). If 5001 is taken, run `flask run --port <other>` and update `VITE_API_BASE_URL` in `frontend/.env` to match. The frontend port 5173 is pinned in `vite.config.ts`; change it there if needed.

**401 Unauthorized from the frontend.** `VITE_API_KEY` (frontend) and `API_KEY` (backend) do not match. Vite env vars are baked in at dev-server start, so restart `npm run dev` after editing `frontend/.env`.

**Where are the logs?** `backend/logs/backend.log` (path from `LOG_DIR`, default `./logs` relative to the working directory — another reason to run from `backend/`). Logs also stream to the console. Both `backend/logs/` and `backend/data/` are gitignored.

## Related docs

- [architecture.md](architecture.md) — pipelines, storage layout, reset-on-run design
- [api.md](api.md) — full HTTP endpoint reference
- [configuration.md](configuration.md) — every environment variable and default
- [metrics.md](metrics.md) — accuracy / credibility / validity scoring
- [chat.md](chat.md) — the chat service
- [history.md](history.md) — how the design evolved
