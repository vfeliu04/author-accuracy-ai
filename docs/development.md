# Development Guide

Working on the v2 codebase: FastAPI backend in `backend/authorai/`, tests in `backend/tests/`, eval sets in `backend/evals/`, React + Vite frontend in `frontend/`.

## Environment

The backend runs in the **`author_ai` conda env** — not uv, not a venv:

```bash
conda activate author_ai
cd backend
pip install -e ".[dev]"        # after any dependency change (deps are pinned in pyproject.toml)
```

Python ≥ 3.11 (CI uses 3.11). Notable pins in `pyproject.toml`: `torch`/`torchvision` are declared explicitly because Docling's default layout engine needs them at runtime but does not declare them; `python-multipart` is required by FastAPI for the upload endpoint; `trafilatura==2.2.0` extracts the main content of web pages. Docling's HTML backend was rejected for web pages because it keeps `<footer>` as content and treats `<nav>`/`<aside>` only as paragraph breaks, so site chrome would reach chunks and the credibility imprint scan. The trafilatura pin is exact on purpose: `web.py` also imports trafilatura's Markdown writer from an internal module (`trafilatura.xml.xmltotxt`), so a version bump means re-running `tests/test_web.py` before trusting it.

Required environment variables (put them in `backend/.env` — settings are anchored to that file regardless of CWD; prefix is `AUTHORAI_` except the provider keys, which are read unprefixed):

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | All Claude calls — the client refuses to construct without it |
| `OPENAI_API_KEY` | Embeddings — same refusal |
| `AUTHORAI_API_KEY` | The HTTP API's shared secret; **the server refuses to start without it** (fail-closed) |
| `AUTHORAI_CROSSREF_MAILTO` | Optional but polite — anonymous Crossref is slower |

Everything else (models, paths, weights, limits) has defaults in `backend/authorai/config.py`.

## Everyday commands

```bash
# tests (offline: FakeEmbedder, dim=8; integration tests excluded)
python -m pytest -q -m "not integration"

# integration tests — local only: Docling layout models + example PDFs (big first
# download), and one live fetch of https://example.com/
python -m pytest -q -m integration

# lint + format
ruff check . && ruff format .

# dev server — port 8000; GET /health (unauthenticated), GET /docs
uvicorn authorai.main:app
```

CI (`.github/workflows/ci.yml`, push/PR on `main` and `v2`) runs exactly: `ruff check .` and `pytest -q -m "not integration"` on Python 3.11, then `npm ci` / `npm run build` / `npm run test` on Node 22. Integration tests never run in CI — they need Docling's GB-class layout models and, for the fetch test, the network.

The link fetcher and the web extractor have their own offline suites. `tests/test_fetch.py` injects a resolver into every fetch and mocks the **pinned-IP** URL with respx, so a request sent to the hostname — an unpinned request — reaches nothing mocked and fails. `tests/test_web.py` extracts saved pages in `tests/fixtures/web/`. The one live test, `test_real_fetch_pins_the_ip_and_verifies_the_certificate_against_the_hostname` (marked `integration`), fetches `https://example.com/` and inspects the TLS socket: the request went to an IP literal, and the handshake ran with `server_hostname` set to `example.com` under certificate and hostname verification.

## Frontend

```bash
cd frontend
npm ci
npm run dev        # http://localhost:5173
npm run build      # tsc + vite build (this is also the type-check)
npm run test       # vitest run (Vitest + React Testing Library)
```

Set `VITE_API_BASE_URL` (default `http://localhost:8000`) and `VITE_API_KEY` (must match `AUTHORAI_API_KEY`, or every call 401s). Vite env vars are baked in at dev-server start — restart after editing.

> **Pin note:** Vitest is pinned to **3.x**. Vitest 4 requires Vite ^6+, and this project pins Vite ^5; npm installs the mismatch without failing (peer conflicts are warnings since npm 7), and it only breaks later at `vitest run` with a confusing Vite-internal error. When adding tooling that plugs into Vite, check `peerDependencies` against the installed Vite version before wiring it up.

## The CLI

All pipeline stages are runnable by hand (from `backend/`, inside the conda env):

```bash
python -m authorai.cli ingest "../example_sources/example source one/"*.pdf   # new run, kind=SOURCE
python -m authorai.cli ingest report.pdf --run <run_id> --kind REPORT
python -m authorai.cli extract <run_id>                           # claims from the run's REPORT
python -m authorai.cli eval-extract <run_id> [--golden PATH] [--allow-stale]
python -m authorai.cli verify <run_id> [--sync] [-k N]            # Batch API by default
python -m authorai.cli eval-verdict <run_id> [--golden PATH] [--allow-stale]
python -m authorai.cli score <run_id> [--allow-stale]
python -m authorai.cli search <run_id> some query terms [-k N]
```

Notes:

- `ingest` describes figures with an LLM by default (`--no-describe-figures` to skip); one bad PDF reports and continues, and the run id is printed first so it is never lost.
- `ingest` takes PDF paths only. Web-page sources come in through `POST /api/runs` (the upload dialog's link field), and the ingest step fetches them.
- `verify --sync` makes per-claim sync calls (full price, immediate) — the debug path; batch is half price at minutes-scale latency.
- **`--allow-stale`**: claims and verdicts are stamped with a hash of the prompt that produced them. `eval-extract`, `eval-verdict`, and `score` refuse rows whose stamp differs from the current prompt — a score over stale rows says nothing about the code you are tuning (this exact mistake produced a wrong conclusion once). `--allow-stale` downgrades the refusal to a loud warning.

## Eval discipline

Two label sets under `backend/evals/`:

- **Dev golden** (`evals/golden.jsonl`, 37 audited records) — tune against it freely. `eval-extract`/`eval-verdict` print a delta against `baseline.json` / `verdict_baseline.json` automatically when scoring this set.
- **Holdout** (`evals/holdout/holdout.jsonl`, 27 audited records, separate fake report) — **scored only at phase boundaries, never tuned against.** No baseline delta is printed for it by design: comparing a holdout score to the dev baseline would mislead.

Run the eval after *any* change to prompts, retrieval, chunking, or verdict logic — never tune blind. Both baselines record a measured noise floor; treat deltas inside it as noise, and re-run before attributing a change to your edit. Current recorded numbers: [metrics.md](metrics.md).

### Eval in CI (`.github/workflows/eval.yml`)

Manual `workflow_dispatch` only — each run costs real API money (~$0.5) and fork PRs have no secrets, so a human presses the button and reads the deltas in the job summary; there is no hard fail-gate. Requirements:

- Repo secrets `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` (optional `AUTHORAI_CROSSREF_MAILTO`).
- **Dispatchable from the default branch only** — GitHub lists dispatchable workflows from the default branch, so the file cannot be dispatched from `v2` until it lands on `main`.

The workflow ingests the pinned source pair + dev report, then runs extract → eval-extract → verify → eval-verdict → score. The `score` step is `continue-on-error` (live Crossref + the validity rubric make it the flakiest) and its output is compared to `evals/score_reference.json` by hand.

## Database and migrations

One SQLite file (`backend/data/authorai.db` by default) holds relational tables + FTS5 + sqlite-vec. Two rules to know:

**A worker on an old checkout dies loudly against a newer database.** `db.py` refuses to open any database whose `PRAGMA user_version` is higher than the build's `SCHEMA_VERSION` (currently 12). This is deliberate: an old build writing through a newer schema corrupts silently (e.g. the pre-partition `add_chunks` would store `doc_kind=NULL` rows that every SOURCE-filtered search then misses). When you see this error, **upgrade the checkout — do not work around the guard.**

**Migration discipline.** Adding a migration means, in the same commit:

1. Bump `SCHEMA_VERSION` in `db.py` and add an `if version < N:` block whose DDL and `PRAGMA user_version = N` bump run inside one `BEGIN…COMMIT` script (interruptions must never leave a half-migrated database).
2. Update every migration-rewind test in `tests/test_db.py`. Those tests build a database at the **latest** schema, then fake an old one by setting `user_version` back — so anything your new migration created is still physically present, and the re-run migration collides ("table already exists"). Grep `tests/test_db.py` for `user_version =` and add your new tables/columns to the DROP list of **every rewind below N** (today the rewinds drop `run_scores`, `source_credibility`, `jobs`, `claims.stance`, `claims.extraction_prompt_hash`, `runs.title`, `verdicts.prompt_hash`; via the shared `V11_REWIND` prefix, `uploads.content_hash`, `documents.embedding_model`, and the `idx_uploads_content_hash`/`idx_chunks_doc`/`idx_figures_doc` indexes; and via `V12_REWIND`, which goes before `V11_REWIND`, `uploads.source_type`, `uploads.url`, `chunks.start_seconds`, and `chunks.end_seconds`). A migration with its own columns follows the same shape: a new prefix, placed before the older ones. Mind SQLite's rule that an indexed column can't be dropped: each `DROP INDEX` must precede its column's `DROP COLUMN`. This has broken the suite twice when done reactively.

Also: the server is **single-process by design** — startup recovery re-queues every RUNNING job unconditionally, which is only correct when no other process can be mid-job. Never run `uvicorn --workers N>1` against one database.

## Trying it end to end

Sample PDFs live in `example_sources/`, one folder per test set. From `example source one/`, upload `World_Hunger_Fake.pdf` as the report and the real PDFs (`2025_world_hunger.pdf`, `disruptions_in_the_food_supply_chain.pdf`, …) as sources — via the UI at `:5173`, or `POST /api/runs` directly. `example source two/` holds a second set (six real water/drought sources for `Water_Stress_Fake_Report.pdf`, which sits at the `example_sources/` root next to its answer key). The pipeline runs as one background job; the dashboard polls the progress feed and flips to the full report on DONE. A real run makes paid Anthropic + OpenAI calls proportional to document size.

A source can also be a link to a public web page: paste it into the dialog's **Add a link** field, or send it as a `source_urls` part. The ingest step fetches the links before it processes any document and stops at the first one that can't be read, so a bad link fails the run within seconds; the error names the link, or, for a page that redirected and then could not be read, the address it ended up at.

## Related docs

- [architecture.md](architecture.md) — system map, storage schema, pipeline
- [metrics.md](metrics.md) — what the scores mean + recorded eval baselines
