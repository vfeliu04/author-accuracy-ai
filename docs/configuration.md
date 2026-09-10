# Configuration Reference

All settings live in the `Settings` class in `backend/authorai/config.py` (pydantic-settings `BaseSettings`). Every setting is an environment variable with the `AUTHORAI_` prefix — except the two provider keys, which are read unprefixed via aliases (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). Names are case-insensitive; unknown variables are ignored (`extra="ignore"`). Real environment variables win over the `.env` file.

The `.env` file is loaded from **`backend/.env`, anchored to the package location regardless of the process working directory** — a CWD-relative `.env` would silently load nothing when the server is started from elsewhere. A template exists at `backend/.env.example`:

```bash
cp backend/.env.example backend/.env
# then fill in the keys
```

`Settings` is constructed where needed and passed explicitly — there is no cached global, but processes read the environment at construction, so changing a value still means restarting the server.

> **Note:** the *path* settings have relative defaults (`data/…`) and are **not** anchored — they resolve against the process working directory. Start the server from `backend/`, or set absolute paths.

## Keys and auth

| Env var | Default | What it does |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Claude access for extraction, verdicts, captions, metadata, validity, and chat. Unprefixed on purpose (validation alias) so an existing `.env` keeps working. The LLM client **refuses to construct** without it — no silent degradation |
| `OPENAI_API_KEY` | unset | OpenAI embeddings (also unprefixed via alias) |
| `AUTHORAI_API_KEY` | unset | Shared secret clients send as `X-API-Key`. **Fail-closed:** if unset, the server refuses to start (`RuntimeError` at startup) rather than serving an open API, and the auth middleware independently rejects everything with 401 when no key is configured |

## HTTP server

| Env var | Default | What it does |
|---|---|---|
| `AUTHORAI_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins (`allow_credentials` is off — auth is a header, not a cookie) |
| `AUTHORAI_DOCS_ENABLED` | `true` | Serves `/docs`, `/redoc`, and `/openapi.json`. They expose the full route surface — keep for local dev, **disable for an exposed deployment** |
| `AUTHORAI_MAX_REQUEST_BYTES` | `220000000` | Whole-request ceiling, checked against `Content-Length` **before** the body is read, so an unauthenticated attacker cannot push gigabytes → 413 |
| `AUTHORAI_MAX_UPLOAD_BYTES` | `50000000` | Per-file upload cap, checked from the spooled part's size without materializing the bytes → 413. Also caps a PDF fetched from a link |
| `AUTHORAI_MAX_SOURCE_FILES` | `20` | Max sources per run, uploaded files and links together → 400 |
| `AUTHORAI_UPLOADS_DIR` | `data/uploads` | Where uploaded PDFs, web-page snapshots (JSON), and PDFs fetched from links are stored (server-generated names); the file endpoint resolve-checks every served path against this directory |

## Source links

How the ingest step fetches links added as sources (`backend/authorai/fetch.py`). A web page's body is capped by `AUTHORAI_FETCH_MAX_BYTES`; a PDF served by a link is capped by `AUTHORAI_MAX_UPLOAD_BYTES`.

| Env var | Default | What it does |
|---|---|---|
| `AUTHORAI_FETCH_TIMEOUT_SECONDS` | `30.0` | Wall-clock budget for one link's whole fetch, every redirect hop included. Checked between hops and after each body chunk, applied as each request's socket timeouts, and enforced by a watchdog that shuts the connection's socket; the OS DNS lookup is the one wait it cannot interrupt |
| `AUTHORAI_FETCH_MAX_BYTES` | `10000000` | Cap on a web page's body, counted in decoded bytes (after `Content-Encoding`), so a compressed response cannot slip past it |
| `AUTHORAI_FETCH_MAX_REDIRECTS` | `5` | Redirect hops followed; each target is re-checked, re-resolved, and re-gated against private addresses |
| `AUTHORAI_FETCH_USER_AGENT` | `AuthorAccuracyAI/2.0 (+https://github.com/vfeliu04/author-accuracy-ai)` | `User-Agent` sent with every fetch |

## Storage and embeddings

| Env var | Default | What it does |
|---|---|---|
| `AUTHORAI_DB_PATH` | `data/authorai.db` | SQLite database file (run-scoped schema; migrated in place on open) |
| `AUTHORAI_FIGURES_DIR` | `data/figures` | Extracted figure PNGs, under `<figures_dir>/<run_id>/<doc_id>/` |
| `AUTHORAI_EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model for chunk vectors |
| `AUTHORAI_EMBEDDING_DIM` | `3072` | Embedding dimension. Baked into the database at creation; opening an existing DB with a different value **fails loudly** rather than silently corrupting similarity search |

## Pipeline models

The split is deliberate: the accuracy-critical judgments (extraction, verdicts, validity) run on the frontier model; cheap bounded tasks (captions, bibliographic metadata) run on Haiku.

| Env var | Default | What it does |
|---|---|---|
| `AUTHORAI_EXTRACTION_MODEL` | `claude-opus-5` | Claim extraction — the language judgment the whole score rests on |
| `AUTHORAI_VERDICT_MODEL` | `claude-opus-5` | Per-claim verdicts with schema-quoted evidence |
| `AUTHORAI_VALIDITY_MODEL` | `claude-opus-5` | The validity rubric over the whole report |
| `AUTHORAI_CAPTION_MODEL` | `claude-haiku-4-5` | Figure descriptions (vision) baked into chunk text |
| `AUTHORAI_METADATA_MODEL` | `claude-haiku-4-5` | Bibliographic metadata extraction for source credibility. A web page uses it only when its markup declares nothing beyond a title |

## Scoring

| Env var | Default | What it does |
|---|---|---|
| `AUTHORAI_VALIDITY_WEIGHTS` | `coverage:0.25,consistency:0.25,methodology:0.2,context:0.2,recency:0.1` | `name:weight` pairs for the validity components. Parsed loudly: unknown names, duplicates, non-finite/negative weights, or a sum ≠ 1 raise instead of falling back |
| `AUTHORAI_AUTHORITY_TIER1` | `FAO,Food and Agriculture Organization,UN,United Nations,World Bank,IMF,WHO,World Health Organization,UNICEF,OECD,Welthungerhilfe,WMO,World Meteorological Organization,UNCCD` | Publishers granted top authority points. Matched as consecutive word-boundary phrases (`UN` matches `U.N.` but never `University`); keep needles as specific as the real names allow |
| `AUTHORAI_AUTHORITY_TIER2` | `Reuters,Associated Press,BBC,Nature,Science,Lancet,Elsevier,National Drought Mitigation Center,NDMC,International Water Management Institute,IWMI,CGIAR,World Climate Research Programme,WCRP` | Second-tier publishers, same matching rules |
| `AUTHORAI_CROSSREF_MAILTO` | unset | Contact email for polite Crossref access (source verification tiers) |

## Jobs

| Env var | Default | What it does |
|---|---|---|
| `AUTHORAI_JOB_POLL_SECONDS` | `2.0` | Worker thread's poll interval for `QUEUED` jobs |

## Chat

See [chat.md](chat.md) for how these interact.

| Env var | Default | What it does |
|---|---|---|
| `AUTHORAI_CHAT_MODEL` | `claude-sonnet-5` | Chat model — Sonnet-class with prompt caching over the static per-run context, cheaper than the Opus judgments |
| `AUTHORAI_CHAT_MAX_TOKENS` | `2048` | Chat response token budget |
| `AUTHORAI_CHAT_HISTORY_TURNS` | `12` | Most recent client-sent history messages kept per request |
