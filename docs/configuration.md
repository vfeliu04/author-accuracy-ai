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
| `AUTHORAI_MAX_UPLOAD_BYTES` | `50000000` | Per-file upload cap, checked from the spooled part's size without materializing the bytes → 413. Also caps a fetched link's response served as `application/pdf` or `application/octet-stream` |
| `AUTHORAI_MAX_SOURCE_FILES` | `20` | Max sources per run, uploaded files and links together → 400 |
| `AUTHORAI_UPLOADS_DIR` | `data/uploads` | Where uploaded PDFs, web-page snapshots (JSON), and PDFs fetched from links are stored (server-generated names); the file endpoint resolve-checks every served path against this directory, and deleting a run removes only upload files that lie inside it |

## Source links

How the ingest step fetches links added as sources (`backend/authorai/fetch.py`) and reads the pages they serve (`backend/authorai/web.py`). The size cap follows the response's declared `Content-Type`: `text/html` and `application/xhtml+xml` are capped by `AUTHORAI_FETCH_MAX_BYTES`, even when the body turns out to be a PDF, and `application/pdf` and `application/octet-stream` by `AUTHORAI_MAX_UPLOAD_BYTES`. DNS lookups count against `AUTHORAI_FETCH_TIMEOUT_SECONDS`, but a slow lookup is not cut short: the budget is checked again once it returns.

| Env var | Default | What it does |
|---|---|---|
| `AUTHORAI_FETCH_TIMEOUT_SECONDS` | `30.0` | Wall-clock budget for one link's whole fetch, every redirect hop included. Checked between hops and after each body chunk, applied as each request's socket timeouts, and enforced by a watchdog that shuts the connection's socket; the OS DNS lookup is the one wait it cannot interrupt. A body the watchdog cut short fails as a timeout, never as a stored page |
| `AUTHORAI_FETCH_MAX_BYTES` | `10000000` | Cap on the body of a response served as `text/html` or `application/xhtml+xml`, counted in decoded bytes (after `Content-Encoding`), so a compressed response cannot slip past it. Only a single coding layer is accepted: more than one layer (`gzip, gzip`, in one header or across repeated `Content-Encoding` headers; `identity` does not count) is refused before the body is read |
| `AUTHORAI_EXTRACT_TIMEOUT_SECONDS` | `60.0` | Wall-clock budget for reading one fetched web page, starting the reader included. The page is read in a separate process that is stopped at the deadline, and the run fails naming the link (`too large or complex`); nothing else could interrupt the single worker thread. The process also carries a CPU-time limit of this budget, rounded up, plus 5 seconds: a backstop that ends a reader whose server is gone and can no longer stop it. The same budget bounds the reference scan's read of the report PDF (`POST /api/references/scan`), in the same kind of process with the same CPU limit and, on Linux, a 1 GiB address-space cap; a read that passes it is a 400 (`could not read the PDF: it was too costly to read`), not a failed run. A link that serves a PDF is not subject to it |
| `AUTHORAI_WEB_MAX_CHARS` | `200000` | How much article text ONE fetched page may contribute, in characters. `AUTHORAI_FETCH_MAX_BYTES` bounds the page's markup, not what the reader gets out of it, and a document's sections are chunked and embedded in one list — at the byte cap that measured about a gigabyte of live Python floats for one page, times `AUTHORAI_MAX_SOURCE_FILES` links. Whole sections past the cap are dropped before the page is stored, with a warning naming the page and the characters dropped, and with `truncated: {kept_chars, dropped_chars}` written into the stored page's provenance — the report's source row then reads *Read in part*, so a partly-read page is visible to the user and not only in the log; a first section that alone exceeds it is cut at its last line break. A long real article is well under the default. PDFs are not subject to it |
| `AUTHORAI_FETCH_MAX_REDIRECTS` | `5` | Redirect hops followed; each target is re-checked, re-resolved, and re-gated against private addresses |
| `AUTHORAI_FETCH_USER_AGENT` | `AuthorAccuracyAI/2.0 (+https://github.com/vfeliu04/author-accuracy-ai)` | `User-Agent` sent with every fetch |

## Storage and embeddings

| Env var | Default | What it does |
|---|---|---|
| `AUTHORAI_DB_PATH` | `data/authorai.db` | SQLite database file (run-scoped schema; migrated in place on open) |
| `AUTHORAI_FIGURES_DIR` | `data/figures` | Extracted figure PNGs, under `<figures_dir>/<run_id>/<doc_id>/`; deleting a run removes its `<figures_dir>/<run_id>/` |
| `AUTHORAI_EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model for chunk vectors |
| `AUTHORAI_EMBEDDING_DIM` | `3072` | Embedding dimension. Baked into the database at creation; opening an existing DB with a different value **fails loudly** rather than silently corrupting similarity search |

## Pipeline models

The split is deliberate: the accuracy-critical judgments (extraction, verdicts, validity) run on the frontier model; cheap bounded tasks (captions, bibliographic metadata, the upload dialog's reference scan) run on Haiku.

| Env var | Default | What it does |
|---|---|---|
| `AUTHORAI_EXTRACTION_MODEL` | `claude-opus-5` | Claim extraction — the language judgment the whole score rests on |
| `AUTHORAI_VERDICT_MODEL` | `claude-opus-5` | Per-claim verdicts with schema-quoted evidence |
| `AUTHORAI_VALIDITY_MODEL` | `claude-opus-5` | The validity rubric over the whole report |
| `AUTHORAI_CAPTION_MODEL` | `claude-haiku-4-5` | Figure descriptions (vision) baked into chunk text |
| `AUTHORAI_METADATA_MODEL` | `claude-haiku-4-5` | Bibliographic metadata extraction for source credibility. A web page uses it only when its markup declares nothing beyond a title |
| `AUTHORAI_REFERENCES_MODEL` | `claude-haiku-4-5` | The upload dialog's reference scan (`POST /api/references/scan`): reads the report's printed reference list into fields — title, authors, year, and the DOI or address the entry prints. The same kind of bounded task as metadata extraction |

## Scoring

| Env var | Default | What it does |
|---|---|---|
| `AUTHORAI_VALIDITY_WEIGHTS` | `coverage:0.25,consistency:0.25,methodology:0.2,context:0.2,recency:0.1` | `name:weight` pairs for the validity components. Parsed loudly: unknown names, duplicates, non-finite/negative weights, or a sum ≠ 1 raise instead of falling back |
| `AUTHORAI_AUTHORITY_TIER1` | `FAO,Fao,Food and Agriculture Organization,UN,United Nations,World Bank,IMF,WHO,World Health Organization,UNICEF,Unicef,OECD,Oecd,Welthungerhilfe,WMO,World Meteorological Organization,UNCCD` | Publishers granted top authority points. Matched as consecutive word-boundary phrases (`UN` matches `U.N.` but never `University`); keep needles as specific as the real names allow. A needle written entirely in capitals is an acronym and matches only in capitals (`WHO` matches `WHO` but not `Who What Wear`; `UN` not the article in `Un Mundo`); other needles ignore case. To accept another spelling of an acronym, add it as its own needle: the defaults carry the mixed-case spellings sites use for `FAO`, `UNICEF`, and `OECD` (`Fao`, `Unicef`, `Oecd`) and, in tier 2, `BBC` (`Bbc`), which ignore case like any needle not written in capitals. `Who` and `Un` are never added, since they would match the ordinary words. Known residual: a publisher styled entirely in capitals (`WHO WHAT WEAR`) still matches |
| `AUTHORAI_AUTHORITY_TIER2` | `Reuters,Associated Press,BBC,Bbc,Nature,Science,Lancet,Elsevier,National Drought Mitigation Center,NDMC,International Water Management Institute,IWMI,CGIAR,World Climate Research Programme,WCRP` | Second-tier publishers, same matching rules |
| `AUTHORAI_CROSSREF_MAILTO` | unset | Contact email for polite Crossref access (source verification tiers). Also the contact email Unpaywall requires: the reference scan looks up free copies of the works a report cites only when it is set — unset, the scan still lists them, with `lookup.status` `unconfigured` and every reference `unknown`. Set, the scan asks about each distinct printed DOI once, four at a time, under a 45-second deadline for the lookup phase as a whole, after which it answers with what was resolved (`lookup.status` `unavailable`) |

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
