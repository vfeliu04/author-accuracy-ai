# API Reference

The backend is a FastAPI app built by `create_app` in `backend/authorai/main.py`; all `/api` routes live in `backend/authorai/api.py`. A run is created by uploading a report PDF plus its source PDFs in one request; the full pipeline (ingest → extract → verify → score) then executes as a background job that the client polls. Limits and defaults referenced below come from `backend/authorai/config.py` — see [configuration.md](configuration.md).

## Base URL

```bash
uvicorn authorai.main:app     # from backend/, default port 8000
```

Default base URL: `http://localhost:8000`. CORS is restricted to `AUTHORAI_CORS_ORIGINS` (default `http://localhost:5173`, comma-separated), with `allow_credentials=False` — auth is a header, not a cookie.

## Authentication

Every request whose path is `/api` or starts with `/api/` must carry the API key in the `X-API-Key` header. The check is done by `ApiGuardMiddleware`, a pure-ASGI middleware that runs **before the request body is parsed** — an unauthenticated request is rejected with 401 without the server ever reading its body (a route-dependency check would run only after FastAPI had already parsed the whole multipart upload). Keys are compared in constant time on raw bytes.

The system is fail-closed: if `AUTHORAI_API_KEY` is unset, the app **refuses to start** (`RuntimeError` in the lifespan), and the middleware independently treats "no configured key" as 401. There is no unauthenticated mode.

```bash
curl -H "X-API-Key: $AUTHORAI_API_KEY" http://localhost:8000/api/runs
```

`/health`, `/docs`, `/redoc`, and `/openapi.json` are outside the `/api` prefix and need no key. The middleware is innermost and CORS outermost, so 401/413 rejections still carry CORS headers and are readable by a browser client.

## Limits

| Limit | Default | Enforced |
|---|---|---|
| Whole request (`Content-Length`) | 220,000,000 bytes (`max_request_bytes`) | In the middleware, before the body is read → 413 |
| Per uploaded file | 50,000,000 bytes (`max_upload_bytes`) | In `POST /api/runs`, from the spooled part's size → 413 |
| Source files per run | 20 (`max_source_files`) | In `POST /api/runs` → 400 |

## Error shapes

All errors are JSON with a `detail` key.

| Status | Producer | Body |
|---|---|---|
| 401 | Middleware (missing/wrong/unconfigured key) | `{"detail": "Invalid or missing API key"}` |
| 413 | Middleware (`Content-Length` over cap) | `{"detail": "Request body exceeds the size limit"}` |
| 413 | Upload validation (one file over cap) | `{"detail": "'<name>' exceeds the <n> byte per-file limit"}` |
| 400 | Upload validation | `detail` is `'<name>' is not a .pdf file` / `'<name>' is not PDF content` / `too many source files (N > 20)` |
| 404 | Route handlers | `{"detail": "Unknown run '<id>'"}` etc. (exact strings per endpoint below) |
| 409 | Chat on an unfinished run | `{"detail": "The run is not scored yet — chat is available once it is DONE"}` |
| 422 | FastAPI/Pydantic validation | `{"detail": [{"type": ..., "loc": [...], "msg": ..., ...}]}` — the standard FastAPI validation-error list |

## Endpoint index

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | no | Liveness: `{"status": "ok", "version": "<pkg version>"}` |
| GET | `/docs`, `/redoc`, `/openapi.json` | no | Interactive docs; served only when `docs_enabled` (see below) |
| POST | `/api/runs` | yes | Upload report + sources, queue the pipeline (202) |
| GET | `/api/runs` | yes | List all runs, newest first |
| GET | `/api/runs/{run_id}` | yes | Run detail + latest job (progress feed) |
| GET | `/api/jobs/{job_id}` | yes | One job by id |
| GET | `/api/runs/{run_id}/report` | yes | Full analysis payload: scores, stats, claims, sources |
| GET | `/api/runs/{run_id}/documents/{doc_id}/file` | yes | Stream a run's stored PDF inline |
| POST | `/api/runs/{run_id}/chat` | yes | Grounded Q&A over a DONE run (see [chat.md](chat.md)) |

## Runs and jobs

### `POST /api/runs` → 202

Multipart form with two fields:

| Field | Type | Rules |
|---|---|---|
| `report` | one file | Required. `.pdf` extension, `%PDF-` magic bytes, ≤ `max_upload_bytes` |
| `sources` | list of files | Required. Same per-file rules; at most `max_source_files` |

Every file is validated (extension, size, magic bytes — without reading it into memory) before any file is written. Files are stored under `uploads_dir` with server-generated names; the client filename is kept only as display metadata and never touches a path. The run, its upload rows, and a `full_pipeline` job commit in **one transaction** — a failure anywhere deletes the written files and leaves no rows.

Response: `{"run_id": "<hex>", "job_id": "<hex>"}`. The job starts `QUEUED`; a single worker thread picks it up (poll interval `job_poll_seconds`). Poll `GET /api/runs/{run_id}` for progress.

### `GET /api/runs`

`{"runs": [<run>, ...]}` ordered by `created_at` descending. A run object:

| Field | Meaning |
|---|---|
| `id` | Run id |
| `created_at` | ISO-8601 UTC timestamp |
| `status` | `CREATED` \| `RUNNING` \| `DONE` \| `FAILED` |
| `error` | Failure message, else `null` |

### `GET /api/runs/{run_id}`

`{"run": <run>, "job": <job> | null}` — the run plus its most recent job. 404 `Unknown run '<id>'` otherwise.

A job object:

| Field | Meaning |
|---|---|
| `id`, `run_id`, `kind` | `kind` is `full_pipeline` |
| `status` | `QUEUED` \| `RUNNING` \| `DONE` \| `FAILED` |
| `payload` | The work order: `{"report_upload_id", "source_upload_ids"}` |
| `progress` | Array of `{step, label, status, ts}`; `step` ∈ `ingest`, `extract`, `verify`, `score` (plus a `recovered` entry if the job was re-queued after a restart); `status` ∈ `running`, `done`, `failed` |
| `error` | Failure message, else `null` |
| `created_at`, `updated_at` | ISO-8601 UTC timestamps |

Jobs are resumable: on startup, any job left `RUNNING` by a crash is re-queued and resumes from its first incomplete step.

### `GET /api/jobs/{job_id}`

The job object above, or 404 `Unknown job '<id>'`.

## Report payload

### `GET /api/runs/{run_id}/report`

404 `Unknown run '<id>'` for unknown runs. Works at any run status — before scoring, `scores` is `null` and `claims`/`sources` may be empty. All rows are read in one transaction, so the payload is a consistent snapshot even while the worker is committing.

```json
{
  "run_id": "…",
  "status": "DONE",
  "report_doc_id": "…",          // the report's document id (for the file endpoint), or null
  "scores": { "accuracy": 0.84, "coverage": 0.97, "credibility": 0.509, "validity": 0.42 },
  "stats": { "claims_total": 37, "claims_supported": 30, "claims_contradicted": 3, "claims_unverifiable": 4 },
  "claims": [ … ],
  "sources": [ … ]
}
```

`scores` values are all 0–1 fractions (credibility and validity are stored 0–100 and divided by 100 here); `accuracy` and `coverage` can be `null` when no claim was decided. `accuracy` is report-position agreement over decided claims; UNVERIFIABLE claims count only against `coverage`, never against `accuracy`.

Each entry in `claims` (ordered by report page, then id):

| Field | Meaning |
|---|---|
| `claim_id`, `text`, `page` | The extracted claim and where it appears in the report |
| `value`, `unit`, `year` | Parsed quantitative fields, `null` when absent |
| `verdict` | `SUPPORTED` \| `CONTRADICTED` \| `UNVERIFIABLE` — relative to the ingested sources only |
| `stance` | `asserted` \| `disavowed` — the report's own position; `disavowed` means the report itself marks the claim false (so a CONTRADICTED verdict there is the report being *right*) |
| `downgraded` | `true` when code-side checks overrode the model's verdict (stored `raw_verdict` differs from `verdict`, e.g. a failed quote check forced UNVERIFIABLE) |
| `quote` | The evidence quote the judge cited, or `null` |
| `quote_verified` | `1` quote found verbatim in the cited chunk, `0` check failed, `null` no quote applicable |
| `rationale` | The judge's reasoning |
| `year_flag` | `1` when the claim's year is absent from the cited chunk, else `null` |
| `evidence_source` | `{"doc_id", "title", "page"}` of the source document behind the quoted chunk, or `null` when no chunk was cited |

Each entry in `sources` (ordered by credibility, highest first):

| Field | Meaning |
|---|---|
| `doc_id`, `title` | The source document |
| `total` | Credibility score 0–100 (no floors — unknown metadata earns nothing) |
| `tier` | Crossref verification tier: `VERIFIED_DOI` \| `VERIFIED_TITLE` \| `METADATA_ONLY` \| `NONE` |
| `components` | `{"metadata_completeness", "authority", "recency", "verification"}` point breakdown |

## Document files

### `GET /api/runs/{run_id}/documents/{doc_id}/file`

Streams the stored PDF (report or source) with `Content-Type: application/pdf` and `Content-Disposition: inline; filename="<original name>"` — meant for an embedded viewer. `doc_id` is a document id from the report payload (`report_doc_id` or a source's `doc_id`).

Access is scoped by `(run_id, doc_id)`: a document id from another run returns 404 `No such document in this run`. As defense in depth, the stored path (already server-generated) is resolve-checked to lie inside `uploads_dir`; a path escaping it, or a missing file, returns 404 `Document file is unavailable`.

## Chat

### `POST /api/runs/{run_id}/chat`

Grounded Q&A over a completed run's analysis. 404 for an unknown run; **409** unless the run's status is `DONE`. JSON body (Pydantic-validated, 422 on violation):

| Field | Rules |
|---|---|
| `question` | Required, 1–4,000 chars |
| `history` | Optional, ≤ 50 turns of `{"role": "user"|"assistant", "content": <1–8,000 chars>}` — client-held; the server stores nothing |
| `mode` | `evidence` (default) \| `guidance` \| `creative` |

Response: `{"answer": "<text>", "mode": "<mode>"}`. Details — context construction, prompt caching, history trimming — in [chat.md](chat.md).

## Health and docs

`GET /health` returns `{"status": "ok", "version": "<package version>"}` with no auth — safe for probes.

Swagger (`/docs`), ReDoc (`/redoc`), and the OpenAPI schema (`/openapi.json`) are served only while `docs_enabled` is true (the default). They expose the full route surface; disable them (`AUTHORAI_DOCS_ENABLED=false`) for an exposed deployment.
