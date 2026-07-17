# API Reference

The backend is a single Flask app defined in `backend/app.py` (`create_app`). It exposes JSON endpoints for uploading PDFs, running the verification pipeline as a background job, reading report summaries/claims, inspecting sources, and chatting about the latest report. The React frontend consumes these endpoints through the typed wrapper in `frontend/src/api/client.ts`. For how the pipeline itself works, see [architecture](architecture.md); for score definitions, see [metrics](metrics.md).

## Base URL

Running `python app.py` starts Flask on port **5001** (`app.run(debug=True, port=5001)`), so the default base URL is:

```
http://localhost:5001
```

The frontend defaults to the same value via `VITE_API_BASE_URL` (see [configuration](configuration.md)). CORS is enabled for all origins via `flask_cors.CORS(app)`.

## Authentication

Every endpoint except `GET /health` is wrapped in a `require_api_key` decorator that compares the `X-API-Key` request header against the `API_KEY` environment variable (read into `settings.api_key`). On mismatch it returns `401 {"error": "Unauthorized"}`.

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:5001/api/uploads
```

> **Warning:** if `API_KEY` is unset, the check is a **no-op** — the API is completely open. Set `API_KEY` on the backend (and the matching `VITE_API_KEY` on the frontend, which sends the header only when that variable is defined) for any non-local deployment.

## Error Handling

Errors are returned as JSON with the shape `{"error": "<message>"}`. `create_app` registers these handlers:

| Exception | Status | Body |
|---|---|---|
| `FileNotFoundError` | 404 | `{"error": "<message>"}` |
| `ValueError` | 400 | `{"error": "<message>"}` |
| `werkzeug.exceptions.HTTPException` | its own code | `{"error": "<description>"}` |
| any other `Exception` | 500 | `{"error": "Internal Server Error"}` |

Several handlers also return explicit 400/404 responses directly (noted per endpoint below). The frontend's `apiFetch` throws on any non-2xx response using the raw response text as the error message.

## Endpoint Index

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | no | Liveness check |
| POST | `/api/uploads/source` | yes | Upload source PDFs |
| POST | `/api/uploads/report` | yes | Upload the report PDF |
| GET | `/api/uploads` | yes | List uploads |
| DELETE | `/api/uploads/<upload_id>` | yes | Delete an upload |
| GET | `/api/uploads/<upload_id>/file` | yes | Download/serve an uploaded file |
| POST | `/api/run_pipeline` | yes | Start a pipeline job (async) |
| GET | `/api/jobs/<job_id>` | yes | Poll job status/progress |
| GET | `/api/dashboard` | yes | Summary of the latest completed job |
| GET | `/api/reports/latest` | yes | Same as `/api/dashboard` |
| GET | `/api/reports/<job_id>/summary` | yes | Summary of a specific job |
| GET | `/api/reports/<job_id>/claims` | yes | Paginated claims with evidence |
| GET | `/api/sources/<source_id>` | yes | Source detail (credibility, claims) |
| GET | `/api/claims` | yes | All claims in the database |
| POST | `/api/ingest/source` | yes | Index a source by server path |
| POST | `/api/verify/report` | yes | Synchronous verify by server path |
| POST | `/api/credibility` | yes | Score one source by server path |
| POST | `/api/chat` | yes | Ask a question about a report |
| GET | `/api/chat/history` | yes | Chat history for a job's report |

## Health

### `GET /health`

Returns `{"status": "ok"}`. No authentication — safe for load-balancer probes.

## Uploads

Upload records (SQLite `uploads` table, see `Repository` in `backend/author_ai/storage/database.py`) are serialized as:

| Field | Meaning |
|---|---|
| `upload_id` | Generated ID, used everywhere else as `source_id` / `report_id` |
| `file_name` | Sanitized original filename |
| `file_type` | `SOURCE` or `REPORT` |
| `path` | Absolute path on the server's filesystem |
| `created_at` | ISO timestamp |
| `file_url` | `/api/uploads/<upload_id>/file` |

### `POST /api/uploads/source`

Multipart form upload of one or more source PDFs under the field name `files` (falls back to `file`). Each file is saved to disk and recorded with type `SOURCE`. Returns `{"uploads": [<upload>, ...]}`. If no non-empty files are provided, a `ValueError` yields 400 `No files were provided.`.

### `POST /api/uploads/report`

Multipart form upload of exactly one report PDF under the field name `file`. Returns a single upload object (not wrapped in a list). Missing file → 400 `Report file is required`.

### `GET /api/uploads?type=source|report`

Lists uploads, newest first, as `{"uploads": [...]}`. The `type` query param is optional and case-insensitive (it is upper-cased before matching `file_type`).

### `DELETE /api/uploads/<upload_id>`

Deletes the file from disk (also removing its now-empty parent directory) and the DB row. Returns `{"status": "deleted"}`, or 404 `{"status": "not_found"}` for an unknown ID.

### `GET /api/uploads/<upload_id>/file`

Serves the raw file inline (`send_file` with `as_attachment=False`) — used by the frontend PDF viewer. 404 `{"error": "File not found"}` for unknown IDs.

## Pipeline & Jobs

### `POST /api/run_pipeline`

JSON body:

| Field | Required | Meaning |
|---|---|---|
| `source_ids` | yes (non-empty) | Upload IDs with `file_type=SOURCE` |
| `report_id` | yes | Upload ID with `file_type=REPORT` |

Creates a job row with status `QUEUED`, then validates that every ID exists and has the right type (400 `Invalid source upload: <id>` / `Invalid report upload` otherwise), starts the pipeline in a **daemon background thread**, and immediately returns `{"job_id": "<uuid>", "status": "QUEUED"}`. Poll `GET /api/jobs/<job_id>` for progress.

The job thread runs, in order: indexing + credibility scoring of each source, claim extraction/verification, validity scoring, credibility aggregation, and OpenAlex-based source recommendations, recording each phase in `progress_json`. On success the job becomes `DONE` with `result_json` containing `claims`, `report_id`, `validity`, `credibility`, and `recommended_sources`; on any exception it becomes `FAILED` with `error_message` set and the last running progress step marked `failed`.

> **Destructive side effect:** the job thread calls `reset_environment()` (`backend/author_ai/services/environment.py`) before indexing. This deletes the FAISS index and cache directories and wipes the `documents`, `chunks`, `claims`, `claim_evidence`, `credibility_scores`, `validity_scores`, and **`chat_logs`** tables. The app is deliberately single-report, last-run-wins: each run erases all prior pipeline data and chat history. Only the `uploads`, `jobs`, and `charts` tables survive. See [history](history.md) for why.

> **Quirk:** the job row is created *before* the source/report IDs are validated, so a 400 validation failure leaves an orphaned job stuck in `QUEUED`.

### `GET /api/jobs/<job_id>`

Returns the full job record (404 `{"error": "Job not found"}` otherwise):

| Field | Meaning |
|---|---|
| `job_id`, `status` | Status is `QUEUED`, `RUNNING`, `DONE`, or `FAILED` |
| `report_id`, `source_ids` | Inputs to the run |
| `result_json` | Pipeline output once `DONE` (`{}` before) |
| `error_message` | Set when `FAILED` |
| `progress_json` | List of `{step, label, status, ts}` where status ∈ `running`/`done`/`failed` and step ∈ `indexing`, `verifying`, `validity`, `credibility`, `recommendations` |
| `created_at`, `updated_at` | ISO timestamps |

## Reports & Claims

### `GET /api/dashboard` and `GET /api/reports/latest`

These two endpoints are **equivalent**: both look up the most recent job with status `DONE` and return `build_report_summary` for it, or 404 `{"error": "No completed reports"}` if none exists. (A `MOCK_DASHBOARD` constant still exists in `app.py` but is no longer served.)

### `GET /api/reports/<job_id>/summary`

Same summary for a specific job. 404 `{"error": "Report not ready"}` unless the job exists and is `DONE`.

Summary response fields (built by `build_report_summary` in `app.py`):

| Field | Meaning |
|---|---|
| `job_id` | The job the summary was built from |
| `report` | `{id, name, pdf_url, summary}` — `summary` is a "X of Y claims supported" line |
| `scores` | `{overall, accuracy, credibility, validity}`, all fractions 0–1. `accuracy` = supported claims / total; `credibility` and `validity` are the pipelines' 0–100 scores divided by 100; `overall` is the plain mean of the three. See [metrics](metrics.md). |
| `recommended_sources` | Normalized OpenAlex recommendations (`id`, `title`, `summary`, `abstract`, `credibility_score`, `validity_score`, `date_published`, `authors`, `doi`, `url`, `openalex_url`, `host_venue`) |
| `chat_messages` | Seed messages: verdict + text of up to 3 claims |
| `sources` | Per-source `{id, name, file_url, summary, scores: {credibility}, usage_count}` where `usage_count` counts claims the source supports |
| `claims` | The full claims list from `result_json` (unpaginated) |
| `stats` | `{claims_total, claims_supported, claims_contradicted, claims_not_found}` |
| `top_sources` | Top 5 sources by usage count, then credibility |

> **Side effect on read:** if the job's `result_json` has no `recommended_sources` (e.g. jobs from before recommendations were persisted), the summary builder calls `RecommendationService.recommend` — a live OpenAlex network fetch — and writes the results back into the job row. The first read of such a report can therefore be slow and mutates the database.

### `GET /api/reports/<job_id>/claims?limit=5&page=0`

Paginated claims for a `DONE` job (404 `Report not ready` otherwise; non-integer pagination params → 400 `Invalid pagination parameters`). Returns `{"claims": [...], "total": <int>, "has_more": <bool>}`.

Each claim carries `claim_id`, `report_id`, `text`, `verdict` (`SUPPORTED`, `CONTRADICTED`, or `NOT_FOUND`), `confidence`, `confidence_band`, `explanation`, `processing_mode`, and `metadata`, and is enriched here with `parent_page` (page in the report PDF) and `evidence`: up to 2 entries of `{snippet, source_id, source_name, page, score}`, excluding evidence rows labeled `ALTERNATIVE`.

## Sources

### `GET /api/sources/<source_id>?claim_limit=5&claim_page=0`

Detail view for one uploaded source (404 `Source not found` for unknown IDs; bad pagination ints → 400). Response:

| Field | Meaning |
|---|---|
| `upload` | The upload record (see Uploads) |
| `credibility` | Row from `credibility_scores` (`score` on a 0–100 scale, `metadata_confidence`, `components`, `updated_at`) or `null` |
| `claims`, `claim_total`, `claim_has_more` | Paginated claims that cite this source |
| `usage_count` | Total distinct claims linked to this source |
| `tables` | At most one extracted table preview |
| `summary` | Document summary, falling back to first line of body text, then filename |
| `validity` | `{score, supported, total}` computed **only over the current page of claims**, or `null` if the page is empty |

### `GET /api/claims`

Returns every claim in the database as `{"claims": [...]}` (no pagination, no evidence enrichment). Because pipeline runs wipe prior data, this is effectively the latest run's claims.

## Ingestion & Credibility (path-based)

These endpoints take a JSON body with a `path` field pointing at a PDF **on the server's filesystem** (`_resolve_pdf_path` raises `FileNotFoundError` → 404 if missing). They predate the upload/job flow; the frontend does not use them.

| Endpoint | Body | Response |
|---|---|---|
| `POST /api/ingest/source` | `{"path": "..."}` | `{"chunks": <int>, "tables": <int>}` — indexes the PDF into the vector store |
| `POST /api/verify/report` | `{"path": "..."}` | `{"claims", "report_id", "validity", "credibility"}` — runs verification **synchronously** (can take minutes; no job record, no environment reset) |
| `POST /api/credibility` | `{"path": "..."}` | The credibility score object for that single source (`source_id`, `score`, `components`, ...) |

> **Currently broken:** these handlers pass the resolved `Path` straight into `AccuracyPipeline.index_source`/`verify_report` and `CredibilityPipeline.score_source`, but those methods now expect an upload dict (`upload["path"]`, `upload["upload_id"]`). A `Path` is not subscriptable, so any call with an existing file raises `TypeError` and returns 500 `{"error": "Internal Server Error"}`. Use the upload + `/api/run_pipeline` flow instead.

## Chat

See [chat](chat.md) for modes and behavior; the HTTP contract is:

### `POST /api/chat`

JSON body:

| Field | Required | Meaning |
|---|---|---|
| `question` | yes | The user's question (defaults to empty string) |
| `job_id` | no | Target job; must be `DONE`, else 400 `Report not ready`. If omitted, falls back to the latest `DONE` job |
| `session_id` | no | Continues an existing chat session |
| `mode` | no | `evidence`, `guidance`, or `creative` |
| `mode_locked` | no | Boolean; pin the mode instead of letting the service switch it |

400 `No completed report` if no report can be resolved. Response (as consumed by the frontend): `{"answer", "claims_used": [{claim_id, text, verdict}], "sources_used": [{source_id, snippet?}], "mode", "suggested_mode"?}`.

### `GET /api/chat/history?job_id=<id>`

Returns `{"history": [{session_id, role, message, timestamp, context_ids}]}` in chronological order for the job's report. 400 `job_id required` without the param; 404 `Job not found` for unknown jobs; `{"history": []}` if the job has no `report_id`.

> **Gotcha:** chat history lives in the `chat_logs` table, which `reset_environment()` clears — every new pipeline run erases all previous chat history.
