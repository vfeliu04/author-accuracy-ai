# API Reference

The backend is a FastAPI app built by `create_app` in `backend/authorai/main.py`; all `/api` routes live in `backend/authorai/api.py`. A run is created by uploading a report PDF plus its sources — PDF files, web links, or both — in one request; the full pipeline (ingest → extract → verify → score) then executes as a background job that the client polls. Before a run exists, the upload dialog can ask for the report's own reference list (`POST /api/references/scan`), which stores nothing. Limits and defaults referenced below come from `backend/authorai/config.py` — see [configuration.md](configuration.md).

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
| Per uploaded file | 50,000,000 bytes (`max_upload_bytes`) | In `POST /api/runs` and `POST /api/references/scan`, from the spooled part's size → 413 |
| Sources per run, files and links together | 20 (`max_source_files`) | In `POST /api/runs` → 400 |
| Link length | 2048 characters, as given and once encoded | In `POST /api/runs` → 400 |
| Run title | 200 characters | In `POST /api/runs` → 400 |
| Pages read by the reference scan | The last 600 (`REFERENCE_MAX_PAGES` in `references.py`) | In `POST /api/references/scan`; earlier pages are not read |
| Closing text the reference scan sends to the model | 30,000 characters (`REFERENCE_MAX_CHARS`), from the last reference-list heading forward (or from an earlier heading within 30,000 characters of it: a multi-page bibliography repeats its heading on each page), else the last 30,000 of the document | In `POST /api/references/scan`; the model never sees more |
| References returned by a scan | 80 (`MAX_REFERENCES`); a longer model answer is cut in code, with a warning in the log | In `POST /api/references/scan` |
| Unpaywall lookups per scan | One per printed DOI, four at a time, 10-second timeout and two retries each; the first registry failure ends the lookup | In `POST /api/references/scan` → `lookup.status` `unavailable`, still 200 |
| Fetched response served as `text/html` or `application/xhtml+xml` (a web page, or a PDF served under an HTML type) | 10,000,000 bytes, counted after decompression (`fetch_max_bytes`) | In the ingest step's fetch → run `FAILED` |
| Fetched response served as `application/pdf` or `application/octet-stream` | 50,000,000 bytes, counted after decompression (`max_upload_bytes`) | In the ingest step's fetch → run `FAILED` |
| Time to fetch one link, redirects and DNS lookups included | 30 seconds (`fetch_timeout_seconds`); a slow DNS lookup is not cut short, and the budget is checked again once it returns | In the ingest step's fetch → run `FAILED` |
| Redirects per link | 5 (`fetch_max_redirects`) | In the ingest step's fetch → run `FAILED` |
| Compression layers on a fetched response | One (`gzip` or `deflate`); more than one compression layer, such as `gzip, gzip`, is refused before the body is read, whether the layers come in one header or in repeated `Content-Encoding` headers (`identity` is not a layer) | In the ingest step's fetch → run `FAILED` |
| Time to read one fetched web page | 60 seconds (`extract_timeout_seconds`), in a separate process that is stopped at the deadline; starting the process counts against it | In the ingest step → run `FAILED` |

## Error shapes

All errors are JSON with a `detail` key.

| Status | Producer | Body |
|---|---|---|
| 401 | Middleware (missing/wrong/unconfigured key) | `{"detail": "Invalid or missing API key"}` |
| 413 | Middleware (`Content-Length` over cap) | `{"detail": "Request body exceeds the size limit"}` |
| 413 | Upload validation (one file over cap) | `{"detail": "'<name>' exceeds the <n> byte per-file limit"}` — `POST /api/runs` and `POST /api/references/scan` alike |
| 400 | Upload validation | `detail` is one of `'<name>' is not a .pdf file` / `'<name>' is not PDF content` / `at least one source (a PDF file or a web link) is required` / `too many sources (N > 20)` / `not a usable link: <reason>` / `'<link>': YouTube links are not supported yet` / `'<link>' was added twice` / `title is limited to 200 characters` |
| 400 | Reference scan | `'<name>' is not a .pdf file` / `'<name>' is not PDF content` as above, or `'<name>': could not read the PDF (<reason>)` when the file passes the magic check but cannot be opened |
| 404 | Route handlers | `{"detail": "Unknown run '<id>'"}` etc. (exact strings per endpoint below) |
| 409 | Chat on an unfinished run | `{"detail": "The run is not scored yet — chat is available once it is DONE"}` |
| 409 | Retry or delete in the wrong state | Exact strings under each endpoint below |
| 422 | FastAPI/Pydantic validation | `{"detail": [{"type": ..., "loc": [...], "msg": ..., ...}]}` — the standard FastAPI validation-error list (a scan without its `report` part lands here) |
| 500 | Reference scan, chat | The model call failed (a provider outage, or no parseable answer); nothing was stored, and the request can simply be repeated |

## Endpoint index

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | no | Liveness: `{"status": "ok", "version": "<pkg version>"}` |
| GET | `/docs`, `/redoc`, `/openapi.json` | no | Interactive docs; served only when `docs_enabled` (see below) |
| POST | `/api/references/scan` | yes | Read a report PDF's own reference list, with whether a free copy is known; stores nothing (200) |
| POST | `/api/runs` | yes | Upload report + sources, queue the pipeline (202) |
| GET | `/api/runs` | yes | List all runs, newest first |
| GET | `/api/runs/{run_id}` | yes | Run detail + latest job (progress feed) + uploads |
| POST | `/api/runs/{run_id}/retry` | yes | Requeue a FAILED run (202) |
| DELETE | `/api/runs/{run_id}` | yes | Delete a run and its stored files (204) |
| GET | `/api/jobs/{job_id}` | yes | One job by id |
| GET | `/api/runs/{run_id}/report` | yes | Full analysis payload: scores, stats, claims, sources |
| GET | `/api/runs/{run_id}/documents/{doc_id}/file` | yes | Stream a run's stored file inline: a PDF, or a web page's JSON snapshot |
| POST | `/api/runs/{run_id}/chat` | yes | Grounded Q&A over a DONE run (see [chat.md](chat.md)) |

## Reference scan

### `POST /api/references/scan` → 200

A pre-upload aid: the upload dialog sends the report PDF the moment it is picked, and shows the works the report cites that are not among the user's sources, offering the ones with a free copy online as links. Multipart form with one part, `report`, under the same rules as the report of `POST /api/runs` (`.pdf` extension, `%PDF-` magic bytes, ≤ `max_upload_bytes`; 413 over the cap, 422 when the part is missing). The middleware's 401 and whole-request 413 apply as to every `/api` route. As in `POST /api/runs`, the per-file cap is checked from the spooled part's size, so a report between 50 MB and the 220 MB whole-request cap is received and spooled to disk before its 413 is answered — accepted, since the scan then reads nothing and stores nothing.

**The scan stores nothing.** It creates no run, upload or job row (the handler has no database dependency, so it cannot), writes no file under `uploads_dir` (the uploaded part is read in place), never fetches a cited work, and caches nothing — a second scan of the same file reads it and asks the model again. Only the PDF's closing pages are read: at most the last 600 pages, with pypdf rather than the ingest pipeline's layout parser, so the answer arrives while the user is still choosing sources. The text the model reads starts at the report's **last** reference-list heading (a line reading `References`, `Bibliography`, `Works Cited`, `Reference list` or `Literature cited`, optionally numbered) — or at the earliest such heading within 30,000 characters before it, since a bibliography that spans several pages prints its heading on each page as a running header, while a contents-page mention farther back stays excluded — and is capped at 30,000 characters; a report without such a heading gets the last 30,000 characters of the document instead. A file with no extractable text at all (a scanned PDF) is answered without a model call, so no bibliography can be invented for it. The model that reads the printed entries into fields is `references_model` (`AUTHORAI_REFERENCES_MODEL`, default `claude-haiku-4-5`; see [configuration.md](configuration.md)).

Response:

```json
{
  "text_source": "heading",                  // "heading" | "tail" | "none"
  "lookup": { "status": "ok", "detail": null },
  "references": [
    {
      "title": "Domestic water consumption and personal habits",
      "authors": ["Smith, J.", "Lee, K."],
      "year": 2024,
      "doi": "10.1016/j.heliyon.2024.e34730",
      "url": null,
      "entry": "Smith, J., & Lee, K. (2024). Domestic water consumption and personal habits. Heliyon, 10(8). https://doi.org/10.1016/j.heliyon.2024.e34730",
      "retrievability": "landing",
      "suggested_url": "https://doi.org/10.1016/j.heliyon.2024.e34730"
    }
  ]
}
```

`text_source` says where the text came from: `heading` (from the reference-list heading, as above), `tail` (no heading; the document's end) or `none` (no text at all; `references` is then empty). At most 80 references are returned; a longer model answer is cut, with a warning in the server log.

Each reference carries what its entry **prints** — `title`, `authors`, `year`, `doi` (no URL prefix) and `url` are `null` or empty when the entry does not print them, and the model is told never to supply a DOI from memory — plus `entry`, the reference as printed, which the dialog shows when there is no title. `retrievability` and `suggested_url` come from Unpaywall, asked by DOI:

| `retrievability` | Meaning | `suggested_url` |
|---|---|---|
| `pdf` | Unpaywall knows a free PDF of the work | The PDF's address |
| `landing` | Unpaywall knows a free copy but only its landing page (for many open-access works its `url_for_pdf` is null) | The landing page |
| `paywalled` | Unpaywall says the work is not open access | Always `null` — a closed work is never offered |
| `unknown` | No DOI printed, the DOI was not found, the record did not say, no lookup was made, or the registry's address failed the link gate | `null`, **except** for an entry with no DOI that prints its own address: that address is offered as addable but unchecked — the dialog labels it "printed in the entry, not checked" |

Every `suggested_url` has passed the syntax gate a pasted link passes (`http`/`https`, a valid host, no credentials, ≤ 2048 characters; the `#fragment` dropped, scheme and host lowercased) and one check more, the refusal the fetcher would make without a network: a literal IP host must be a public address (loopback, private, link-local, carrier-grade NAT, multicast and reserved ranges are refused, IPv4 wrapped in IPv6 included), and `localhost` or any name under it is refused. A name is not resolved — the scan never connects to an address, and never looks one up — so a name that resolves to a private address is offered here and refused at ingest by the fetch gate, which does resolve it. An address that fails is dropped with a warning in the server log, and the record's next address is tried. A DOI is looked up only when it is DOI-shaped (`10.NNNN/…`); anything else is `unknown` without a request.

`lookup.status` is the registry's story for the whole scan:

| `status` | Meaning |
|---|---|
| `ok` | Every printed DOI was asked about (also when there was nothing to ask) |
| `unconfigured` | `AUTHORAI_CROSSREF_MAILTO` is unset, which Unpaywall requires (it answers 422 without a contact email): no lookup was made and every reference is `unknown`; printed addresses are still offered |
| `unavailable` | Unpaywall failed part-way (throttled, down, or unreachable after the retries); `detail` says how. The response is still 200: references resolved before the failure keep their verdict, the rest are `unknown` |

Lookups run four at a time; the first registry failure stops the rest. A failure of the model call itself is not caught and is a 500, as for chat.

## Runs and jobs

### `POST /api/runs` → 202

Multipart form:

| Field | Type | Rules |
|---|---|---|
| `report` | one file | Required. `.pdf` extension, `%PDF-` magic bytes, ≤ `max_upload_bytes` |
| `sources` | list of files | Optional. Same per-file rules |
| `source_urls` | text, one part per link | Optional. Each an `http` or `https` link (rules below) |
| `title` | text | Optional display title for the run, at most 200 characters; whitespace-only or absent falls back to the report filename stem |

At least one source is required — a file or a link — and files and links together count against `max_source_files`.

Each link is checked for syntax only; the request makes no DNS lookup and no connection. Surrounding whitespace is trimmed; the scheme must be `http` or `https`; the host must be a hostname or an IPv6 literal without a zone ID; the port must be valid; the link must not carry a username or password, and must be at most 2048 characters as given and once encoded. A failure is `400 not a usable link: <reason>`, where the reason quotes the link with any credentials removed (for example `Source URL 'ftp://example.org/file' must start with http:// or https://`). The link is then normalized — the `#fragment` dropped, scheme and host lowercased, the host IDNA-encoded, the path percent-encoded — and two links that normalize to the same string are refused (`'<link>' was added twice`), never merged. Links to `youtube.com`, `youtu.be`, or `youtube-nocookie.com`, bare or on `www.`, `m.`, or `music.`, are refused with `'<link>': YouTube links are not supported yet`. Only the link itself is checked against these hosts: a link that redirects to one is fetched like any other page.

Every file (extension, size, magic bytes — without reading it into memory) and every link is validated before any file is written. Files are stored under `uploads_dir` with server-generated names; the client filename is kept only as display metadata and never touches a path. A link is recorded as a `SOURCE` upload with `source_type` `web`, the normalized link as its `url` and `file_name`, and the path its page will be stored at. The run, its upload rows (the report, then the files, then the links, each in request order), and a `full_pipeline` job commit in **one transaction** — a failure anywhere deletes the written files and leaves no rows.

Response: `{"run_id": "<hex>", "job_id": "<hex>"}`. The job starts `QUEUED`; a single worker thread picks it up (poll interval `job_poll_seconds`). Poll `GET /api/runs/{run_id}` for progress.

**Links are read by the pipeline, not by this request.** The ingest step fetches the links one at a time, in the order they were added, before it processes any document ([architecture.md](architecture.md#fetching-links-fetchpy) describes the fetch). A link that serves a PDF becomes a PDF upload; a web page is stored as a JSON snapshot. The first link that cannot be read fails the run, and links after it are not fetched in that attempt. The run's `status` becomes `FAILED` and its `error` names the link as added. A fetch error names it as `'<target>' (redirected from '<link>')` when it failed at a redirect target. A page that redirected and then could not be read (too little readable text, bytes that cannot be decoded, too large or complex to read in time, or a reader that failed unexpectedly) keeps its error type and message, with both addresses in front: `'<final>' (redirected from '<link>'): <message>`. For example:

```
BlockedAddressError: Refusing to fetch 'http://10.0.0.5/admin': '10.0.0.5' resolves to a private or reserved network address
FetchError: Fetching 'https://example.org/missing' failed: the server answered HTTP 404
FetchError: Fetching 'https://example.org/data.zip' failed: unsupported content type 'application/zip' (a source must be an HTML page or a PDF)
FetchError: Fetching 'https://example.org/page' failed: unsupported stacked Content-Encoding 'gzip, gzip'
ThinPageError: https://example.org/app has no readable article text (<n> characters extracted, at least 250 needed) — JavaScript-only pages are not supported
ThinPageError: 'https://www.example.org/app' (redirected from 'https://example.org/app'): https://www.example.org/app has no readable article text (<n> characters extracted, at least 250 needed) — JavaScript-only pages are not supported
ExtractionTimeoutError: https://example.org/huge took longer than 60 seconds to read (the page is too large or complex)
```

`POST /api/runs/{run_id}/retry` resumes the run: a link whose PDF, or a page that loads, is already stored is not fetched again, and the pass picks up at the link that failed. A stored page that will not load (cut short, or written under an older format) is fetched again, unless a finished document was already made from it.

### `GET /api/runs`

`{"runs": [<run>, ...]}` ordered by `created_at` descending. A run object:

| Field | Meaning |
|---|---|
| `id` | Run id |
| `created_at` | ISO-8601 UTC timestamp |
| `status` | `CREATED` \| `RUNNING` \| `DONE` \| `FAILED` |
| `error` | Failure message, else `null` |
| `title` | Display title (`null` on runs created before titles existed) |
| `source_count` | Number of sources, files and links, from the latest job's payload (`null` for runs without a job) |
| `scores` | `null` until scored, else the same 0–1 shape as the report payload: `{accuracy, coverage, credibility, validity}` |

### `GET /api/runs/{run_id}`

`{"run": <run>, "job": <job> | null, "uploads": [<upload>, ...]}` — the run plus its most recent job and its uploads (report first, then sources in upload order; empty for runs without a job). 404 `Unknown run '<id>'` otherwise.

An upload object:

| Field | Meaning |
|---|---|
| `id`, `kind` | `kind` is `REPORT` or `SOURCE` |
| `file_name` | The uploaded file's name; for a link, the link |
| `source_type` | `pdf` for an uploaded file, `web` for a link. A link that turns out to serve a PDF becomes `pdf` once the ingest step has fetched it |
| `url` | The link, kept when it served a PDF; `null` for uploaded files |

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

### `POST /api/runs/{run_id}/retry` → 202

Requeues a `FAILED` run's job. The worker resumes from the first incomplete step and keeps what earlier attempts finished — ingested documents, and every link already stored (a stored page that will not load is fetched again, unless a finished document was already made from it). Response: `{"run_id": "<hex>", "job_id": "<hex>", "status": "QUEUED"}`.

404 `Unknown run '<id>'`. 409 `Run '<id>' has no job to retry`, `Run '<id>' job is <STATUS> — only FAILED runs can be retried`, or, when two retries race, `Job '<job_id>' is not FAILED — nothing to retry`.

### `DELETE /api/runs/{run_id}` → 204

Permanently deletes the run: every database row in one transaction, then the stored upload files (PDFs and web-page snapshots — for a link, every file it left, including a PDF or partial file an interrupted fetch left behind) and the run's figure directory (`<figures_dir>/<run_id>/`, outside `uploads_dir` by default), removed whole wherever it lies, ignoring any error removing it. An upload file is removed only if it lies inside `uploads_dir` and no other upload still names it: a run ingested through the CLI records the user's original file, which deletion leaves in place. Whether another upload names the file is decided after the transaction commits, by filesystem identity (device and inode), so a relative and an absolute path, a symlink, or another letter case on a case-insensitive volume all count as the same file, and a stored path that names no readable file (missing, in a symlink loop) protects nothing and never blocks the deletion. Reading those paths never holds the database's write lock: a stalled one delays only this request. 404 `Unknown run '<id>'`; 409 `Run '<id>' has a <STATUS> job — wait for it to finish or fail first` while its job is queued or running.

### `GET /api/jobs/{job_id}`

The job object above, or 404 `Unknown job '<id>'`.

## Report payload

### `GET /api/runs/{run_id}/report`

404 `Unknown run '<id>'` for unknown runs. Works at any run status — before scoring, `scores` is `null` and `claims`/`sources` may be empty. All rows are read in one transaction, so the payload is a consistent snapshot even while the worker is committing.

```json
{
  "run_id": "…",
  "title": "Water Stress Report",
  "status": "DONE",
  "report_doc_id": "…",          // the report's document id (for the file endpoint), or null
  "scores": { "accuracy": 0.84, "coverage": 0.97, "credibility": 0.509, "validity": 0.42 },
  "accuracy_detail": { "supported": 12, "contradicted": 9, "unverifiable": 16, "total": 37, "correct": 19, "incorrect": 2, "disavowed": 7 },
  "validity_detail": { "components": { "coverage": { "score": 70, "justification": "…", "quote": "…", "quote_verified": 1 }, … }, "weights_used": { … } },
  "credibility_detail": { "method": "usage_weighted_mean", "sources": [ { "doc_id": "…", "total": 62.5, "tier": "VERIFIED_DOI", "usage": 4 } ], "excluded": [] },
  "stats": { "claims_total": 37, "claims_supported": 30, "claims_contradicted": 3, "claims_unverifiable": 4 },
  "claims": [ … ],
  "sources": [ … ]
}
```

`scores` values are all 0–1 fractions (credibility and validity are stored 0–100 and divided by 100 here); `accuracy` and `coverage` can be `null` when no claim was decided. `accuracy` is report-position agreement over decided claims; UNVERIFIABLE claims count only against `coverage`, never against `accuracy`.

The three `*_detail` blocks expose what scoring stored, read-only: `accuracy_detail` is the decided-claims arithmetic (`correct`/`incorrect`/`disavowed`), `validity_detail` carries the per-component rubric (score, justification, illustrative quote, and whether code found the quote in the report), and `credibility_detail` carries the aggregation method plus each source's evidence-usage weight. All three are `null` before scoring, and individual keys are `null` on runs scored by older versions that didn't store them.

`credibility_detail.method` is `usage_weighted_mean`, `unweighted_mean_no_usage`, `no_scorable_sources`, or `no_sources`; with either of the last two, `scores.credibility` is `null`. `credibility_detail.excluded` lists sources kept out of the average as `{doc_id, reason, usage}`, where `reason` is `image` (`[]` when nothing was excluded, and on runs scored before the field existed). `no_scorable_sources` means every source was excluded.

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
| `evidence_source` | Where the quoted chunk came from, or `null` when no chunk was cited — see below |

`evidence_source` is `{"doc_id", "title", "page", "source_type", "url", "section", "start_seconds", "chunk_id"}`. `source_type` decides which locator applies: `page` for a PDF, `section` (the heading the quoted text sits under) for a web page, `start_seconds` for a time-coded source. `url` is the source's link (`null` for uploaded files), `chunk_id` is the quoted chunk, and `source_type` is `pdf` for a document with no upload row.

Each entry in `sources` — every source document of the run, scored sources first by credibility (highest first), then the rest in ingest order:

| Field | Meaning |
|---|---|
| `doc_id`, `title` | The source document |
| `source_type`, `url` | What the source is (`pdf` or `web` for anything uploaded or linked; `pdf` for documents with no upload row) and its link (`null` for uploaded files) |
| `scorable` | `false` for an `image` source, which has no bibliographic identity to score; `true` otherwise |
| `total` | Credibility score 0–100 (no floors — unknown metadata earns nothing); `null` when the source was not scored in this run |
| `tier` | Verification tier: `VERIFIED_DOI` \| `VERIFIED_TITLE` \| `VERIFIED_ISBN` \| `MATCHED_RECORD` \| `METADATA_ONLY` \| `NONE`; `null` when unscored |
| `components` | `{"metadata_completeness", "authority", "recency", "verification"}` point breakdown; `null` when unscored |
| `metadata` | The bibliographic fields the points were computed from (`title`, `authors`, `publisher`, `publication_date`, `doi`, `isbn`); `null` when unscored |
| `truncated` | `{"kept_chars", "dropped_chars"}` when the page cap (`AUTHORAI_WEB_MAX_CHARS`) read only part of a web page — what the run was scored against, and what it never saw; `null` for a page read whole and for every PDF or image |

## Document files

### `GET /api/runs/{run_id}/documents/{doc_id}/file`

Streams the stored file for a document (the report or a source) as `Content-Disposition: inline`, named by the upload's `file_name`: `filename="<file_name>"` when the name needs no percent-encoding, otherwise `filename*=utf-8''<percent-encoded file_name>`. A link's `file_name` is its URL, so a link is always served in the encoded form (`inline; filename*=utf-8''https%3A//example.org/page`). `doc_id` is a document id from the report payload (`report_doc_id`, a source's `doc_id`, or an `evidence_source.doc_id`). The media type follows the upload's `source_type`:

| `source_type` | Served file | `Content-Type` |
|---|---|---|
| `pdf` | The PDF — uploaded, or fetched from a link | `application/pdf` |
| `web` | The page's stored snapshot | `application/json` |

(The endpoint also maps `image` to `image/png` or `image/jpeg` by file extension and `youtube` to `application/json`; no upload creates either type.)

A snapshot is `{"schema": 1, "document": {"title", "sections": [{"title", "page", "text"}]}, "provenance": {...}}`: the page's readable text as sections — Markdown from trafilatura's writer with emphasis removed (list items as `- ` lines, tables as pipe rows), and `page` always `null` — plus where it came from. `provenance` carries `url` (the link as normalized at upload), `final_url` (after redirects), `fetched_at`, `content_type`, and the page's declared `title`, `authors`, `publisher`, `publication_date`, `doi`, and `scholarly` (whether the page carries `citation_*` tags). See [architecture.md](architecture.md#links-in-the-ingest-step-jobspy) for how it is produced.

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
