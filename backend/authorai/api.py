"""Authenticated HTTP API.

Auth is enforced by a pure-ASGI middleware (`ApiGuardMiddleware`) that runs
BEFORE the request body is parsed: it rejects any `/api` request without the
key and any request whose declared Content-Length exceeds the cap. Doing this
in the route dependency (as v1-style per-route auth would) is too late —
FastAPI parses the whole multipart body first, so an unauthenticated client
could push gigabytes before the 401. The middleware guards the `/api` prefix
as a whole, so a new endpoint cannot forget it, and it fails CLOSED (no
configured key ⇒ 401).
"""

import hashlib
import json
import secrets
import shutil
import sqlite3
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from authorai import chat as chatmod
from authorai import db as dbmod
from authorai.config import Settings
from authorai.llm import AnthropicClient
from authorai.log import setup_logger

logger = setup_logger(__name__)

API_PREFIX = "/api"


def _key_ok(provided: bytes | None, expected: str | None) -> bool:
    """Constant-time key check on RAW BYTES — comparing decoded strings raises
    TypeError on any non-ASCII header value, turning a bad key into a 500."""
    if not expected or provided is None:
        return False
    return secrets.compare_digest(provided, expected.encode("utf-8"))


class ApiGuardMiddleware:
    """Enforces auth and the request-size cap before the body is read."""

    def __init__(self, app, settings: Settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        if path == API_PREFIX or path.startswith(API_PREFIX + "/"):
            headers = dict(scope["headers"])  # lowercased byte keys
            if not _key_ok(headers.get(b"x-api-key"), self.settings.api_key):
                logger.warning("unauthorized %s %s", scope.get("method"), path)
                await self._reject(send, 401, "Invalid or missing API key")
                return
            length = headers.get(b"content-length")
            if length and length.isdigit() and int(length) > self.settings.max_request_bytes:
                await self._reject(send, 413, "Request body exceeds the size limit")
                return
        await self.app(scope, receive, send)

    async def _reject(self, send, status: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def get_conn(request: Request):
    """One connection per request, never shared between concurrent requests.

    check_same_thread is off because FastAPI runs this dependency and the
    endpoint on the threadpool — sequentially, one at a time, which is the one
    arrangement where crossing threads is safe.
    """
    settings = request.app.state.settings
    conn = dbmod.connect(settings.db_path, settings.embedding_dim, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


router = APIRouter(prefix=API_PREFIX)

Conn = Annotated[sqlite3.Connection, Depends(get_conn)]


def _validate_pdf(upload: UploadFile, max_bytes: int) -> None:
    """Extension, size, and magic — WITHOUT reading the file into memory.

    `.size` comes from the already-spooled part, and the magic check reads
    only the first 5 bytes; the full bytes are never materialized (v1's cap
    fired only after the whole file was resident, so the cap was decorative).
    """
    name = upload.filename
    if not name or not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"{name!r} is not a .pdf file")
    if upload.size is not None and upload.size > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"{name!r} exceeds the {max_bytes} byte per-file limit"
        )
    upload.file.seek(0)
    head = upload.file.read(5)
    upload.file.seek(0)
    if head != b"%PDF-":
        raise HTTPException(status_code=400, detail=f"{name!r} is not PDF content")


@router.post("/runs", status_code=202)
def create_run(
    request: Request,
    report: Annotated[UploadFile, File()],
    sources: Annotated[list[UploadFile], File()],
    conn: Conn,
    title: Annotated[str | None, Form()] = None,
) -> dict:
    """Accept a report + its sources and queue the full pipeline.

    A sync endpoint (runs on the threadpool, so its blocking file/DB writes
    never freeze the event loop). EVERY file is validated before ANY file is
    written, and the run/uploads/job rows all commit in ONE transaction — a
    failure anywhere leaves nothing behind (v1 stranded rows and blobs).
    """
    settings: Settings = request.app.state.settings
    if len(sources) > settings.max_source_files:
        raise HTTPException(
            status_code=400,
            detail=f"too many source files ({len(sources)} > {settings.max_source_files})",
        )
    uploads = [("REPORT", report)] + [("SOURCE", s) for s in sources]
    for _kind, upload in uploads:
        _validate_pdf(upload, settings.max_upload_bytes)
    # The run's display title: the dialog's Name field when given, else the
    # report filename stem (filename is validated non-empty by _validate_pdf).
    # Capped explicitly — the only other bound is starlette's incidental 1MB
    # part limit, which would let a megabyte of title ride every gallery load.
    if title is not None and len(title.strip()) > 200:
        raise HTTPException(status_code=400, detail="title is limited to 200 characters")
    title = ((title or "").strip() or Path(report.filename).stem or report.filename)[:200]

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        rows: list[tuple[str, str, str, str | None]] = []
        for kind, upload in uploads:
            # Server-generated disk name: the client filename never touches the
            # path (kept only as the uploads.file_name column for display).
            path = settings.uploads_dir / f"{dbmod.new_id()}.pdf"
            upload.file.seek(0)
            # Hash while writing — the content fingerprint that lets a later
            # run reuse this file's ingested data instead of recomputing it.
            hasher = hashlib.sha256()
            with path.open("wb") as handle:
                while block := upload.file.read(1024 * 1024):
                    hasher.update(block)
                    handle.write(block)
            written.append(path)
            rows.append((kind, upload.filename, str(path), hasher.hexdigest()))
        run_id, job_id = dbmod.create_run_with_uploads_and_job(conn, rows, title=title)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return {"run_id": run_id, "job_id": job_id}


@router.get("/runs")
def list_runs(conn: Conn) -> dict:
    """Run list enriched for the gallery: title, source count, and a scores
    summary in the same 0–1 shape the report endpoint uses (null until
    scored)."""
    items = []
    for item in dbmod.list_runs_enriched(conn):
        stored = item.pop("scores")
        item["scores"] = (
            {
                "accuracy": stored["accuracy"]["accuracy"],
                "coverage": stored["accuracy"]["coverage"],
                "credibility": _fraction(stored["credibility"]["score"]),
                "validity": _fraction(stored["validity"]["score"]),
            }
            if stored is not None
            else None
        )
        items.append(item)
    return {"runs": items}


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(run_id: str, request: Request, conn: Conn) -> None:
    """Permanently delete a run: every database row plus the uploaded PDFs
    and figure images behind it. Refused (409) while its job is queued or
    running — the worker would otherwise write into deleted rows mid-step."""
    settings: Settings = request.app.state.settings
    try:
        upload_paths = dbmod.delete_run_data(conn, run_id)
    except dbmod.RunBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # Files AFTER the commit: a crash here leaves harmless orphan files,
    # never database rows pointing at missing ones.
    for path in upload_paths:
        Path(path).unlink(missing_ok=True)
    shutil.rmtree(settings.run_figures_dir(run_id), ignore_errors=True)


@router.get("/runs/{run_id}")
def get_run(run_id: str, conn: Conn) -> dict:
    run = dbmod.get_run(conn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run {run_id!r}")
    return {
        "run": run,
        "job": dbmod.get_run_job(conn, run_id),
        "uploads": dbmod.list_run_uploads(conn, run_id),
    }


@router.post("/runs/{run_id}/retry", status_code=202)
def retry_run(run_id: str, conn: Conn) -> dict:
    """Requeue a FAILED run's job; the worker resumes from the first
    incomplete step, keeping already-ingested documents (a transient failure
    like a dropped connection should not cost the whole ingest again)."""
    if dbmod.get_run(conn, run_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown run {run_id!r}")
    job = dbmod.get_run_job(conn, run_id)
    if job is None:
        raise HTTPException(status_code=409, detail=f"Run {run_id!r} has no job to retry")
    if job["status"] != "FAILED":
        raise HTTPException(
            status_code=409,
            detail=f"Run {run_id!r} job is {job['status']} — only FAILED runs can be retried",
        )
    try:
        dbmod.requeue_job(conn, job["id"], run_id)
    except ValueError as exc:
        # Two overlapping retries both read FAILED; the loser's guarded UPDATE
        # matches nothing. That's a conflict, not a server error.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run_id": run_id, "job_id": job["id"], "status": "QUEUED"}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, conn: Conn) -> dict:
    job = dbmod.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job {job_id!r}")
    return job


@router.get("/runs/{run_id}/documents/{doc_id}/file")
def get_document_file(run_id: str, doc_id: str, request: Request, conn: Conn) -> FileResponse:
    """Stream a run's stored PDF (report or a source) for inline viewing.

    Access is scoped by (run_id, doc_id): a doc from another run resolves to
    nothing. The served path comes only from the uploads table (a
    server-generated name), never the client — but it is still resolve-checked
    to be inside uploads_dir as defense against a tampered DB or a symlink.
    """
    settings: Settings = request.app.state.settings
    resolved = dbmod.get_document_path(conn, run_id, doc_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail="No such document in this run")
    path_str, file_name = resolved
    path = Path(path_str).resolve()
    uploads_root = settings.uploads_dir.resolve()
    if not path.is_relative_to(uploads_root) or not path.is_file():
        raise HTTPException(status_code=404, detail="Document file is unavailable")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=file_name,
        content_disposition_type="inline",
    )


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    # Bounded so a chat request can't be a memory-amplification vector or a
    # single very expensive model call — the 220MB Content-Length cap is
    # sized for PDF uploads, far too large for a question.
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=50)
    mode: Literal["evidence", "guidance", "creative"] = "evidence"


@router.post("/runs/{run_id}/chat")
def chat(run_id: str, body: ChatRequest, request: Request, conn: Conn) -> dict:
    """Answer a question grounded in a DONE run's analysis (prompt-cached)."""
    run = dbmod.get_run(conn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run {run_id!r}")
    if run["status"] != "DONE":
        raise HTTPException(
            status_code=409, detail="The run is not scored yet — chat is available once it is DONE"
        )
    settings: Settings = request.app.state.settings
    llm = AnthropicClient(settings.anthropic_api_key)
    reply = chatmod.answer(
        conn,
        llm,
        run_id,
        body.question,
        [turn.model_dump() for turn in body.history],
        body.mode,
        settings,
    )
    return {"answer": reply, "mode": body.mode}


def _fraction(score: float | None) -> float | None:
    """0–100 component scores leave the API as 0–1 fractions like accuracy."""
    return None if score is None else round(score / 100, 4)


@router.get("/runs/{run_id}/report")
def get_report(run_id: str, conn: Conn) -> dict:
    # One read transaction so the verdicts, scores, and per-source rows are a
    # consistent snapshot — the worker may be committing all three while the
    # frontend polls, and separate autocommit reads can straddle that write.
    conn.execute("BEGIN")
    try:
        run = dbmod.get_run(conn, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run {run_id!r}")

        verdict_rows = dbmod.list_verdicts_with_evidence(conn, run_id)
        stored = dbmod.get_run_scores(conn, run_id)
        source_rows = dbmod.list_source_credibility(conn, run_id)
        report_doc_id = dbmod.get_report_doc_id(conn, run_id)
    finally:
        conn.rollback()  # read-only; release the snapshot

    stats = {
        "claims_total": len(verdict_rows),
        "claims_supported": sum(1 for r in verdict_rows if r["verdict"] == "SUPPORTED"),
        "claims_contradicted": sum(1 for r in verdict_rows if r["verdict"] == "CONTRADICTED"),
        "claims_unverifiable": sum(1 for r in verdict_rows if r["verdict"] == "UNVERIFIABLE"),
    }

    scores = None
    accuracy_detail = validity_detail = credibility_detail = None
    if stored is not None:
        scores = {
            "accuracy": stored["accuracy"]["accuracy"],
            "coverage": stored["accuracy"]["coverage"],
            "credibility": _fraction(stored["credibility"]["score"]),
            "validity": _fraction(stored["validity"]["score"]),
        }
        # Read-only exposure of what scoring already stored — .get()-safe
        # because rows scored before a field existed simply lack the key.
        accuracy_detail = {
            key: stored["accuracy"].get(key)
            for key in (
                "supported",
                "contradicted",
                "unverifiable",
                "total",
                "correct",
                "incorrect",
                "disavowed",
            )
        }
        validity_detail = {
            "components": stored["validity"].get("components"),
            "weights_used": stored["validity"].get("weights_used"),
        }
        credibility_detail = {
            "method": stored["credibility"].get("method"),
            "sources": stored["credibility"].get("sources"),
        }

    claims = [
        {
            "claim_id": r["claim_id"],
            "text": r["text"],
            "page": r["page"],
            "value": r["value"],
            "unit": r["unit"],
            "year": r["year"],
            "verdict": r["verdict"],
            "stance": r["stance"],
            "downgraded": dbmod.is_downgraded(r),
            "quote": r["quote"],
            "quote_verified": r["quote_verified"],
            "rationale": r["rationale"],
            "year_flag": r["year_flag"],
            "evidence_source": (
                {
                    "doc_id": r["evidence_doc_id"],
                    "title": r["evidence_doc_title"],
                    "page": r["evidence_page"],
                }
                if r["evidence_doc_id"]
                else None
            ),
        }
        for r in verdict_rows
    ]

    sources = [
        {
            "doc_id": row["doc_id"],
            "title": row["doc_title"],
            "total": row["total"],
            "tier": row["tier"],
            "components": row["components"],
            # The extracted bibliographic fields — what the completeness /
            # authority / recency points were actually computed from.
            "metadata": row["metadata"],
        }
        for row in source_rows
    ]

    return {
        "run_id": run_id,
        "title": run["title"],
        "status": run["status"],
        "report_doc_id": report_doc_id,
        "scores": scores,
        "accuracy_detail": accuracy_detail,
        "validity_detail": validity_detail,
        "credibility_detail": credibility_detail,
        "stats": stats,
        "claims": claims,
        "sources": sources,
    }
