"""Authenticated HTTP API.

Auth is structural: every route on this router inherits `require_api_key`
through the router's dependencies, so a new endpoint cannot forget it (v1
authenticated per-route with decorators and missed some). The dependency
fails CLOSED — no configured key means 401, never open access.
"""

import secrets
import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, Security, UploadFile
from fastapi.security import APIKeyHeader

from authorai import db as dbmod
from authorai.log import setup_logger

logger = setup_logger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(request: Request, provided: str | None = Security(_api_key_header)) -> None:
    expected = request.app.state.settings.api_key
    if not expected or provided is None or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_conn(request: Request):
    """One connection per request, never shared between concurrent requests.

    check_same_thread is off because FastAPI runs this dependency and the
    endpoint on different threadpool threads — sequentially, one at a time,
    which is the one arrangement where crossing threads is safe.
    """
    settings = request.app.state.settings
    conn = dbmod.connect(settings.db_path, settings.embedding_dim, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])

Conn = Annotated[sqlite3.Connection, Depends(get_conn)]


def _validate_pdf(file_name: str | None, data: bytes, max_bytes: int) -> None:
    if not file_name or not file_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"{file_name!r} is not a .pdf file")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"{file_name!r} exceeds the {max_bytes} byte upload limit"
        )
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail=f"{file_name!r} is not PDF content")


@router.post("/runs", status_code=202)
async def create_run(
    request: Request,
    report: Annotated[UploadFile, File()],
    sources: Annotated[list[UploadFile], File()],
    conn: Conn,
) -> dict:
    """Accept a report + its sources and queue the full pipeline.

    EVERY file is validated before ANY row is written — v1 inserted the queue
    row first and stranded it when a later file failed validation.
    """
    settings = request.app.state.settings
    validated: list[tuple[str, str, bytes]] = []
    for kind, upload in [("REPORT", report)] + [("SOURCE", s) for s in sources]:
        data = await upload.read()
        _validate_pdf(upload.filename, data, settings.max_upload_bytes)
        validated.append((kind, upload.filename, data))

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    upload_ids: dict[str, list[str]] = {"REPORT": [], "SOURCE": []}
    for kind, file_name, data in validated:
        # Server-generated disk names: client filenames never touch the path.
        path = settings.uploads_dir / f"{dbmod.new_id()}.pdf"
        path.write_bytes(data)
        upload_ids[kind].append(dbmod.add_upload(conn, kind, file_name, str(path)))

    run_id = dbmod.create_run(conn)
    job_id = dbmod.create_job(
        conn,
        run_id,
        {"report_upload_id": upload_ids["REPORT"][0], "source_upload_ids": upload_ids["SOURCE"]},
    )
    return {"run_id": run_id, "job_id": job_id}


@router.get("/runs")
def list_runs(conn: Conn) -> dict:
    return {"runs": dbmod.list_runs(conn)}


@router.get("/runs/{run_id}")
def get_run(run_id: str, conn: Conn) -> dict:
    run = dbmod.get_run(conn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run {run_id!r}")
    return {"run": run, "job": dbmod.get_run_job(conn, run_id)}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, conn: Conn) -> dict:
    job = dbmod.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job {job_id!r}")
    return job


def _fraction(score: float | None) -> float | None:
    """0–100 component scores leave the API as 0–1 fractions like accuracy."""
    return None if score is None else round(score / 100, 4)


@router.get("/runs/{run_id}/report")
def get_report(run_id: str, conn: Conn) -> dict:
    run = dbmod.get_run(conn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run {run_id!r}")

    verdict_rows = dbmod.list_verdicts_with_evidence(conn, run_id)
    stats = {
        "claims_total": len(verdict_rows),
        "claims_supported": sum(1 for r in verdict_rows if r["verdict"] == "SUPPORTED"),
        "claims_contradicted": sum(1 for r in verdict_rows if r["verdict"] == "CONTRADICTED"),
        "claims_unverifiable": sum(1 for r in verdict_rows if r["verdict"] == "UNVERIFIABLE"),
    }

    stored = dbmod.get_run_scores(conn, run_id)
    scores = None
    if stored is not None:
        scores = {
            "accuracy": stored["accuracy"]["accuracy"],
            "coverage": stored["accuracy"]["coverage"],
            "credibility": _fraction(stored["credibility"]["score"]),
            "validity": _fraction(stored["validity"]["score"]),
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
        }
        for row in dbmod.list_source_credibility(conn, run_id)
    ]

    return {
        "run_id": run_id,
        "status": run["status"],
        "scores": scores,
        "stats": stats,
        "claims": claims,
        "sources": sources,
    }
