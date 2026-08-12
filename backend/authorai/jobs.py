"""Background pipeline jobs: one worker thread, resumable steps, startup recovery.

Design constraints, each the negation of a v1 defect:
- ONE persistent worker claiming queued jobs (v1 spawned a daemon thread per
  request), so concurrency over shared SQLite is bounded and predictable.
- The job row carries its work order (upload ids) in `payload`, so recovery
  never guesses what a run should contain.
- A RUNNING job found at startup is re-queued and RESUMED from its first
  incomplete step (v1 stranded it RUNNING forever). Steps make that safe:
  extract/verify/score are replace-semantics idempotent, and the ingest step
  reconciles per upload — a document with chunks is done; a document without
  chunks is a torn ingest (ingest_pdf's writes span transactions) and is
  deleted and re-ingested; a missing document is ingested fresh.
- The worker is single, and so is the PROCESS: startup recovery re-queues
  every RUNNING job unconditionally, which is only correct when no other
  process can be mid-job (never run uvicorn --workers N>1 against one
  database). A long verify (batch poll) blocks the queue behind it —
  acceptable for a single-user tool; the jobs table makes a second worker a
  drop-in change later, but it would need a lease column first.
"""

import shutil
import sqlite3
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

from authorai import db as dbmod
from authorai.claims import extract_claims
from authorai.config import Settings
from authorai.embeddings import OpenAIEmbedder
from authorai.ingest import FIGURE_DESCRIPTION_PROMPT, ingest_pdf
from authorai.llm import AnthropicClient
from authorai.log import setup_logger
from authorai.scoring import score_run
from authorai.verification import verify_run

logger = setup_logger(__name__)

PIPELINE_STEPS = ("ingest", "extract", "verify", "score")

# Human labels for the progress feed the frontend polls.
_STEP_LABELS = {
    "ingest": "Ingesting documents",
    "extract": "Extracting claims",
    "verify": "Verifying claims against sources",
    "score": "Scoring the report",
}


class PipelineContext:
    """Lazily-built shared clients for the real pipeline steps."""

    def __init__(self, conn: sqlite3.Connection, settings: Settings):
        self.conn = conn
        self.settings = settings
        self._embedder: OpenAIEmbedder | None = None
        self._llm: AnthropicClient | None = None

    @property
    def embedder(self) -> OpenAIEmbedder:
        if self._embedder is None:
            self._embedder = OpenAIEmbedder(
                api_key=self.settings.openai_api_key,
                model=self.settings.embedding_model,
                dim=self.settings.embedding_dim,
            )
        return self._embedder

    @property
    def llm(self) -> AnthropicClient:
        if self._llm is None:
            self._llm = AnthropicClient(self.settings.anthropic_api_key)
        return self._llm


def _chunk_count(conn: sqlite3.Connection, doc_id: str) -> int:
    return conn.execute("SELECT count(*) FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()[0]


def _reconcile_upload(context: PipelineContext, run_id: str, upload_id: str) -> None:
    """Ingest one upload, tolerating a previous torn attempt.

    ingest_pdf's writes span several transactions, so a crash can leave a
    document row with zero chunks. Chunks present -> done; document without
    chunks -> torn, delete its figures + row and re-ingest (no claims can
    exist before extract, so the FKs permit it); nothing -> ingest fresh.
    """
    conn = context.conn
    upload = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if upload is None:
        raise ValueError(f"Job payload references unknown upload {upload_id!r}")
    document = conn.execute(
        "SELECT * FROM documents WHERE run_id = ? AND upload_id = ?", (run_id, upload_id)
    ).fetchone()
    settings = context.settings
    if document is not None:
        if _chunk_count(conn, document["id"]):
            return  # completed on a previous attempt
        logger.warning("torn ingest for upload %s — re-ingesting", upload_id)
        # The figure PNGs land on disk BEFORE any DB write and re-ingest gets
        # a fresh doc_id, so the old directory would be unreferenced garbage.
        figure_dir = Path(settings.figures_dir) / run_id / document["id"]
        if figure_dir.exists():
            shutil.rmtree(figure_dir)
        with conn:
            conn.execute("DELETE FROM figures WHERE doc_id = ?", (document["id"],))
            conn.execute("DELETE FROM documents WHERE id = ?", (document["id"],))

    llm = context.llm
    caption_model = settings.caption_model

    def describe(image):
        return llm.describe_image(
            model=caption_model, image=image, prompt=FIGURE_DESCRIPTION_PROMPT
        )

    ingest_pdf(
        conn,
        context.embedder,
        run_id,
        Path(upload["path"]),
        kind=upload["kind"],
        figures_dir=settings.figures_dir,
        upload_id=upload_id,
        describe=describe,
        fallback_title=upload["file_name"],
    )


def step_ingest(context: PipelineContext, run_id: str, payload: dict) -> str:
    upload_ids = [payload["report_upload_id"], *payload["source_upload_ids"]]
    for upload_id in upload_ids:
        _reconcile_upload(context, run_id, upload_id)
    return f"Ingested {len(upload_ids)} documents"


def step_extract(context: PipelineContext, run_id: str, payload: dict) -> str:
    conn = context.conn
    report = conn.execute(
        "SELECT * FROM documents WHERE run_id = ? AND kind = 'REPORT'", (run_id,)
    ).fetchone()
    if report is None:
        raise ValueError(f"Run {run_id!r} has no REPORT document after ingest")
    import json as _json

    sections = _json.loads(report["metadata"]).get("sections", [])
    tables = dbmod.list_chunks_by_kind(conn, report["id"], "table")
    claims = extract_claims(context.llm, sections, context.settings.extraction_model, tables=tables)
    if not claims:
        # Recording an empty extraction as a green step would push the
        # failure to `verify`, whose "no claims — run extract first" message
        # would then point at a step that looks successful.
        raise ValueError(f"Extraction produced 0 claims for run {run_id!r}")
    dbmod.add_claims(
        conn, run_id, report["id"], [claim.model_dump() for claim in claims], replace=True
    )
    return f"Extracted {len(claims)} claims"


def step_verify(context: PipelineContext, run_id: str, payload: dict) -> str:
    summary = verify_run(
        context.conn,
        context.embedder,
        context.llm,
        run_id,
        model=context.settings.verdict_model,
        batch=True,
    )
    counts = summary["counts"]
    return (
        f"Verified {summary['total']} claims "
        f"({counts['SUPPORTED']} supported, {counts['CONTRADICTED']} contradicted, "
        f"{counts['UNVERIFIABLE']} unverifiable)"
    )


def step_score(context: PipelineContext, run_id: str, payload: dict) -> str:
    result = score_run(context.conn, context.llm, run_id, context.settings)
    return (
        f"accuracy {result['accuracy']['accuracy']}, "
        f"credibility {result['credibility']['score']}, "
        f"validity {result['validity']['score']}"
    )


REAL_STEPS: dict[str, Callable] = {
    "ingest": step_ingest,
    "extract": step_extract,
    "verify": step_verify,
    "score": step_score,
}


def run_job(
    conn: sqlite3.Connection,
    settings: Settings,
    job: dict,
    steps: dict[str, Callable] | None = None,
) -> None:
    """Execute one claimed job, resuming from its first incomplete step.

    Terminal writes are atomic over (job, run) and the DONE write happens
    OUTSIDE the failure handler: if that last write itself fails, the job
    stays RUNNING for startup recovery to re-queue — every step is already
    recorded done, so the retry only repeats the finish. A successful run is
    never rewritten as FAILED by a bookkeeping failure.
    """
    steps = steps if steps is not None else REAL_STEPS
    context = PipelineContext(conn, settings)
    run_id = job["run_id"]
    try:
        completed = {p["step"] for p in job["progress"] if p["status"] == "done"}
        dbmod.set_run_status(conn, run_id, "RUNNING")
        for name in PIPELINE_STEPS:
            if name in completed:
                continue
            dbmod.push_job_progress(conn, job["id"], name, _STEP_LABELS[name], status="running")
            label = steps[name](context, run_id, job["payload"])
            dbmod.push_job_progress(conn, job["id"], name, label, status="done")
    except Exception as exc:  # noqa: BLE001 - recorded on the job, run marked FAILED
        logger.exception("job %s failed", job["id"])
        error = f"{type(exc).__name__}: {exc}"
        try:
            # Flip the in-flight step to failed so the progress feed shows where.
            for entry in dbmod.get_job(conn, job["id"])["progress"]:
                if entry["status"] == "running":
                    dbmod.push_job_progress(
                        conn, job["id"], entry["step"], entry["label"], status="failed"
                    )
            dbmod.finish_job_and_run(conn, job["id"], run_id, "FAILED", error=error)
        except Exception:  # noqa: BLE001
            # The handler failing must not kill the worker; the job stays
            # RUNNING and startup recovery re-queues it.
            logger.critical(
                "job %s: recording the failure (%s) itself failed", job["id"], error, exc_info=True
            )
        logger.debug("traceback:\n%s", traceback.format_exc())
        return
    dbmod.finish_job_and_run(conn, job["id"], run_id, "DONE")


class Worker:
    """The single background worker: claim → run → repeat."""

    def __init__(self, settings: Settings, steps: dict[str, Callable] | None = None):
        self._settings = settings
        self._steps = steps
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_pending(self, conn: sqlite3.Connection) -> int:
        """Drain the queue synchronously; returns jobs run (test/deterministic path)."""
        count = 0
        while (job := dbmod.claim_next_job(conn)) is not None:
            run_job(conn, self._settings, job, steps=self._steps)
            count += 1
        return count

    def _loop(self) -> None:
        # The worker owns its connection — sqlite3 objects are thread-bound.
        try:
            conn = dbmod.connect(self._settings.db_path, self._settings.embedding_dim)
        except Exception:
            # This thread is the only thing that runs jobs; dying silently at
            # startup would leave every job QUEUED while /health reports ok.
            logger.critical("worker could not open the database — NO jobs will run", exc_info=True)
            raise
        try:
            while not self._stop.is_set():
                try:
                    ran = self.run_pending(conn)
                except Exception:  # noqa: BLE001
                    # The ONLY worker thread must survive anything run_job
                    # lets escape (e.g. a lock timeout on the final DONE
                    # write) — a dead thread would leave every future job
                    # QUEUED forever while /health still reports ok.
                    logger.critical("worker loop error — worker still alive", exc_info=True)
                    ran = 0
                if not ran:
                    self._stop.wait(self._settings.job_poll_seconds)
        finally:
            conn.close()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("worker is already running")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="authorai-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
