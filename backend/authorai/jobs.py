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

import codecs
import hashlib
import shutil
import sqlite3
import threading
import traceback
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from authorai import db as dbmod
from authorai.claims import claims_as_rows, extract_claims
from authorai.config import Settings
from authorai.embeddings import OpenAIEmbedder
from authorai.fetch import FetchedResponse, _where, fetch_url
from authorai.ingest import (
    FIGURE_DESCRIPTION_PROMPT,
    ingest_pdf,
    ingest_snapshot,
    load_snapshot,
    replaced_atomically,
    write_snapshot,
)
from authorai.llm import AnthropicClient, StaleBatchError
from authorai.log import setup_logger
from authorai.scoring import score_run
from authorai.verification import verify_run
from authorai.web import ExtractionTimeoutError, ThinPageError, browser_codec, extract_web_bounded

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


def _copy_ingested_document(
    context: PipelineContext, run_id: str, upload: sqlite3.Row, donor: dict
) -> None:
    """Copy a donor document's ingest output into this run — files, then rows.

    PNGs are copied BEFORE any DB write (verify attaches them, so a row must
    never point at a missing file; a vanished donor PNG raises loudly). The
    rows then land in copy_document_data's single transaction, so a failure
    after the files leaves no rows — the fresh directory is removed on the
    way out and a retry starts clean.
    """
    conn = context.conn
    doc_id = dbmod.new_id()
    donor_figures = conn.execute(
        "SELECT id, image_path FROM figures WHERE doc_id = ? ORDER BY rowid", (donor["doc_id"],)
    ).fetchall()
    # Resolved, like ingest_pdf's own figure dir — an unresolved default
    # ('data/figures') would store CWD-relative image paths that break
    # verification when the server starts from a different directory.
    figure_dir = context.settings.run_figures_dir(run_id) / doc_id
    figure_map: dict[str, tuple[str, str]] = {}
    try:
        if donor_figures:
            figure_dir.mkdir(parents=True, exist_ok=True)
        for row in donor_figures:
            target = figure_dir / Path(row["image_path"]).name
            shutil.copyfile(row["image_path"], target)
            figure_map[row["id"]] = (dbmod.new_id(), str(target))
        dbmod.copy_document_data(
            conn,
            donor["doc_id"],
            run_id=run_id,
            doc_id=doc_id,
            upload_id=upload["id"],
            kind=upload["kind"],
            figure_map=figure_map,
            # Same rule as a fresh ingest: an API upload's real name, not its
            # generated on-disk name.
            fallback_title=Path(upload["file_name"]).stem,
        )
    except Exception:
        if figure_dir.exists():
            shutil.rmtree(figure_dir)
        raise


def _maybe_reuse_ingest(context: PipelineContext, run_id: str, upload: sqlite3.Row) -> bool:
    """Reuse a previous ingest of these exact bytes, when one is safe to copy.

    Falls through to a fresh ingest when there is no hash (legacy/CLI
    upload) or no complete donor made by the CURRENT embedding model (the
    donor filter is per document, so vectors from another model can never
    answer for this one, and unstamped pre-dedup documents never donate).
    A copy that FAILS — donor deleted mid-copy, a donor PNG lost from disk —
    also falls through: dedup is an optimization, and no optimization
    failure may fail an ingest that recomputing would complete (the failed
    attempt already cleaned its files and rolled back its rows). On reuse
    the donor's kind is irrelevant — the copy is written under THIS
    upload's kind.
    """
    if not upload["content_hash"]:
        return False
    donor = dbmod.find_ingest_donor(
        context.conn,
        upload["content_hash"],
        embedding_model=context.settings.embedding_model,
        exclude_upload_id=upload["id"],
    )
    if donor is None:
        return False
    try:
        _copy_ingested_document(context, run_id, upload, donor)
    except Exception:
        logger.warning(
            "reusing document %s for upload %s failed — ingesting fresh instead",
            donor["doc_id"],
            upload["id"],
            exc_info=True,
        )
        return False
    logger.info(
        "reused ingest for upload %s from document %s (run %s)",
        upload["id"],
        donor["doc_id"],
        donor["run_id"],
    )
    return True


def _reconcile_upload(context: PipelineContext, run_id: str, upload_id: str) -> bool:
    """Ingest one upload, tolerating a previous torn attempt.

    ingest_pdf's writes span several transactions, so a crash can leave a
    document row with zero chunks. Chunks present -> done; document without
    chunks -> torn, delete its figures + row and re-ingest (no claims can
    exist before extract, so the FKs permit it); nothing -> reuse a previous
    ingest of byte-identical bytes if one exists, else ingest fresh.

    Returns True when THIS attempt reused a previous ingest — already-done
    and fresh both return False, so the step label counts only this
    attempt's actual reuse.
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
            return False  # completed on a previous attempt
        logger.warning("torn ingest for upload %s — re-ingesting", upload_id)
        # The figure PNGs land on disk BEFORE any DB write and re-ingest gets
        # a fresh doc_id, so the old directory would be unreferenced garbage.
        figure_dir = settings.run_figures_dir(run_id) / document["id"]
        if figure_dir.exists():
            shutil.rmtree(figure_dir)
        with conn:
            conn.execute("DELETE FROM figures WHERE doc_id = ?", (document["id"],))
            conn.execute("DELETE FROM documents WHERE id = ?", (document["id"],))

    # The dedup check MUST precede context.llm/context.embedder: a fully
    # reused ingest constructs no provider client, so it needs no API key —
    # which is also the proof that reuse recomputes nothing.
    if _maybe_reuse_ingest(context, run_id, upload):
        return True

    if upload["source_type"] in LINK_SOURCE_TYPES:
        # A link's page was stored by the fetch pre-pass; ingesting it needs no
        # figure captions (a stored page carries sections only).
        ingest_snapshot(
            conn,
            context.embedder,
            run_id,
            Path(upload["path"]),
            kind=upload["kind"],
            figures_dir=settings.figures_dir,
            upload_id=upload_id,
            # A link's display name IS the link: no filename stem to strip.
            fallback_title=upload["file_name"],
        )
        return False

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
        # Stem, not the raw name — so an API upload titles the same as the CLI
        # path (path.stem), not "report.pdf" vs "report".
        fallback_title=Path(upload["file_name"]).stem,
    )
    return False


# Upload source types whose content is fetched by the ingest step itself.
LINK_SOURCE_TYPES = ("web", "youtube")


def _fetch_pending_links(context: PipelineContext, run_id: str, upload_ids: list[str]) -> int:
    """Fetch every link whose page is not stored yet, BEFORE any document is
    processed: a link that cannot be read fails the run in seconds, not after
    the report's parse, figure captions, and embeddings have spent money.
    A stored page that loads is never fetched again, so a retry resumes where
    it stopped. One that will not load (cut short by a power loss, or written
    under an older snapshot schema) is fetched again — otherwise every retry
    would fail on it identically — unless a finished document was already cut
    from it: that document's chunks came from the page as stored, so it stays.
    Returns how many links this attempt fetched."""
    conn = context.conn
    fetched = 0
    for upload_id in upload_ids:
        upload = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
        if upload is None or upload["source_type"] not in LINK_SOURCE_TYPES:
            continue  # an unknown upload id is reported loudly by _reconcile_upload
        if upload["source_type"] == "youtube":
            raise ValueError(f"YouTube sources are not supported yet: {upload['url']!r}")
        path = Path(upload["path"])
        if path.exists():
            problem = _snapshot_problem(path)
            if problem is None:
                continue
            if _has_finished_document(conn, run_id, upload_id):
                logger.info(
                    "stored page for %s will not load (%s) — kept: a finished document was cut "
                    "from it",
                    upload["url"],
                    problem,
                )
                continue
            # The page is deleted next, so this line is the only record of why.
            logger.warning(
                "stored page for %s will not load (%s) — fetching it again", upload["url"], problem
            )
            path.unlink()
        _fetch_link(context, upload)
        fetched += 1
    return fetched


def _snapshot_problem(path: Path) -> str | None:
    """Why a stored page will not load — load_snapshot's message, which names the
    file and tells a cut-short write from an older schema or an unreadable file —
    or None when it loads."""
    try:
        load_snapshot(path)
    except ValueError as exc:
        return str(exc)
    return None


def _has_finished_document(conn: sqlite3.Connection, run_id: str, upload_id: str) -> bool:
    document = conn.execute(
        "SELECT id FROM documents WHERE run_id = ? AND upload_id = ?", (run_id, upload_id)
    ).fetchone()
    return document is not None and bool(_chunk_count(conn, document["id"]))


def _fetch_link(context: PipelineContext, upload: sqlite3.Row) -> None:
    """Fetch one link and store what it serves, files first.

    A page becomes a snapshot at the planned path — never hashed: link pages do
    not dedup. A PDF is stored beside it at .pdf and recorded (record_fetch) as
    a PDF upload with its content hash, so it is ingested, and dedups, exactly
    like an uploaded PDF. The page is read out of process under a wall-clock
    budget (a page can hold trafilatura for hours). FetchError, ThinPageError
    and ExtractionTimeoutError propagate: the step fails with the link named —
    after a redirect, as '<final>' (redirected from '<link>') like fetch.py's
    own errors — and the retry fetches again.
    """
    fetched = fetch_url(upload["url"], context.settings)
    planned = Path(upload["path"])
    planned.parent.mkdir(parents=True, exist_ok=True)
    if fetched.is_pdf:
        target = planned.with_suffix(".pdf")
        with replaced_atomically(target) as part:
            part.write_bytes(fetched.body)
        dbmod.record_fetch(
            context.conn,
            upload["id"],
            path=str(target),
            source_type="pdf",
            content_hash=hashlib.sha256(fetched.body).hexdigest(),
        )
        _drop_other_link_artifacts(planned, keep=target)
        return
    text = _page_text(fetched)
    try:
        parsed, page = extract_web_bounded(
            text, url=fetched.final_url, timeout=context.settings.extract_timeout_seconds
        )
    except (ValueError, RuntimeError) as exc:
        if fetched.final_url == upload["url"]:
            raise  # the reader's message already names the link as added
        raise _redirected(exc, fetched.final_url, upload["url"]) from exc
    provenance = {
        "url": upload["url"],
        "final_url": fetched.final_url,
        "fetched_at": dbmod.now_iso(),
        "content_type": fetched.content_type,
        **asdict(page),
    }
    write_snapshot(planned, parsed, provenance)
    _drop_other_link_artifacts(planned, keep=planned)


def _drop_other_link_artifacts(planned: Path, *, keep: Path) -> None:
    """Remove what an earlier attempt at this link left beside the file its row
    now names: a PDF stored but never recorded, a page the link no longer serves,
    a .part never renamed. Only once the row names the kept file, so no row ever
    points at nothing; no row names the others (a stored PDF's row is never
    fetched again), and deletion is refused while the job runs."""
    for path in dbmod.link_artifact_paths(planned):
        if path != keep:
            path.unlink(missing_ok=True)


# What reading a page raises, most specific first.
_READING_ERRORS = (ThinPageError, ExtractionTimeoutError, ValueError, RuntimeError)


def _redirected(exc: Exception, final_url: str, added_url: str) -> Exception:
    """A reading error that names the link as added, not only where it ended up:
    the UI flags a source row only when the run's error names that row's link.
    Same kind and original message, so the error's type name and the failure
    hints still match — built from the known class, never type(exc)(...), since a
    ValueError subclass such as UnicodeDecodeError takes other arguments."""
    kind = next(cls for cls in _READING_ERRORS if isinstance(exc, cls))
    return kind(f"{_where(added_url, final_url)}: {exc}")


def _page_text(fetched: FetchedResponse) -> bytes | str:
    """The page as extract_web should read it. Raw bytes by default —
    extract_web honors a byte-order mark, valid UTF-8, and a <meta> charset —
    except when the bytes are NOT UTF-8, carry no byte-order mark, and the
    HTTP Content-Type names a text encoding: the one declaration only the
    fetch has seen. A label that names no text encoding leaves the bytes to
    extract_web, which fails naming the link when nothing else declares one."""
    if fetched.charset is None or fetched.body.startswith(
        (codecs.BOM_UTF8, codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)
    ):
        return fetched.body  # WHATWG: a byte-order mark beats the header
    try:
        fetched.body.decode("utf-8")
        return fetched.body
    except UnicodeDecodeError:
        pass
    try:
        info = codecs.lookup(fetched.charset)
    except (LookupError, ValueError):  # ValueError: a label holding a NUL byte
        return fetched.body  # an unknown label: the page's own declaration decides
    codec = browser_codec(info)
    if codec is None:
        return fetched.body  # a byte-to-byte codec: no charset a browser knows either
    try:
        return fetched.body.decode(codec, errors="replace")
    except UnicodeError:
        return fetched.body  # a text codec that refuses any byte ("undefined", "idna")


def step_ingest(context: PipelineContext, run_id: str, payload: dict) -> str:
    upload_ids = [payload["report_upload_id"], *payload["source_upload_ids"]]
    fetched = _fetch_pending_links(context, run_id, upload_ids)
    reused = sum(_reconcile_upload(context, run_id, upload_id) for upload_id in upload_ids)
    # Shown verbatim under the finished step, so it counts in the reader's words.
    label = f"Read {len(upload_ids)} documents"
    notes = []
    if fetched:  # links, not pages: a link that served a PDF counts too
        notes.append(f"{fetched} {'link' if fetched == 1 else 'links'} opened")
    if reused:
        notes.append(f"{reused} already read")
    return f"{label} ({', '.join(notes)})" if notes else label


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
    dbmod.add_claims(conn, run_id, report["id"], claims_as_rows(claims), replace=True)
    return f"Extracted {len(claims)} claims"


def step_verify(context: PipelineContext, run_id: str, payload: dict) -> str:
    # A verify batch is paid work: its id is persisted on the job the moment
    # it is submitted, so a retry after a timeout resumes the SAME batch
    # instead of paying for a duplicate (the id survives in the payload; a
    # stale or mismatched batch falls back to a fresh submission loudly).
    job = dbmod.get_run_job(context.conn, run_id)

    def remember_batch(batch_id: str) -> None:
        if job is not None:
            dbmod.update_job_payload(context.conn, job["id"], {"verify_batch_id": batch_id})

    try:
        summary = verify_run(
            context.conn,
            context.embedder,
            context.llm,
            run_id,
            model=context.settings.verdict_model,
            batch=True,
            resume_batch_id=payload.get("verify_batch_id"),
            on_batch_created=remember_batch,
        )
    except StaleBatchError:
        # The stored batch belongs to a different claim set. Clear it, or
        # every retry re-resolves the same dead batch and fails identically —
        # with the key gone, the next retry submits fresh.
        if job is not None:
            dbmod.update_job_payload(context.conn, job["id"], {"verify_batch_id": None})
        raise
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
