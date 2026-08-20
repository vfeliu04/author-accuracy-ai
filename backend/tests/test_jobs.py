"""Job worker tests: claiming, resume, failure, and torn-ingest recovery."""

import inspect
import time

import pytest

from authorai import db as dbmod
from authorai.config import Settings
from authorai.ingest import ingest_pdf
from authorai.jobs import (
    PIPELINE_STEPS,
    REAL_STEPS,
    PipelineContext,
    Worker,
    _reconcile_upload,
    run_job,
)
from tests.conftest import DIM

SETTINGS = Settings(anthropic_api_key="x", openai_api_key="x")


def _job(conn, payload=None):
    run_id = dbmod.create_run(conn)
    job_id = dbmod.create_job(
        conn, run_id, payload or {"report_upload_id": "r", "source_upload_ids": []}
    )
    return run_id, job_id


def _fake_steps(record, fail_at=None):
    def make(name):
        def step(context, run_id, payload):
            if name == fail_at:
                raise RuntimeError(f"{name} exploded")
            record.append(name)
            return f"{name} ok"

        return step

    return {name: make(name) for name in ("ingest", "extract", "verify", "score")}


def test_worker_runs_all_steps_and_finishes(conn):
    run_id, job_id = _job(conn)
    record: list[str] = []
    worker = Worker(SETTINGS, steps=_fake_steps(record))
    assert worker.run_pending(conn) == 1

    assert record == ["ingest", "extract", "verify", "score"]
    job = dbmod.get_job(conn, job_id)
    assert job["status"] == "DONE"
    assert [p["step"] for p in job["progress"]] == ["ingest", "extract", "verify", "score"]
    assert all(p["status"] == "done" for p in job["progress"])
    assert dbmod.get_run(conn, run_id)["status"] == "DONE"


def test_failed_step_marks_job_run_and_progress(conn):
    run_id, job_id = _job(conn)
    record: list[str] = []
    worker = Worker(SETTINGS, steps=_fake_steps(record, fail_at="verify"))
    worker.run_pending(conn)

    job = dbmod.get_job(conn, job_id)
    assert job["status"] == "FAILED"
    assert "verify exploded" in job["error"]
    by_step = {p["step"]: p["status"] for p in job["progress"]}
    assert by_step["extract"] == "done"
    assert by_step["verify"] == "failed"
    assert "score" not in by_step  # never reached
    run = dbmod.get_run(conn, run_id)
    assert run["status"] == "FAILED"
    assert "verify exploded" in run["error"]


def test_requeued_job_resumes_from_first_incomplete_step(conn):
    run_id, job_id = _job(conn)
    # Simulate a job interrupted after extract: two steps done, then restart.
    dbmod.claim_next_job(conn)
    dbmod.push_job_progress(conn, job_id, "ingest", "done earlier", status="done")
    dbmod.push_job_progress(conn, job_id, "extract", "done earlier", status="done")

    recovered = dbmod.requeue_running_jobs(conn)
    assert recovered == [job_id]
    assert dbmod.get_job(conn, job_id)["status"] == "QUEUED"

    record: list[str] = []
    Worker(SETTINGS, steps=_fake_steps(record)).run_pending(conn)
    assert record == ["verify", "score"]  # completed steps were NOT re-run
    assert dbmod.get_job(conn, job_id)["status"] == "DONE"


def test_requeue_touches_only_running_jobs(conn):
    _, queued_job = _job(conn)
    assert dbmod.requeue_running_jobs(conn) == []
    assert dbmod.get_job(conn, queued_job)["status"] == "QUEUED"


def test_claim_next_job_is_fifo_and_exhausts(conn):
    _, first = _job(conn)
    _, second = _job(conn)
    assert dbmod.claim_next_job(conn)["id"] == first
    assert dbmod.claim_next_job(conn)["id"] == second
    assert dbmod.claim_next_job(conn) is None


def test_torn_ingest_is_deleted_and_reingested(conn, tmp_path, monkeypatch):
    """A document row with zero chunks is a torn ingest — recovery must delete
    it (figures included, ON DISK too) and ingest fresh, not skip it."""
    from authorai import jobs as jobsmod

    settings = Settings(anthropic_api_key="x", openai_api_key="x", figures_dir=tmp_path / "figures")
    run_id = dbmod.create_run(conn)
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-fake")
    upload_id = dbmod.add_upload(conn, "SOURCE", "source.pdf", str(pdf))
    torn_doc = dbmod.add_document(conn, run_id, "SOURCE", upload_id=upload_id)
    # The torn attempt's PNG landed on disk BEFORE the crash; re-ingest gets a
    # fresh doc_id, so this file would be unreferenced garbage if left behind.
    png = tmp_path / "figures" / run_id / torn_doc / "fig-1.png"
    png.parent.mkdir(parents=True)
    png.write_bytes(b"fake png")
    dbmod.add_figure(conn, run_id, torn_doc, image_path=str(png), page=1)
    # No chunks for torn_doc — the torn state.

    ingested: list[tuple] = []

    def fake_ingest_pdf(
        conn_, embedder, run_id_, path, *, kind, figures_dir, upload_id, describe, fallback_title
    ):
        # The fake must not drift from the real contract: a renamed real
        # parameter would otherwise pass here and TypeError in production.
        inspect.signature(ingest_pdf).bind(
            conn_,
            embedder,
            run_id_,
            path,
            kind=kind,
            figures_dir=figures_dir,
            upload_id=upload_id,
            describe=describe,
            fallback_title=fallback_title,
        )
        ingested.append((str(path), kind, upload_id, fallback_title))
        return "new-doc-id"

    monkeypatch.setattr(jobsmod, "ingest_pdf", fake_ingest_pdf)
    context = PipelineContext(conn, settings)
    _reconcile_upload(context, run_id, upload_id)

    # fallback_title carries the original file name's STEM — the disk path is a
    # generated hex name, and this matches the CLI's path.stem convention.
    assert ingested == [(str(pdf), "SOURCE", upload_id, "source")]
    assert (
        conn.execute("SELECT count(*) FROM documents WHERE id = ?", (torn_doc,)).fetchone()[0] == 0
    )
    assert (
        conn.execute("SELECT count(*) FROM figures WHERE doc_id = ?", (torn_doc,)).fetchone()[0]
        == 0
    )
    assert not png.parent.exists()  # the orphaned figure directory is gone


def test_completed_ingest_is_skipped(conn, tmp_path, monkeypatch):
    from authorai import jobs as jobsmod
    from authorai.embeddings import FakeEmbedder

    run_id = dbmod.create_run(conn)
    pdf = tmp_path / "done.pdf"
    pdf.write_bytes(b"%PDF-fake")
    upload_id = dbmod.add_upload(conn, "SOURCE", "done.pdf", str(pdf))
    doc_id = dbmod.add_document(conn, run_id, "SOURCE", upload_id=upload_id)
    embedder = FakeEmbedder(dim=DIM)
    dbmod.add_chunks(conn, run_id, doc_id, [{"text": "content"}], embedder.embed(["content"]))

    monkeypatch.setattr(
        jobsmod, "ingest_pdf", lambda *a, **k: pytest.fail("must not re-ingest a completed doc")
    )
    _reconcile_upload(PipelineContext(conn, SETTINGS), run_id, upload_id)


def test_unknown_upload_in_payload_is_loud(conn):
    run_id = dbmod.create_run(conn)
    with pytest.raises(ValueError, match="unknown upload"):
        _reconcile_upload(PipelineContext(conn, SETTINGS), run_id, "no-such-upload")


def test_run_job_directly_without_worker(conn):
    """run_job is callable outside the worker loop (recovery paths, tests)."""
    run_id, job_id = _job(conn)
    job = dbmod.claim_next_job(conn)
    record: list[str] = []
    run_job(conn, SETTINGS, job, steps=_fake_steps(record))
    assert dbmod.get_job(conn, job_id)["status"] == "DONE"


def test_zero_claim_extraction_fails_the_extract_step(conn, monkeypatch):
    """An empty extraction recorded as a green step would push the failure to
    verify, whose 'run extract first' message would point at a step that
    looks successful."""
    from authorai import jobs as jobsmod
    from authorai.jobs import step_extract

    run_id = dbmod.create_run(conn)
    dbmod.add_document(conn, run_id, "REPORT", metadata='{"sections": [{"text": "x"}]}')
    monkeypatch.setattr(jobsmod, "extract_claims", lambda *a, **k: [])
    with pytest.raises(ValueError, match="0 claims"):
        step_extract(PipelineContext(conn, SETTINGS), run_id, {})


def test_requeued_failed_job_resumes_without_repeating_done_steps(conn):
    """The retry path end to end: fail at verify, requeue, finish — ingest and
    extract must NOT run again (a transient failure must not cost the whole
    ingest twice), and both rows must end DONE."""
    run_id, job_id = _job(conn)
    record: list[str] = []
    job = dbmod.claim_next_job(conn)
    run_job(conn, SETTINGS, job, steps=_fake_steps(record, fail_at="verify"))
    assert dbmod.get_job(conn, job_id)["status"] == "FAILED"
    assert dbmod.get_run(conn, run_id)["status"] == "FAILED"
    assert record == ["ingest", "extract"]

    dbmod.requeue_job(conn, job_id, run_id)
    requeued = dbmod.get_job(conn, job_id)
    assert requeued["status"] == "QUEUED"
    assert requeued["error"] is None
    assert dbmod.get_run(conn, run_id)["status"] == "RUNNING"

    job = dbmod.claim_next_job(conn)
    run_job(conn, SETTINGS, job, steps=_fake_steps(record))
    assert dbmod.get_job(conn, job_id)["status"] == "DONE"
    assert dbmod.get_run(conn, run_id)["status"] == "DONE"
    assert record == ["ingest", "extract", "verify", "score"]  # no repeats


def test_update_job_payload_merges_and_deletes(conn):
    run_id, job_id = _job(conn, payload={"report_upload_id": "r", "source_upload_ids": []})
    dbmod.update_job_payload(conn, job_id, {"verify_batch_id": "msgbatch_1"})
    assert dbmod.get_job(conn, job_id)["payload"]["verify_batch_id"] == "msgbatch_1"
    # Existing keys survive a merge; None deletes.
    assert dbmod.get_job(conn, job_id)["payload"]["report_upload_id"] == "r"
    dbmod.update_job_payload(conn, job_id, {"verify_batch_id": None})
    assert "verify_batch_id" not in dbmod.get_job(conn, job_id)["payload"]


def test_step_verify_resumes_stored_batch_and_persists_new_ids(conn, monkeypatch):
    """The don't-pay-twice contract: a stored batch id is passed to verify_run
    as resume_batch_id, and a newly created batch's id is persisted on the job
    the moment the callback fires."""
    from authorai import jobs as jobsmod
    from authorai.jobs import step_verify

    run_id, job_id = _job(conn)
    captured: dict = {}

    def fake_verify_run(conn_, embedder, llm, run_id_, **kwargs):
        captured["resume_batch_id"] = kwargs["resume_batch_id"]
        kwargs["on_batch_created"]("msgbatch_new")
        return {
            "counts": {"SUPPORTED": 1, "CONTRADICTED": 0, "UNVERIFIABLE": 0},
            "downgraded": 0,
            "year_flagged": 0,
            "no_evidence": 0,
            "total": 1,
        }

    monkeypatch.setattr(jobsmod, "verify_run", fake_verify_run)
    dbmod.update_job_payload(conn, job_id, {"verify_batch_id": "msgbatch_old"})
    payload = dbmod.get_job(conn, job_id)["payload"]
    step_verify(PipelineContext(conn, SETTINGS), run_id, payload)

    assert captured["resume_batch_id"] == "msgbatch_old"
    assert dbmod.get_job(conn, job_id)["payload"]["verify_batch_id"] == "msgbatch_new"


def test_step_verify_clears_batch_id_on_stale_batch(conn, monkeypatch):
    """A poisoned verify_batch_id must not wedge the retry loop: when the
    stored batch turns out to belong to a different claim set, the key is
    cleared so the NEXT retry submits fresh instead of failing identically
    forever."""
    from authorai import jobs as jobsmod
    from authorai.jobs import step_verify
    from authorai.llm import StaleBatchError

    run_id, job_id = _job(conn)
    dbmod.update_job_payload(conn, job_id, {"verify_batch_id": "msgbatch_poisoned"})

    def fake_verify_run(*args, **kwargs):
        raise StaleBatchError("batch does not match the current claims")

    monkeypatch.setattr(jobsmod, "verify_run", fake_verify_run)
    import pytest

    with pytest.raises(StaleBatchError):
        step_verify(PipelineContext(conn, SETTINGS), run_id, dbmod.get_job(conn, job_id)["payload"])
    assert "verify_batch_id" not in dbmod.get_job(conn, job_id)["payload"]


def test_requeue_refuses_non_failed_jobs(conn):
    run_id, job_id = _job(conn)  # QUEUED
    import pytest

    with pytest.raises(ValueError, match="not FAILED"):
        dbmod.requeue_job(conn, job_id, run_id)


def test_failure_handler_failing_does_not_propagate(conn):
    """If even recording the failure fails, run_job must swallow it (the
    worker thread survives) and leave the job RUNNING for startup recovery."""
    run_id, job_id = _job(conn)
    job = dbmod.claim_next_job(conn)
    job["run_id"] = "no-such-run"  # set_run_status raises, then so does the handler
    run_job(conn, SETTINGS, job, steps=_fake_steps([]))  # must not raise
    assert dbmod.get_job(conn, job_id)["status"] == "RUNNING"


def test_real_step_registry_matches_the_pipeline_and_signature():
    """A typo'd key or drifted step signature in REAL_STEPS would only ever
    surface on the first production job — every other test injects fakes."""
    assert set(REAL_STEPS) == set(PIPELINE_STEPS)
    for step in REAL_STEPS.values():
        inspect.signature(step).bind("context", "run_id", "payload")


def test_done_write_failure_leaves_job_running_not_failed(conn, monkeypatch):
    """The documented invariant: a bookkeeping failure AFTER all steps
    succeeded must never rewrite the run as FAILED — the job stays RUNNING
    (its steps recorded done) for startup recovery to re-finish."""
    from authorai import jobs as jobsmod

    run_id, job_id = _job(conn)
    job = dbmod.claim_next_job(conn)
    real_finish = dbmod.finish_job_and_run

    def flaky_finish(conn_, job_id_, run_id_, status, error=None):
        raise RuntimeError("lock timeout on the final write")

    monkeypatch.setattr(jobsmod.dbmod, "finish_job_and_run", flaky_finish)
    with pytest.raises(RuntimeError, match="lock timeout"):
        run_job(conn, SETTINGS, job, steps=_fake_steps([]))

    stranded = dbmod.get_job(conn, job_id)
    assert stranded["status"] == "RUNNING"  # never FAILED — all steps succeeded
    assert all(p["status"] == "done" for p in stranded["progress"])

    # Startup recovery then completes it without re-running any step.
    monkeypatch.setattr(jobsmod.dbmod, "finish_job_and_run", real_finish)
    assert dbmod.requeue_running_jobs(conn) == [job_id]
    record: list[str] = []
    Worker(SETTINGS, steps=_fake_steps(record)).run_pending(conn)
    assert record == []  # nothing re-ran; only the finish was repeated
    assert dbmod.get_job(conn, job_id)["status"] == "DONE"
    assert dbmod.get_run(conn, run_id)["status"] == "DONE"


def _poll(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_worker_thread_completes_jobs_and_survives_step_failures(tmp_path):
    """The REAL thread loop, not the synchronous drain: it must claim queued
    jobs, and a failing job must not kill the only worker thread."""
    settings = Settings(
        anthropic_api_key="x",
        openai_api_key="x",
        db_path=tmp_path / "w.db",
        embedding_dim=DIM,
        job_poll_seconds=0.01,
    )
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    _, failing = _job(conn)
    record: list[str] = []
    worker = Worker(settings, steps=_fake_steps(record, fail_at="verify"))
    worker.start()
    try:
        assert _poll(lambda: dbmod.get_job(conn, failing)["status"] == "FAILED")
        assert worker._thread.is_alive()  # the failure did not kill the loop
        # Hand it a clean job AFTER the failure — the same thread must run it.
        worker._steps = _fake_steps(record)
        _, ok_job = _job(conn)
        assert _poll(lambda: dbmod.get_job(conn, ok_job)["status"] == "DONE")
    finally:
        worker.stop()
    conn.close()


def test_worker_start_twice_is_loud_and_restart_after_stop_works(conn, tmp_path):
    settings = Settings(
        anthropic_api_key="x",
        openai_api_key="x",
        db_path=tmp_path / "w.db",
        embedding_dim=8,
        job_poll_seconds=0.01,
    )
    worker = Worker(settings, steps=_fake_steps([]))
    worker.start()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            worker.start()
    finally:
        worker.stop()
    # stop() sets the event; start() must clear it or the new thread is a no-op.
    worker.start()
    assert not worker._stop.is_set()
    worker.stop()
