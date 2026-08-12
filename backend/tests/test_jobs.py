"""Job worker tests: claiming, resume, failure, and torn-ingest recovery."""

import pytest

from authorai import db as dbmod
from authorai.config import Settings
from authorai.jobs import PipelineContext, Worker, _reconcile_upload, run_job
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
    it (figures included) and ingest fresh, not skip it."""
    from authorai import jobs as jobsmod

    run_id = dbmod.create_run(conn)
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-fake")
    upload_id = dbmod.add_upload(conn, "SOURCE", "source.pdf", str(pdf))
    torn_doc = dbmod.add_document(conn, run_id, "SOURCE", upload_id=upload_id)
    dbmod.add_figure(conn, run_id, torn_doc, image_path="/tmp/x.png", page=1)
    # No chunks for torn_doc — the torn state.

    ingested: list[tuple] = []

    def fake_ingest_pdf(
        conn_, embedder, run_id_, path, *, kind, figures_dir, upload_id, describe, fallback_title
    ):
        ingested.append((str(path), kind, upload_id, fallback_title))
        return "new-doc-id"

    monkeypatch.setattr(jobsmod, "ingest_pdf", fake_ingest_pdf)
    context = PipelineContext(conn, SETTINGS)
    _reconcile_upload(context, run_id, upload_id)

    # fallback_title carries the ORIGINAL file name — the disk path is a
    # generated hex name, so untitled documents would otherwise show as ids.
    assert ingested == [(str(pdf), "SOURCE", upload_id, "source.pdf")]
    assert (
        conn.execute("SELECT count(*) FROM documents WHERE id = ?", (torn_doc,)).fetchone()[0] == 0
    )
    assert (
        conn.execute("SELECT count(*) FROM figures WHERE doc_id = ?", (torn_doc,)).fetchone()[0]
        == 0
    )


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
