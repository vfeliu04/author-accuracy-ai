"""Job worker tests: claiming, resume, failure, torn-ingest recovery, dedup."""

import codecs
import inspect
import time
from pathlib import Path

import pytest

from authorai import db as dbmod
from authorai.config import Settings
from authorai.embeddings import FakeEmbedder
from authorai.ingest import ingest_pdf
from authorai.jobs import (
    PIPELINE_STEPS,
    REAL_STEPS,
    PipelineContext,
    Worker,
    _reconcile_upload,
    run_job,
    step_ingest,
)
from authorai.web import ExtractionTimeoutError, ThinPageError, extract_web
from tests.conftest import DIM, poison_providers

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


# --------------------------------------------------------------------------
# Cross-run ingest dedup: _reconcile_upload reuses a prior ingest of
# byte-identical bytes instead of recomputing it.


def _dedup_settings(tmp_path):
    return Settings(
        anthropic_api_key="x",
        openai_api_key="x",
        figures_dir=tmp_path / "figures",
        uploads_dir=tmp_path,  # the donor PDFs live here, so deleting a run removes them
    )


def _ingested_upload(
    conn, tmp_path, *, kind="SOURCE", content_hash="feed01", run_id=None, embedding_model=None
):
    """A COMPLETE prior ingest at the jobs level: rows plus a PNG on disk.

    Stamped with the Settings-default embedding model unless overridden, so
    the donor passes the per-document model filter the way a real prior
    ingest under the same configuration would."""
    run_id = run_id or dbmod.create_run(conn)
    pdf = tmp_path / f"{dbmod.new_id()}.pdf"
    pdf.write_bytes(b"%PDF-donor")
    upload_id = dbmod.add_upload(conn, kind, "donor.pdf", str(pdf), content_hash)
    doc_id = dbmod.add_document(
        conn,
        run_id,
        kind,
        upload_id=upload_id,
        title="Donor Doc",
        embedding_model=embedding_model or SETTINGS.embedding_model,
    )
    png = tmp_path / "figures" / run_id / doc_id / "fig-1.png"
    png.parent.mkdir(parents=True)
    png.write_bytes(b"donor png bytes")
    figure_id = dbmod.add_figure(conn, run_id, doc_id, str(png), page=2, caption="cap")
    chunks = [
        {"text": "alpha wheat statistics", "page": 1, "section": "Intro"},
        {"text": "beta table of yields", "page": 2, "kind": "table"},
        {"text": "figure about crops", "page": 2, "kind": "figure", "figure_id": figure_id},
    ]
    embedder = FakeEmbedder(dim=DIM)
    dbmod.add_chunks(conn, run_id, doc_id, chunks, embedder.embed([c["text"] for c in chunks]))
    return run_id, upload_id, doc_id


def test_dedup_reuses_prior_ingest_without_recomputing(conn, tmp_path, monkeypatch):
    """The headline contract: identical bytes reuse the donor's rows and PNGs
    with NO provider client constructed."""
    _, _, donor_doc = _ingested_upload(conn, tmp_path, content_hash="beef01")
    poison_providers(monkeypatch)

    run_id = dbmod.create_run(conn)
    pdf = tmp_path / "again.pdf"
    pdf.write_bytes(b"%PDF-donor")
    upload_id = dbmod.add_upload(conn, "SOURCE", "again.pdf", str(pdf), "beef01")
    context = PipelineContext(conn, _dedup_settings(tmp_path))
    assert _reconcile_upload(context, run_id, upload_id) is True

    new_doc = conn.execute(
        "SELECT * FROM documents WHERE run_id = ? AND upload_id = ?", (run_id, upload_id)
    ).fetchone()
    assert new_doc["title"] == "Donor Doc"
    texts = "SELECT text FROM chunks WHERE doc_id = ? ORDER BY id"
    assert [r["text"] for r in conn.execute(texts, (new_doc["id"],))] == [
        r["text"] for r in conn.execute(texts, (donor_doc,))
    ]
    new_figure = conn.execute("SELECT * FROM figures WHERE doc_id = ?", (new_doc["id"],)).fetchone()
    assert Path(new_figure["image_path"]).read_bytes() == b"donor png bytes"
    assert Path(new_figure["image_path"]).parent == tmp_path / "figures" / run_id / new_doc["id"]
    # Idempotent on retry: the copy is now a completed ingest, nothing reruns.
    assert _reconcile_upload(context, run_id, upload_id) is False


def test_dedup_rewrites_kind_for_the_new_upload(conn, tmp_path, monkeypatch):
    """A SOURCE donor serving a REPORT upload: the copy lands under doc_kind
    REPORT — the partition verification retrieval filters on."""
    from authorai.search import vector_search

    _ingested_upload(conn, tmp_path, kind="SOURCE", content_hash="beef02")
    poison_providers(monkeypatch)

    run_id = dbmod.create_run(conn)
    pdf = tmp_path / "as-report.pdf"
    pdf.write_bytes(b"%PDF-donor")
    upload_id = dbmod.add_upload(conn, "REPORT", "as-report.pdf", str(pdf), "beef02")
    assert _reconcile_upload(PipelineContext(conn, _dedup_settings(tmp_path)), run_id, upload_id)

    doc = conn.execute("SELECT * FROM documents WHERE run_id = ?", (run_id,)).fetchone()
    assert doc["kind"] == "REPORT"
    query = [1.0] + [0.0] * (DIM - 1)
    assert len(vector_search(conn, run_id, query, k=5, doc_kind="REPORT")) == 3
    assert vector_search(conn, run_id, query, k=5, doc_kind="SOURCE") == []


def test_dedup_refused_when_donor_used_another_embedding_model(conn, tmp_path, monkeypatch):
    """Stored vectors from another embedding model must not answer for this
    one. The filter is PER DOCUMENT, so an A→B→A model flip-flop cannot
    launder B-vectors under A, and an unstamped pre-dedup document never
    donates — both fall through to a fresh ingest."""
    from authorai import jobs as jobsmod

    _ingested_upload(conn, tmp_path, content_hash="beef03", embedding_model="older-model")
    legacy_run = dbmod.create_run(conn)
    legacy_pdf = tmp_path / "legacy-donor.pdf"
    legacy_pdf.write_bytes(b"%PDF-donor")
    legacy_upload = dbmod.add_upload(conn, "SOURCE", "legacy.pdf", str(legacy_pdf), "beef03")
    legacy_doc = dbmod.add_document(conn, legacy_run, "SOURCE", upload_id=legacy_upload)
    dbmod.add_chunks(
        conn, legacy_run, legacy_doc, [{"text": "old"}], FakeEmbedder(dim=DIM).embed(["old"])
    )  # complete but NULL-stamped: pre-dedup

    ingested: list[int] = []
    monkeypatch.setattr(jobsmod, "ingest_pdf", lambda *a, **k: ingested.append(1))
    run_id = dbmod.create_run(conn)
    pdf = tmp_path / "again.pdf"
    pdf.write_bytes(b"%PDF-donor")
    upload_id = dbmod.add_upload(conn, "SOURCE", "again.pdf", str(pdf), "beef03")
    context = PipelineContext(conn, _dedup_settings(tmp_path))
    assert _reconcile_upload(context, run_id, upload_id) is False
    assert ingested == [1]


def test_failed_copy_falls_back_to_fresh_ingest(conn, tmp_path, monkeypatch):
    """The wedge class: a complete donor whose PNG vanished from disk must
    not fail the run — the copy attempt cleans up after itself and the
    reconcile ingests fresh (dedup is an optimization, never a blocker)."""
    from authorai import jobs as jobsmod

    donor_run, _, donor_doc = _ingested_upload(conn, tmp_path, content_hash="beef06")
    donor_png = conn.execute(
        "SELECT image_path FROM figures WHERE doc_id = ?", (donor_doc,)
    ).fetchone()["image_path"]
    Path(donor_png).unlink()  # rows intact, file lost

    ingested: list[int] = []
    monkeypatch.setattr(jobsmod, "ingest_pdf", lambda *a, **k: ingested.append(1))
    settings = _dedup_settings(tmp_path)
    run_id = dbmod.create_run(conn)
    pdf = tmp_path / "again.pdf"
    pdf.write_bytes(b"%PDF-donor")
    upload_id = dbmod.add_upload(conn, "SOURCE", "again.pdf", str(pdf), "beef06")
    assert _reconcile_upload(PipelineContext(conn, settings), run_id, upload_id) is False
    assert ingested == [1]
    # The aborted copy left nothing: no rows for the new run, no figure files
    # (the empty run-level directory may remain — same as torn-ingest cleanup).
    assert (
        conn.execute("SELECT count(*) FROM documents WHERE run_id = ?", (run_id,)).fetchone()[0]
        == 0
    )
    assert not any(settings.run_figures_dir(run_id).rglob("*.png"))


def test_copied_figure_paths_are_absolute_with_relative_figures_dir(conn, tmp_path, monkeypatch):
    """The default figures_dir is RELATIVE ('data/figures'): fresh ingest
    resolves it before storing image_path, so the copy must too — a
    CWD-relative stored path breaks verification (and deletion) as soon as
    the server starts from a different directory."""
    monkeypatch.chdir(tmp_path)
    settings = Settings(anthropic_api_key="x", openai_api_key="x", figures_dir=Path("figures-rel"))
    donor_run = dbmod.create_run(conn)
    donor_pdf = tmp_path / "donor.pdf"
    donor_pdf.write_bytes(b"%PDF-donor")
    donor_upload = dbmod.add_upload(conn, "SOURCE", "donor.pdf", str(donor_pdf), "beef07")
    donor_doc = dbmod.add_document(
        conn,
        donor_run,
        "SOURCE",
        upload_id=donor_upload,
        title="Donor Doc",
        embedding_model=settings.embedding_model,
    )
    donor_png = tmp_path / "donor-figs" / "fig-1.png"
    donor_png.parent.mkdir()
    donor_png.write_bytes(b"donor png bytes")
    figure_id = dbmod.add_figure(conn, donor_run, donor_doc, str(donor_png), page=1)
    dbmod.add_chunks(
        conn,
        donor_run,
        donor_doc,
        [{"text": "figure", "kind": "figure", "figure_id": figure_id}],
        FakeEmbedder(dim=DIM).embed(["figure"]),
    )
    poison_providers(monkeypatch)

    run_id = dbmod.create_run(conn)
    pdf = tmp_path / "again.pdf"
    pdf.write_bytes(b"%PDF-donor")
    upload_id = dbmod.add_upload(conn, "SOURCE", "again.pdf", str(pdf), "beef07")
    assert _reconcile_upload(PipelineContext(conn, settings), run_id, upload_id) is True
    [copied] = conn.execute("SELECT image_path FROM figures WHERE run_id = ?", (run_id,)).fetchall()
    stored = Path(copied["image_path"])
    assert stored.is_absolute()
    assert stored.read_bytes() == b"donor png bytes"
    assert stored.is_relative_to(tmp_path / "figures-rel")


def test_dedup_ignores_torn_donors_and_null_hashes(conn, tmp_path, monkeypatch):
    """A torn donor (no chunks) and a hashless legacy upload both fall
    through to a fresh ingest — never a degraded copy."""
    from authorai import jobs as jobsmod

    torn_run = dbmod.create_run(conn)
    torn_pdf = tmp_path / "torn.pdf"
    torn_pdf.write_bytes(b"%PDF-torn")
    torn_upload = dbmod.add_upload(conn, "SOURCE", "torn.pdf", str(torn_pdf), "dead01")
    # Model-stamped so TORNNESS (zero chunks) is what excludes it, not the
    # per-document model filter.
    dbmod.add_document(
        conn,
        torn_run,
        "SOURCE",
        upload_id=torn_upload,
        embedding_model=SETTINGS.embedding_model,
    )

    ingested: list[int] = []
    monkeypatch.setattr(jobsmod, "ingest_pdf", lambda *a, **k: ingested.append(1))
    settings = _dedup_settings(tmp_path)

    run_id = dbmod.create_run(conn)
    same_hash = tmp_path / "same-hash.pdf"
    same_hash.write_bytes(b"%PDF-torn")
    matching = dbmod.add_upload(conn, "SOURCE", "same-hash.pdf", str(same_hash), "dead01")
    assert _reconcile_upload(PipelineContext(conn, settings), run_id, matching) is False

    hashless = dbmod.add_upload(conn, "SOURCE", "legacy.pdf", str(same_hash), None)
    assert _reconcile_upload(PipelineContext(conn, settings), run_id, hashless) is False
    assert ingested == [1, 1]


def test_step_ingest_label_counts_reuse(conn, tmp_path, monkeypatch):
    """The done-label the UI shows verbatim: '(M already read)' only when M > 0."""
    from authorai import jobs as jobsmod

    monkeypatch.setattr(jobsmod, "ingest_pdf", lambda *a, **k: None)
    settings = _dedup_settings(tmp_path)
    _ingested_upload(conn, tmp_path, content_hash="feed99")

    run_id = dbmod.create_run(conn)
    report_pdf = tmp_path / "r.pdf"
    report_pdf.write_bytes(b"%PDF-r")
    source_pdf = tmp_path / "s.pdf"
    source_pdf.write_bytes(b"%PDF-donor")
    report_upload = dbmod.add_upload(conn, "REPORT", "r.pdf", str(report_pdf), "aaaa01")
    source_upload = dbmod.add_upload(conn, "SOURCE", "s.pdf", str(source_pdf), "feed99")
    payload = {"report_upload_id": report_upload, "source_upload_ids": [source_upload]}
    context = PipelineContext(conn, settings)
    assert step_ingest(context, run_id, payload) == "Read 2 documents (1 already read)"

    other_run = dbmod.create_run(conn)
    fresh_a = dbmod.add_upload(conn, "REPORT", "a.pdf", str(report_pdf), "bbbb01")
    fresh_b = dbmod.add_upload(conn, "SOURCE", "b.pdf", str(source_pdf), "bbbb02")
    payload = {"report_upload_id": fresh_a, "source_upload_ids": [fresh_b]}
    assert step_ingest(context, other_run, payload) == "Read 2 documents"


def _api_style_delete(conn, settings, run_id):
    """Exactly what DELETE /runs/{id} does — the route itself, called directly:
    rows in one txn, then the files it owns."""
    from types import SimpleNamespace

    from authorai.api import delete_run

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))
    delete_run(run_id, request, conn)


def test_deleting_the_donor_run_leaves_the_copy_whole(conn, tmp_path, monkeypatch):
    """Copy-never-share, proven from the copy's side: after the donor run is
    FULLY deleted (rows, its PDF, its PNGs), the copy keeps its rows, its
    byte-identical vectors, its own PNG, its own PDF — and both indexes
    still answer."""
    from authorai.search import keyword_search

    donor_run, _, donor_doc = _ingested_upload(conn, tmp_path, content_hash="c0de01")
    poison_providers(monkeypatch)
    settings = _dedup_settings(tmp_path)
    run_id = dbmod.create_run(conn)
    pdf = tmp_path / "mine.pdf"
    pdf.write_bytes(b"%PDF-donor")
    upload_id = dbmod.add_upload(conn, "SOURCE", "mine.pdf", str(pdf), "c0de01")
    assert _reconcile_upload(PipelineContext(conn, settings), run_id, upload_id) is True
    copy_doc = conn.execute("SELECT id FROM documents WHERE run_id = ?", (run_id,)).fetchone()["id"]
    blob_sql = (
        "SELECT v.embedding FROM chunks_vec v JOIN chunks c ON c.id = v.chunk_id"
        " WHERE c.doc_id = ? ORDER BY c.id"
    )
    donor_blobs = [r["embedding"] for r in conn.execute(blob_sql, (donor_doc,))]

    _api_style_delete(conn, settings, donor_run)

    assert conn.execute("SELECT count(*) FROM runs WHERE id = ?", (donor_run,)).fetchone()[0] == 0
    assert [r["embedding"] for r in conn.execute(blob_sql, (copy_doc,))] == donor_blobs
    figure = conn.execute("SELECT * FROM figures WHERE doc_id = ?", (copy_doc,)).fetchone()
    assert Path(figure["image_path"]).read_bytes() == b"donor png bytes"
    assert pdf.exists()  # the copy's upload PDF was its own, not the donor's
    assert keyword_search(conn, run_id, "wheat") != []


def test_deleting_the_copy_leaves_the_donor_a_valid_donor(conn, tmp_path, monkeypatch):
    donor_run, _, donor_doc = _ingested_upload(conn, tmp_path, content_hash="c0de02")
    poison_providers(monkeypatch)
    settings = _dedup_settings(tmp_path)
    first = dbmod.create_run(conn)
    first_pdf = tmp_path / "first.pdf"
    first_pdf.write_bytes(b"%PDF-donor")
    first_upload = dbmod.add_upload(conn, "SOURCE", "first.pdf", str(first_pdf), "c0de02")
    assert _reconcile_upload(PipelineContext(conn, settings), first, first_upload) is True

    _api_style_delete(conn, settings, first)

    donor_figure = conn.execute("SELECT * FROM figures WHERE doc_id = ?", (donor_doc,)).fetchone()
    assert Path(donor_figure["image_path"]).read_bytes() == b"donor png bytes"
    assert conn.execute("SELECT count(*) FROM runs WHERE id = ?", (donor_run,)).fetchone()[0] == 1
    # ... and it still serves the NEXT identical upload.
    second = dbmod.create_run(conn)
    second_pdf = tmp_path / "second.pdf"
    second_pdf.write_bytes(b"%PDF-donor")
    second_upload = dbmod.add_upload(conn, "SOURCE", "second.pdf", str(second_pdf), "c0de02")
    assert _reconcile_upload(PipelineContext(conn, settings), second, second_upload) is True


def test_same_file_twice_in_one_run_reuses_within_the_run(conn, tmp_path, monkeypatch):
    """Report and source are the same bytes: the source reconcile reuses the
    report's just-finished ingest rewritten to SOURCE — the partition
    verification retrieval reads from."""
    from authorai.search import vector_search

    settings = _dedup_settings(tmp_path)
    run_id = dbmod.create_run(conn)
    _ingested_upload(conn, tmp_path, kind="REPORT", content_hash="feed77", run_id=run_id)
    poison_providers(monkeypatch)
    source_pdf = tmp_path / "same-bytes.pdf"
    source_pdf.write_bytes(b"%PDF-donor")
    source_upload = dbmod.add_upload(conn, "SOURCE", "same-bytes.pdf", str(source_pdf), "feed77")
    assert _reconcile_upload(PipelineContext(conn, settings), run_id, source_upload) is True

    docs = {
        row["kind"]: row["id"]
        for row in conn.execute("SELECT id, kind FROM documents WHERE run_id = ?", (run_id,))
    }
    assert set(docs) == {"REPORT", "SOURCE"}
    source_chunk_ids = [
        row["id"]
        for row in conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (docs["SOURCE"],))
    ]
    query = [1.0] + [0.0] * (DIM - 1)
    assert sorted(vector_search(conn, run_id, query, k=10, doc_kind="SOURCE")) == sorted(
        source_chunk_ids
    )


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


# --- link sources: fetch pre-pass, snapshot, per-type dispatch -------------


def _completed_report(conn, tmp_path, run_id):
    """A REPORT upload whose ingest already finished, so reconcile skips it."""
    pdf = tmp_path / f"{dbmod.new_id()}.pdf"
    pdf.write_bytes(b"%PDF-report")
    upload_id = dbmod.add_upload(conn, "REPORT", "report.pdf", str(pdf), dbmod.new_id())
    doc_id = dbmod.add_document(conn, run_id, "REPORT", upload_id=upload_id)
    dbmod.add_chunks(
        conn,
        run_id,
        doc_id,
        [{"text": "report text"}],
        FakeEmbedder(dim=DIM).embed(["report text"]),
    )
    return upload_id


def _link_upload(conn, tmp_path, url, *, source_type="web"):
    planned = tmp_path / "uploads" / f"{dbmod.new_id()}.json"
    planned.parent.mkdir(parents=True, exist_ok=True)
    upload_id = dbmod.add_upload(
        conn, "SOURCE", url, str(planned), source_type=source_type, url=url
    )
    return upload_id, planned


def _fetched(
    url,
    *,
    body=b"<html><body>page</body></html>",
    content_type="text/html",
    final_url=None,
    charset="utf-8",
):
    from authorai.fetch import FetchedResponse

    return FetchedResponse(
        url=url,
        final_url=final_url or url,
        content_type=content_type,
        charset=charset,
        body=body,
        is_pdf=body.startswith(b"%PDF-"),
    )


def test_unreadable_link_fails_ingest_before_any_document_is_processed(conn, tmp_path, monkeypatch):
    """A bad link must fail the run in seconds: before the report's Docling
    parse, figure captions, or embeddings run — none of which a retry needs."""
    from authorai import jobs as jobsmod
    from authorai.fetch import FetchError

    run_id = dbmod.create_run(conn)
    report_pdf = tmp_path / "r.pdf"
    report_pdf.write_bytes(b"%PDF-r")
    report = dbmod.add_upload(
        conn, "REPORT", "r.pdf", str(report_pdf), "aaaa09"
    )  # NOT yet ingested
    source, _ = _link_upload(conn, tmp_path, "https://intranet.example/secret")
    poison_providers(monkeypatch)

    def refuse(url, settings, **kwargs):
        raise FetchError(f"{url} resolves to a private or reserved network address")

    monkeypatch.setattr(jobsmod, "fetch_url", refuse)
    payload = {"report_upload_id": report, "source_upload_ids": [source]}
    with pytest.raises(FetchError, match="https://intranet.example/secret"):
        step_ingest(PipelineContext(conn, SETTINGS), run_id, payload)
    assert (
        conn.execute("SELECT count(*) FROM documents WHERE run_id = ?", (run_id,)).fetchone()[0]
        == 0
    )


def test_web_link_is_fetched_into_a_snapshot_then_ingested_from_it(conn, tmp_path, monkeypatch):
    from authorai import jobs as jobsmod
    from authorai.ingest import ParsedDocument, ParsedSection, ingest_snapshot, load_snapshot
    from authorai.web import PageMetadata

    run_id = dbmod.create_run(conn)
    report = _completed_report(conn, tmp_path, run_id)
    url = "https://www.who.int/facts"
    upload_id, planned = _link_upload(conn, tmp_path, url)
    fetches: list[str] = []

    def fake_fetch(u, settings, **kwargs):
        fetches.append(u)
        return _fetched(u, final_url=u + "/")

    def fake_extract(html, *, url, timeout):
        document = ParsedDocument(
            title="Drinking-water",
            sections=[
                ParsedSection(title="Key facts", page=None, text="2.2 billion lack safe water.")
            ],
            tables=[],
            figures=[],
        )
        page = PageMetadata(
            title="Drinking-water",
            publisher="World Health Organization",
            publication_date="2026-03-01",
        )
        return document, page

    ingested: list[tuple] = []

    def fake_ingest_snapshot(
        conn_, embedder, run_id_, path, *, kind, figures_dir, upload_id, fallback_title
    ):
        inspect.signature(ingest_snapshot).bind(
            conn_,
            embedder,
            run_id_,
            path,
            kind=kind,
            figures_dir=figures_dir,
            upload_id=upload_id,
            fallback_title=fallback_title,
        )
        ingested.append((str(path), kind, upload_id, fallback_title))
        return "doc"

    monkeypatch.setattr(jobsmod, "fetch_url", fake_fetch)
    monkeypatch.setattr(jobsmod, "extract_web_bounded", fake_extract)
    monkeypatch.setattr(jobsmod, "ingest_snapshot", fake_ingest_snapshot)
    payload = {"report_upload_id": report, "source_upload_ids": [upload_id]}

    assert step_ingest(PipelineContext(conn, SETTINGS), run_id, payload) == (
        "Read 2 documents (1 web page opened)"
    )
    assert fetches == [url]
    parsed, provenance = load_snapshot(planned)
    assert parsed.sections[0].text == "2.2 billion lack safe water."
    assert (provenance["url"], provenance["final_url"]) == (url, url + "/")
    assert provenance["publisher"] == "World Health Organization"
    assert provenance["content_type"] == "text/html"
    assert provenance["fetched_at"]
    row = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    # A page snapshot is never hashed: link sources do not dedup this round.
    assert (row["source_type"], row["content_hash"], row["path"]) == ("web", None, str(planned))
    assert ingested == [(str(planned), "SOURCE", upload_id, url)]


def test_retry_with_an_existing_snapshot_never_fetches_again(conn, tmp_path, monkeypatch):
    from authorai import jobs as jobsmod
    from authorai.ingest import ParsedDocument, ParsedSection, write_snapshot

    run_id = dbmod.create_run(conn)
    report = _completed_report(conn, tmp_path, run_id)
    upload_id, planned = _link_upload(conn, tmp_path, "https://www.who.int/facts")
    write_snapshot(
        planned,
        ParsedDocument(
            title="T",
            sections=[ParsedSection(title="", page=None, text="kept")],
            tables=[],
            figures=[],
        ),
        {"url": "https://www.who.int/facts"},
    )
    monkeypatch.setattr(
        jobsmod, "fetch_url", lambda *a, **k: pytest.fail("re-fetched a stored page")
    )
    monkeypatch.setattr(jobsmod, "ingest_snapshot", lambda *a, **k: "doc")
    payload = {"report_upload_id": report, "source_upload_ids": [upload_id]}
    assert step_ingest(PipelineContext(conn, SETTINGS), run_id, payload) == "Read 2 documents"


@pytest.mark.parametrize("torn", [False, True], ids=["no-document", "torn-document"])
@pytest.mark.parametrize(
    "content",
    [
        b"",
        b'{"document": {"sections": [',
        b'{"schema": 0, "document": {"title": null, "sections": []}, "provenance": {}}',
    ],
    ids=["zero-byte", "truncated", "other-schema"],
)
def test_retry_refetches_a_stored_page_that_will_not_load(
    conn, tmp_path, monkeypatch, content, torn
):
    """A stored page that will not load (lost to a power cut mid-write, or from an
    older snapshot schema) wedged every retry: the pre-pass skipped it because the
    file existed, and ingest failed on it again. With no finished document cut
    from it, it is fetched again."""
    from authorai import jobs as jobsmod
    from authorai.ingest import ParsedDocument, ParsedSection, load_snapshot
    from authorai.web import PageMetadata

    run_id = dbmod.create_run(conn)
    report = _completed_report(conn, tmp_path, run_id)
    url = "https://www.who.int/facts"
    upload_id, planned = _link_upload(conn, tmp_path, url)
    planned.write_bytes(content)
    if torn:
        dbmod.add_document(conn, run_id, "SOURCE", upload_id=upload_id)  # no chunks
    fetches: list[str] = []
    loaded: list[str] = []

    def fake_fetch(u, settings, **kwargs):
        fetches.append(u)
        return _fetched(u)

    def fresh_page(html, *, url, timeout):
        section = ParsedSection(title="", page=None, text="fresh page text")
        return ParsedDocument(title=None, sections=[section], tables=[], figures=[]), PageMetadata()

    monkeypatch.setattr(jobsmod, "fetch_url", fake_fetch)
    monkeypatch.setattr(jobsmod, "extract_web_bounded", fresh_page)
    monkeypatch.setattr(
        jobsmod,
        "ingest_snapshot",
        lambda c, e, r, path, **k: loaded.append(load_snapshot(path)[0].sections[0].text),
    )
    payload = {"report_upload_id": report, "source_upload_ids": [upload_id]}

    label = step_ingest(PipelineContext(conn, SETTINGS), run_id, payload)

    assert fetches == [url]
    assert label == "Read 2 documents (1 web page opened)"
    assert loaded == ["fresh page text"]


def test_an_unloadable_page_behind_a_completed_document_is_never_refetched(
    conn, tmp_path, monkeypatch
):
    """A finished document's chunks were cut from its stored page, so that page
    stays exactly as it is (an old schema is meant to fail loudly in the evidence
    pane): the conftest fetch_url poison is the guard here."""
    run_id = dbmod.create_run(conn)
    report = _completed_report(conn, tmp_path, run_id)
    upload_id, planned = _link_upload(conn, tmp_path, "https://www.who.int/facts")
    planned.write_bytes(b"")
    doc_id = dbmod.add_document(conn, run_id, "SOURCE", upload_id=upload_id)
    dbmod.add_chunks(
        conn,
        run_id,
        doc_id,
        [{"text": "cut earlier"}],
        FakeEmbedder(dim=DIM).embed(["cut earlier"]),
    )
    poison_providers(monkeypatch)
    payload = {"report_upload_id": report, "source_upload_ids": [upload_id]}

    assert step_ingest(PipelineContext(conn, SETTINGS), run_id, payload) == "Read 2 documents"
    assert planned.read_bytes() == b""


def _stored_page(conn, tmp_path):
    """A web upload whose page the fetch pre-pass already stored (and that loads)."""
    from authorai.ingest import ParsedDocument, ParsedSection, write_snapshot

    url = "https://www.who.int/facts"
    upload_id, planned = _link_upload(conn, tmp_path, url)
    section = ParsedSection(
        title="Key facts",
        page=None,
        text="Two point two billion people lacked safely managed drinking water.",
    )
    write_snapshot(
        planned,
        ParsedDocument(title="Facts", sections=[section], tables=[], figures=[]),
        {"url": url},
    )
    return upload_id, planned


def _link_documents(conn, run_id, upload_id):
    return [
        tuple(row)
        for row in conn.execute(
            "SELECT d.id, (SELECT count(*) FROM chunks c WHERE c.doc_id = d.id) FROM documents d "
            "WHERE d.run_id = ? AND d.upload_id = ?",
            (run_id, upload_id),
        )
    ]


@pytest.mark.parametrize("embedder_ready", [False, True], ids=["no-embedder", "embedder-built"])
def test_retry_over_a_completed_link_document_recomputes_nothing(
    conn, tmp_path, monkeypatch, embedder_ready
):
    """An ingest retry (the retry endpoint, startup requeue) over a web page whose
    document already finished: no fetch, no re-ingest, no duplicate document. The
    conftest poisons are the guard — with no embedder built yet, constructing one
    fails first; with one already built, re-ingesting the stored page does; and
    fetching a stored page again trips the fetch poison either way."""
    from authorai import ingest as ingest_mod

    settings = _dedup_settings(tmp_path)
    run_id = dbmod.create_run(conn)
    report = _completed_report(conn, tmp_path, run_id)
    upload_id, planned = _stored_page(conn, tmp_path)
    doc_id = ingest_mod.ingest_snapshot(
        conn,
        FakeEmbedder(dim=DIM),
        run_id,
        planned,
        kind="SOURCE",
        figures_dir=settings.figures_dir,
        upload_id=upload_id,
    )
    poison_providers(monkeypatch)  # no overrides: the poisons ARE the guard here
    context = PipelineContext(conn, settings)
    if embedder_ready:
        context._embedder = FakeEmbedder(dim=DIM)
    payload = {"report_upload_id": report, "source_upload_ids": [upload_id]}

    assert step_ingest(context, run_id, payload) == "Read 2 documents"
    assert _link_documents(conn, run_id, upload_id) == [(doc_id, 1)]


def test_torn_link_document_is_reingested_from_its_stored_page_without_refetching(
    conn, tmp_path, monkeypatch
):
    """A document row with no chunks is a torn ingest for a link too: it is deleted
    and ingested again from the page as stored — never skipped, never fetched."""
    from authorai import jobs as jobsmod

    run_id = dbmod.create_run(conn)
    report = _completed_report(conn, tmp_path, run_id)
    upload_id, planned = _stored_page(conn, tmp_path)
    stored = planned.read_bytes()
    torn = dbmod.add_document(conn, run_id, "SOURCE", upload_id=upload_id)  # no chunks
    monkeypatch.setattr(
        jobsmod, "fetch_url", lambda *a, **k: pytest.fail("re-fetched a stored page")
    )
    context = PipelineContext(conn, _dedup_settings(tmp_path))
    context._embedder = FakeEmbedder(dim=DIM)
    payload = {"report_upload_id": report, "source_upload_ids": [upload_id]}

    assert step_ingest(context, run_id, payload) == "Read 2 documents"

    rows = _link_documents(conn, run_id, upload_id)
    assert len(rows) == 1
    assert rows[0][0] != torn and rows[0][1] > 0
    assert planned.read_bytes() == stored


def test_link_serving_a_pdf_becomes_a_pdf_upload_and_dedups_by_its_bytes(
    conn, tmp_path, monkeypatch
):
    import hashlib

    from authorai import jobs as jobsmod

    body = b"%PDF-donor"
    digest = hashlib.sha256(body).hexdigest()
    _ingested_upload(conn, tmp_path, content_hash=digest)  # a prior ingest of these exact bytes
    poison_providers(monkeypatch)
    monkeypatch.setattr(
        jobsmod,
        "fetch_url",
        lambda u, s, **k: _fetched(u, body=body, content_type="application/pdf"),
    )
    run_id = dbmod.create_run(conn)
    report = _completed_report(conn, tmp_path, run_id)
    upload_id, planned = _link_upload(conn, tmp_path, "https://example.org/report.pdf")
    payload = {"report_upload_id": report, "source_upload_ids": [upload_id]}

    label = step_ingest(PipelineContext(conn, _dedup_settings(tmp_path)), run_id, payload)
    assert label == "Read 2 documents (1 web page opened, 1 already read)"
    row = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    assert (row["source_type"], row["content_hash"]) == ("pdf", digest)
    assert row["url"] == "https://example.org/report.pdf"
    stored = Path(row["path"])
    assert stored == planned.with_suffix(".pdf") and stored.read_bytes() == body
    assert not planned.exists()


def test_a_link_is_read_out_of_process_within_the_configured_budget(conn, tmp_path, monkeypatch):
    """Reading a page runs in the bounded reader, never in the worker thread, with
    Settings.extract_timeout_seconds as its wall-clock budget (default 60 s,
    AUTHORAI_EXTRACT_TIMEOUT_SECONDS overrides)."""
    from authorai import jobs as jobsmod
    from authorai.ingest import ParsedDocument, ParsedSection

    assert SETTINGS.extract_timeout_seconds == 60.0
    monkeypatch.setenv("AUTHORAI_EXTRACT_TIMEOUT_SECONDS", "12.5")
    settings = Settings(anthropic_api_key="x", openai_api_key="x")
    assert settings.extract_timeout_seconds == 12.5

    from authorai.web import PageMetadata

    reads: list[tuple[str, float]] = []

    def fake_bounded(html, *, url, timeout):
        reads.append((url, timeout))
        section = ParsedSection(title="", page=None, text="page text")
        return ParsedDocument(title=None, sections=[section], tables=[], figures=[]), PageMetadata()

    monkeypatch.setattr(jobsmod, "fetch_url", lambda u, s, **k: _fetched(u))
    monkeypatch.setattr(jobsmod, "extract_web_bounded", fake_bounded)
    monkeypatch.setattr(jobsmod, "ingest_snapshot", lambda *a, **k: "doc")
    run_id = dbmod.create_run(conn)
    report = _completed_report(conn, tmp_path, run_id)
    url = "https://www.who.int/facts"
    upload_id, planned = _link_upload(conn, tmp_path, url)
    payload = {"report_upload_id": report, "source_upload_ids": [upload_id]}
    step_ingest(PipelineContext(conn, settings), run_id, payload)
    assert reads == [(url, 12.5)]
    assert planned.exists()


def test_a_link_too_slow_to_read_fails_the_ingest_step_naming_it(conn, tmp_path, monkeypatch):
    """The REAL bounded reader, given a budget no child process can meet: the run
    fails at ingest, loudly, with the link named, and no page is stored."""
    from authorai import jobs as jobsmod

    run_id = dbmod.create_run(conn)
    report = _completed_report(conn, tmp_path, run_id)
    url = "https://www.who.int/facts"
    upload_id, planned = _link_upload(conn, tmp_path, url)
    poison_providers(monkeypatch)
    monkeypatch.setattr(jobsmod, "fetch_url", lambda u, s, **k: _fetched(u))
    job_id = dbmod.create_job(
        conn, run_id, {"report_upload_id": report, "source_upload_ids": [upload_id]}
    )
    settings = Settings(anthropic_api_key="x", openai_api_key="x", extract_timeout_seconds=0.001)

    run_job(conn, settings, dbmod.claim_next_job(conn))

    job = dbmod.get_job(conn, job_id)
    assert job["status"] == "FAILED"
    assert [(p["step"], p["status"]) for p in job["progress"]] == [("ingest", "failed")]
    assert dbmod.get_run(conn, run_id)["error"] == (
        f"ExtractionTimeoutError: {url} took longer than 0.001 seconds to read "
        "(the page is too large or complex)"
    )
    assert list(planned.parent.iterdir()) == []  # no snapshot, no .part left behind


_JS_SHELL = b"<html><body><div id='app'></div><noscript>Enable JavaScript.</noscript></body></html>"
_UNKNOWN_CHARSET = (
    '<html><head><meta charset="x-bogus-8"></head><body><p>Organización</p></body></html>'
).encode("latin-1")


@pytest.mark.parametrize(
    "body, budget, error_type",
    [
        (_JS_SHELL, 60.0, ThinPageError),
        (_UNKNOWN_CHARSET, 60.0, ValueError),
        (_JS_SHELL, 0.001, ExtractionTimeoutError),
    ],
    ids=["thin", "undecodable", "too-slow"],
)
def test_a_page_unreadable_after_a_redirect_names_the_link_as_added(
    conn, tmp_path, monkeypatch, body, budget, error_type
):
    """The REAL reader only ever sees where a link ended up (a DOI resolving to its
    publisher). The error must still name the link as added, the way fetch.py
    names a redirect — the UI flags a source row only when the error names that
    row's link — and keep its type and original wording for the failure hints."""
    from authorai import jobs as jobsmod
    from authorai.fetch import _shown

    added = "https://doi.org/10.1234/abcd.5678"
    final = "https://journals.publisher.example/article/S0001"
    run_id = dbmod.create_run(conn)
    report = _completed_report(conn, tmp_path, run_id)
    upload_id, planned = _link_upload(conn, tmp_path, added)
    poison_providers(monkeypatch)
    monkeypatch.setattr(
        jobsmod,
        "fetch_url",
        lambda u, s, **k: _fetched(u, body=body, final_url=final, charset=None),
    )
    settings = Settings(anthropic_api_key="x", openai_api_key="x", extract_timeout_seconds=budget)
    payload = {"report_upload_id": report, "source_upload_ids": [upload_id]}

    with pytest.raises(error_type) as raised:
        step_ingest(PipelineContext(conn, settings), run_id, payload)

    if error_type is ExtractionTimeoutError:
        original = (
            f"{final} took longer than 0.001 seconds to read (the page is too large or complex)"
        )
    else:
        with pytest.raises(error_type) as direct:
            extract_web(body, url=final)  # what the reader says, read in this process
        original = str(direct.value)
    assert type(raised.value) is error_type
    assert str(raised.value) == f"{_shown(final)} (redirected from {_shown(added)}): {original}"
    assert list(planned.parent.iterdir()) == []


@pytest.mark.parametrize("redirected", [False, True], ids=["direct", "redirected"])
@pytest.mark.parametrize(
    "error_type, template",
    [
        (ThinPageError, "{url} has no readable article text"),
        (ValueError, "{url} declares an unknown charset 'x-bogus-8'"),
        (ExtractionTimeoutError, "{url} took longer than 60 seconds to read"),
        (RuntimeError, "{url} could not be read: the reader process exited without a result"),
    ],
    ids=["thin", "undecodable", "too-slow", "reader-died"],
)
def test_reading_errors_keep_their_type_and_name_the_added_link_once(
    conn, tmp_path, monkeypatch, error_type, template, redirected
):
    """Without a redirect the reader's message already names the link as added and
    stays byte-identical; after one, the same type carries both addresses and the
    original message, chained to the original exception."""
    from authorai import jobs as jobsmod
    from authorai.fetch import _shown

    added = "https://example.org/app"
    final = "https://www.example.org/app/" if redirected else added
    original = error_type(template.format(url=final))

    def failing_reader(html, *, url, timeout):
        raise original

    monkeypatch.setattr(jobsmod, "fetch_url", lambda u, s, **k: _fetched(u, final_url=final))
    monkeypatch.setattr(jobsmod, "extract_web_bounded", failing_reader)
    run_id = dbmod.create_run(conn)
    report = _completed_report(conn, tmp_path, run_id)
    upload_id, _ = _link_upload(conn, tmp_path, added)
    payload = {"report_upload_id": report, "source_upload_ids": [upload_id]}

    with pytest.raises(error_type) as raised:
        step_ingest(PipelineContext(conn, SETTINGS), run_id, payload)

    assert type(raised.value) is error_type
    if redirected:
        expected = f"{_shown(final)} (redirected from {_shown(added)}): {original}"
        assert str(raised.value) == expected
        assert raised.value.__cause__ is original
    else:
        assert raised.value is original
        assert str(raised.value).count(added) == 1


def test_step_ingest_label_counts_opened_pages_in_plain_words(conn, tmp_path, monkeypatch):
    """Every link this attempt opened counts, pluralized; a zero count is left out."""
    from authorai import jobs as jobsmod

    bodies = iter([b"%PDF-first", b"%PDF-second"])
    monkeypatch.setattr(
        jobsmod,
        "fetch_url",
        lambda u, s, **k: _fetched(u, body=next(bodies), content_type="application/pdf"),
    )
    monkeypatch.setattr(jobsmod, "ingest_pdf", lambda *a, **k: None)
    run_id = dbmod.create_run(conn)
    report = _completed_report(conn, tmp_path, run_id)
    first, _ = _link_upload(conn, tmp_path, "https://example.org/a.pdf")
    second, _ = _link_upload(conn, tmp_path, "https://example.org/b.pdf")
    payload = {"report_upload_id": report, "source_upload_ids": [first, second]}
    label = step_ingest(PipelineContext(conn, _dedup_settings(tmp_path)), run_id, payload)
    assert label == "Read 3 documents (2 web pages opened)"


def test_youtube_links_fail_loudly_until_supported(conn, tmp_path, monkeypatch):
    from authorai import jobs as jobsmod

    run_id = dbmod.create_run(conn)
    report = _completed_report(conn, tmp_path, run_id)
    upload_id, _ = _link_upload(
        conn, tmp_path, "https://www.youtube.com/watch?v=abc123def45", source_type="youtube"
    )
    monkeypatch.setattr(jobsmod, "fetch_url", lambda *a, **k: pytest.fail("fetched a video page"))
    payload = {"report_upload_id": report, "source_upload_ids": [upload_id]}
    with pytest.raises(ValueError, match="YouTube"):
        step_ingest(PipelineContext(conn, SETTINGS), run_id, payload)


def test_a_page_whose_charset_only_the_http_header_names_is_decoded_with_it(
    conn, tmp_path, monkeypatch
):
    """Latin-1 bytes, no <meta> charset: only the Content-Type header says how to
    read them, and only the fetch step has that header."""
    from authorai import jobs as jobsmod
    from authorai.fetch import FetchedResponse
    from authorai.ingest import load_snapshot

    run_id = dbmod.create_run(conn)
    report = _completed_report(conn, tmp_path, run_id)
    url = "https://www.salud.example.gob/agua"
    upload_id, planned = _link_upload(conn, tmp_path, url)
    paragraph = (
        "La Organización Mundial de la Salud informó que el acceso al agua potable mejoró "
        "en la región durante la última década, aunque millones de personas todavía "
        "carecen de servicios básicos."
    )
    # Two DIFFERENT paragraphs: trafilatura drops a repeated identical paragraph,
    # and one alone is under the thin-page floor.
    second = (
        "Los gobiernos regionales ampliaron las redes de distribución y los sistemas de "
        "tratamiento, con inversiones concentradas en las zonas rurales afectadas por la sequía."
    )
    html = (
        "<html><head><title>Agua potable</title></head><body><article><h1>Agua potable</h1>"
        f"<p>{paragraph}</p><p>{second}</p></article></body></html>"
    ).encode("latin-1")
    monkeypatch.setattr(
        jobsmod,
        "fetch_url",
        lambda u, s, **k: FetchedResponse(
            url=u,
            final_url=u,
            content_type="text/html",
            charset="iso-8859-1",
            body=html,
            is_pdf=False,
        ),
    )
    monkeypatch.setattr(jobsmod, "ingest_snapshot", lambda *a, **k: "doc")
    payload = {"report_upload_id": report, "source_upload_ids": [upload_id]}
    step_ingest(PipelineContext(conn, SETTINGS), run_id, payload)
    parsed, _ = load_snapshot(planned)
    assert "Organización Mundial de la Salud" in " ".join(s.text for s in parsed.sections)


_SPANISH_PAGE = "<p>La Organización Mundial — “agua”, €5 millones, œuvre</p>"


def _header_labeled(body: bytes, charset: str):
    from authorai.fetch import FetchedResponse

    url = "https://www.salud.example.gob/agua"
    return FetchedResponse(
        url=url, final_url=url, content_type="text/html", charset=charset, body=body, is_pdf=False
    )


@pytest.mark.parametrize(
    "body, charset",
    [
        # Valid UTF-8 wins over a wrong header label (servers defaulting to ISO-8859-1).
        (_SPANISH_PAGE.encode("utf-8"), "iso-8859-1"),
        # An iso-8859-1 label reads as windows-1252: 0x80 €, 0x93/0x94 quotes, 0x97 dash, 0x9C œ.
        (_SPANISH_PAGE.encode("cp1252"), "iso-8859-1"),
        # A UTF-16 byte-order mark beats the header, as in browsers and web._decode.
        (codecs.BOM_UTF16_LE + _SPANISH_PAGE.encode("utf-16-le"), "utf-8"),
        (codecs.BOM_UTF16_BE + _SPANISH_PAGE.encode("utf-16-be"), "iso-8859-1"),
    ],
    ids=[
        "utf8-bytes-mislabeled",
        "cp1252-bytes-labeled-latin1",
        "utf16le-bom-labeled-utf8",
        "utf16be-bom-labeled-latin1",
    ],
)
def test_header_charset_decodes_like_a_browser_except_a_bom_or_valid_utf8_wins(body, charset):
    from authorai.jobs import _page_text
    from authorai.web import _decode

    fetched = _header_labeled(body, charset)
    result = _page_text(fetched)
    assert (result if isinstance(result, str) else _decode(result, fetched.url)) == _SPANISH_PAGE


@pytest.mark.parametrize(
    "charset",
    # Python codecs that are not text encodings (bytes.decode refuses them with a
    # LookupError), and text codecs that refuse any byte ("undefined", "idna",
    # "punycode" raise a UnicodeError even with errors="replace").
    ["base64", "zlib_codec", "bz2_codec", "rot13", "hex", "quoted-printable", "uu"]
    + ["undefined", "idna", "punycode"],
)
def test_a_header_charset_naming_no_text_encoding_leaves_the_page_bytes_to_the_reader(charset):
    from authorai.jobs import _page_text

    fetched = _header_labeled(_SPANISH_PAGE.encode("cp1252"), charset)
    assert _page_text(fetched) is fetched.body


def test_a_page_whose_header_charset_is_no_text_encoding_fails_naming_the_link(
    conn, tmp_path, monkeypatch
):
    """Not a LookupError about Python codecs: the reader's own decoding failure,
    which names the link (and the UI can flag its row)."""
    from authorai import jobs as jobsmod

    run_id = dbmod.create_run(conn)
    report = _completed_report(conn, tmp_path, run_id)
    upload_id, planned = _link_upload(conn, tmp_path, "https://www.salud.example.gob/agua")
    monkeypatch.setattr(
        jobsmod,
        "fetch_url",
        lambda u, s, **k: _header_labeled(_SPANISH_PAGE.encode("cp1252"), "base64"),
    )
    payload = {"report_upload_id": report, "source_upload_ids": [upload_id]}
    with pytest.raises(ValueError, match="^https://www.salud.example.gob/agua is not valid UTF-8"):
        step_ingest(PipelineContext(conn, SETTINGS), run_id, payload)
    assert list(planned.parent.iterdir()) == []
