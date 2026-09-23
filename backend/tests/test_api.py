"""API tests: fail-closed startup, structural auth, upload validation, report shape.

TestClient is always used as a context manager — otherwise the lifespan
(where fail-closed and recovery live) never runs and the tests would pass
against an app that can't actually start.
"""

import json
import threading
import time
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.routing import Route

from authorai import db as dbmod
from authorai.config import Settings
from authorai.jobs import Worker
from authorai.main import create_app
from authorai.references import UNPAYWALL_BASE, Reference, ReferenceList
from tests.conftest import DIM, FakeLLM, pdf_with_pages, poison_providers, reader_that_never_answers

# Routes that are intentionally open (no API key). Everything else must 401.
OPEN_PATHS = {"/health", "/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"}

KEY = "test-key"
AUTH = {"X-API-Key": KEY}
PDF_BYTES = b"%PDF-1.4 fake pdf content"


class _NoopWorker:
    """Stands in for the background worker so tests drain jobs synchronously."""

    def start(self):
        pass

    def stop(self, timeout=None):
        pass


def _settings(tmp_path, **overrides) -> Settings:
    defaults = dict(
        api_key=KEY,
        db_path=tmp_path / "api.db",
        uploads_dir=tmp_path / "uploads",
        figures_dir=tmp_path / "figures",
        embedding_dim=DIM,
        anthropic_api_key="x",
        openai_api_key="x",
    )
    return Settings(**{**defaults, **overrides})


@pytest.fixture()
def client(tmp_path):
    app = create_app(_settings(tmp_path), worker=_NoopWorker())
    with TestClient(app) as test_client:
        yield test_client


def _upload_files(source_count=1):
    return [
        ("report", ("report.pdf", PDF_BYTES, "application/pdf")),
        *[
            ("sources", (f"source{i}.pdf", PDF_BYTES, "application/pdf"))
            for i in range(source_count)
        ],
    ]


def test_health_is_open(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_startup_fails_closed_without_api_key(tmp_path):
    app = create_app(_settings(tmp_path, api_key=None), worker=_NoopWorker())
    with pytest.raises(RuntimeError, match="AUTHORAI_API_KEY"), TestClient(app):
        pass


def test_every_api_route_requires_the_key(tmp_path):
    """Every route on the /api router must 401 without a key and with a wrong
    key — a new endpoint that forgets auth fails here (auth is enforced by the
    middleware over the whole /api prefix, so this covers new routes for free)."""
    from authorai.api import router as api_router

    app = create_app(_settings(tmp_path), worker=_NoopWorker())
    guarded = 0
    with TestClient(app) as client:
        for route in api_router.routes:
            url = (
                route.path.replace("{run_id}", "x")
                .replace("{job_id}", "x")
                .replace("{doc_id}", "y")
            )
            for method in (route.methods or set()) - {"HEAD", "OPTIONS"}:
                assert client.request(method, url).status_code == 401, f"{method} {url}"
                wrong = client.request(method, url, headers={"X-API-Key": "wrong"})
                assert wrong.status_code == 401, f"{method} {url} with wrong key"
                guarded += 1
    assert guarded >= 5  # sanity: the router is actually mounted


def test_nothing_sensitive_is_mounted_outside_the_api_prefix(tmp_path):
    """The middleware guards the /api prefix; a route mounted on the app
    directly would bypass it. Every top-level app route must therefore be a
    /api route or explicitly open-listed."""
    app = create_app(_settings(tmp_path), worker=_NoopWorker())
    for route in app.routes:
        path = getattr(route, "path", "")
        if isinstance(route, APIRoute | Route) and not path.startswith("/api"):
            assert path in OPEN_PATHS, f"{path} is mounted outside /api and not open-listed"


def test_valid_key_reaches_the_route(client):
    # A regression that rejected EVERY key (not just wrong ones) must be caught.
    assert client.get("/api/runs", headers=AUTH).status_code == 200


def test_key_check_handles_non_ascii_without_crashing():
    """Finding: comparing a decoded non-ASCII header raises TypeError → 500.
    The byte-level comparison must return a clean False instead."""
    from authorai.api import _key_ok

    assert _key_ok("café".encode(), "test-key") is False
    assert _key_ok(b"test-key", "test-key") is True
    assert _key_ok(b"anything", None) is False  # fail closed when unset


def test_oversize_content_length_is_rejected_before_the_body(tmp_path):
    settings = _settings(tmp_path, max_request_bytes=1000)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        response = client.post(
            "/api/runs",
            headers={**AUTH, "Content-Length": "5000"},
            content=b"x" * 5000,
        )
        assert response.status_code == 413


def test_startup_requeues_running_jobs(tmp_path):
    settings = _settings(tmp_path)
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    run_id = dbmod.create_run(conn)
    job_id = dbmod.create_job(conn, run_id, {"report_upload_id": "r", "source_upload_ids": []})
    dbmod.claim_next_job(conn)  # leave it RUNNING, as a crash would
    conn.close()

    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        job = client.get(f"/api/jobs/{job_id}", headers=AUTH).json()
    assert job["status"] == "QUEUED"
    assert any(p["step"] == "recovered" for p in job["progress"])


def test_upload_rows_carry_sha256_content_hash(tmp_path):
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        run_id = client.post("/api/runs", headers=AUTH, files=_upload_files(source_count=2)).json()[
            "run_id"
        ]
    import hashlib

    expected = hashlib.sha256(PDF_BYTES).hexdigest()
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    hashes = [
        row["content_hash"]
        for row in conn.execute(
            "SELECT u.content_hash FROM uploads u"
            " JOIN jobs j ON j.run_id = ?"
            " WHERE u.id IN (SELECT value FROM json_each(j.payload, '$.source_upload_ids')"
            "                UNION SELECT json_extract(j.payload, '$.report_upload_id'))",
            (run_id,),
        )
    ]
    conn.close()
    assert hashes and all(h == expected for h in hashes)


def test_second_identical_upload_reuses_ingest_end_to_end(tmp_path, monkeypatch):
    """The API seam for dedup: POST identical files twice; run 2's REAL
    ingest step reuses run 1's rows — its progress label says so, and no
    parser, embedder, or LLM client is even constructed — yet run 2 still
    serves its own PDF after run 1 is deleted (copy-never-share over HTTP)."""
    from authorai import jobs as jobsmod
    from authorai.embeddings import FakeEmbedder
    from authorai.jobs import step_ingest

    settings = _settings(tmp_path)

    def fake_ingest_pdf(
        conn, embedder, run_id, path, *, kind, figures_dir, upload_id, describe, fallback_title
    ):
        doc_id = dbmod.add_document(
            conn,
            run_id,
            kind,
            upload_id=upload_id,
            title=fallback_title,
            # What the real ingest stamps — the donor filter matches on it.
            embedding_model=settings.embedding_model,
        )
        png = Path(figures_dir) / run_id / doc_id / "fig-1.png"
        png.parent.mkdir(parents=True)
        png.write_bytes(b"png bytes")
        figure_id = dbmod.add_figure(conn, run_id, doc_id, str(png), page=1)
        texts = ["alpha wheat facts", "figure about crops"]
        chunks = [{"text": texts[0]}, {"text": texts[1], "kind": "figure", "figure_id": figure_id}]
        dbmod.add_chunks(conn, run_id, doc_id, chunks, FakeEmbedder(dim=DIM).embed(texts))
        return doc_id

    def _fake(name):
        def step(context, run_id, payload):
            return f"{name} ok"

        return step

    steps = {
        "ingest": step_ingest,
        "extract": _fake("extract"),
        "verify": _fake("verify"),
        "score": _fake("score"),
    }

    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        first = client.post("/api/runs", headers=AUTH, files=_upload_files()).json()["run_id"]
        conn = dbmod.connect(settings.db_path, settings.embedding_dim)
        monkeypatch.setattr(jobsmod, "ingest_pdf", fake_ingest_pdf)
        assert Worker(settings, steps=steps).run_pending(conn) == 1

        second = client.post("/api/runs", headers=AUTH, files=_upload_files()).json()["run_id"]
        poison_providers(monkeypatch)
        assert Worker(settings, steps=steps).run_pending(conn) == 1
        report_doc = conn.execute(
            "SELECT id FROM documents WHERE run_id = ? AND kind = 'REPORT'", (second,)
        ).fetchone()["id"]
        conn.close()

        detail = client.get(f"/api/runs/{second}", headers=AUTH).json()
        ingest = next(p for p in detail["job"]["progress"] if p["step"] == "ingest")
        assert ingest["label"] == "Read 2 documents (2 already read)"
        assert detail["run"]["status"] == "DONE"

        # Run 1's report and source were ALSO the same bytes, so within-run
        # reuse already fired there: the source copied the report's ingest.
        first_detail = client.get(f"/api/runs/{first}", headers=AUTH).json()
        first_ingest = next(p for p in first_detail["job"]["progress"] if p["step"] == "ingest")
        assert first_ingest["label"] == "Read 2 documents (1 already read)"

        assert client.delete(f"/api/runs/{first}", headers=AUTH).status_code == 204
        served = client.get(f"/api/runs/{second}/documents/{report_doc}/file", headers=AUTH)
        assert served.status_code == 200
        assert served.content == PDF_BYTES


def test_run_title_defaults_to_report_filename_stem(tmp_path):
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        # No title given -> the report filename stem.
        run_id = client.post("/api/runs", headers=AUTH, files=_upload_files()).json()["run_id"]
        detail = client.get(f"/api/runs/{run_id}", headers=AUTH).json()
        assert detail["run"]["title"] == "report"
        assert client.get(f"/api/runs/{run_id}/report", headers=AUTH).json()["title"] == "report"

        # An explicit title wins; whitespace-only falls back to the stem.
        titled = client.post(
            "/api/runs", headers=AUTH, files=_upload_files(), data={"title": "My Study"}
        ).json()["run_id"]
        assert client.get(f"/api/runs/{titled}", headers=AUTH).json()["run"]["title"] == "My Study"
        blank = client.post(
            "/api/runs", headers=AUTH, files=_upload_files(), data={"title": "   "}
        ).json()["run_id"]
        assert client.get(f"/api/runs/{blank}", headers=AUTH).json()["run"]["title"] == "report"

        # A megabyte of title must not ride every future gallery load.
        too_long = client.post(
            "/api/runs", headers=AUTH, files=_upload_files(), data={"title": "x" * 201}
        )
        assert too_long.status_code == 400
        assert "200 characters" in too_long.json()["detail"]


def test_report_exposes_stored_score_details(tmp_path):
    settings = _settings(tmp_path)
    run_id = _seed_scored_run(settings)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        report = client.get(f"/api/runs/{run_id}/report", headers=AUTH).json()
        assert report["accuracy_detail"]["correct"] == 1
        assert report["accuracy_detail"]["disavowed"] == 0
        assert report["credibility_detail"]["method"] == "usage_weighted_mean"
        assert report["credibility_detail"]["sources"][0]["usage"] == 1
        component = report["validity_detail"]["components"]["coverage"]
        assert component["justification"] == "treats its stated scope"
        assert report["validity_detail"]["weights_used"] == {"coverage": 1.0}


def test_report_details_tolerate_minimal_legacy_shapes(tmp_path):
    """Rows scored before a field existed lack the key — the endpoint must
    return None-filled details, never 500."""
    settings = _settings(tmp_path)
    run_id = _seed_scored_run(settings)
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    dbmod.save_run_scores(
        conn,
        run_id,
        accuracy={
            "supported": 1,
            "contradicted": 0,
            "unverifiable": 1,
            "total": 2,
            "accuracy": 1.0,
            "coverage": 0.5,
        },
        credibility={"score": 62.5, "weighting": "usage"},
        validity={"score": 71.0, "components": {}},
    )
    conn.close()
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        report = client.get(f"/api/runs/{run_id}/report", headers=AUTH).json()
        assert report["accuracy_detail"]["correct"] is None
        assert report["credibility_detail"]["method"] is None
        assert report["validity_detail"]["weights_used"] is None
        assert report["scores"]["accuracy"] == 1.0


def test_report_details_null_when_unscored(client):
    run_id = client.post("/api/runs", headers=AUTH, files=_upload_files()).json()["run_id"]
    report = client.get(f"/api/runs/{run_id}/report", headers=AUTH).json()
    assert report["scores"] is None
    assert report["accuracy_detail"] is None
    assert report["validity_detail"] is None
    assert report["credibility_detail"] is None


def test_delete_run_removes_every_trace(tmp_path):
    """The destructive path: every table, both search indexes, and the files."""
    settings = _settings(tmp_path)
    keeper = _seed_scored_run(settings)  # proves isolation
    target = _seed_scored_run(settings)
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    target_files = [
        row["path"]
        for row in conn.execute(
            "SELECT u.path FROM uploads u JOIN documents d ON d.upload_id = u.id"
            " WHERE d.run_id = ?",
            (target,),
        )
    ]
    conn.close()
    assert target_files and all(Path(p).exists() for p in target_files)

    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        assert client.delete(f"/api/runs/{target}", headers=AUTH).status_code == 204
        # Gone from the API…
        assert client.get(f"/api/runs/{target}", headers=AUTH).status_code == 404
        remaining = client.get("/api/runs", headers=AUTH).json()["runs"]
        assert [run["id"] for run in remaining] == [keeper]

    # …and from every table, both search indexes, and the disk.
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    for table in (
        "runs",
        "documents",
        "chunks",
        "claims",
        "verdicts",
        "run_scores",
        "source_credibility",
        "jobs",
    ):
        count = conn.execute(
            f"SELECT count(*) FROM {table} WHERE {'id' if table == 'runs' else 'run_id'} = ?",
            (target,),
        ).fetchone()[0]
        assert count == 0, f"{table} still has rows for the deleted run"
    assert (
        conn.execute("SELECT count(*) FROM chunks_vec WHERE run_id = ?", (target,)).fetchone()[0]
        == 0
    )
    # The keeper run's data survives untouched.
    assert conn.execute("SELECT count(*) FROM chunks WHERE run_id = ?", (keeper,)).fetchone()[0]
    conn.close()
    assert all(not Path(p).exists() for p in target_files)


def test_delete_run_removes_its_figure_directory_and_no_other_runs(tmp_path):
    """Figure images are not uploads (no upload row names them): deletion removes
    the run's figure directory as a whole, and leaves another run's in place."""
    settings = _settings(tmp_path)
    keeper = _seed_scored_run(settings)
    target = _seed_scored_run(settings)
    figures = {}
    for run_id in (keeper, target):
        png = settings.run_figures_dir(run_id) / dbmod.new_id() / "fig-1.png"
        png.parent.mkdir(parents=True)
        png.write_bytes(b"\x89PNG\r\n\x1a\n figure bytes")
        figures[run_id] = png

    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        assert client.delete(f"/api/runs/{target}", headers=AUTH).status_code == 204

    assert not settings.run_figures_dir(target).exists()
    assert figures[keeper].read_bytes() == b"\x89PNG\r\n\x1a\n figure bytes"


def test_delete_refuses_active_jobs_and_unknown_runs(tmp_path):
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        run_id = client.post("/api/runs", headers=AUTH, files=_upload_files()).json()["run_id"]
        # QUEUED job → refuse.
        assert client.delete(f"/api/runs/{run_id}", headers=AUTH).status_code == 409
        conn = dbmod.connect(settings.db_path, settings.embedding_dim)
        dbmod.claim_next_job(conn)  # job now RUNNING → still refuse
        conn.close()
        assert client.delete(f"/api/runs/{run_id}", headers=AUTH).status_code == 409
        conn = dbmod.connect(settings.db_path, settings.embedding_dim)
        job = dbmod.get_run_job(conn, run_id)
        dbmod.finish_job_and_run(conn, job["id"], run_id, "FAILED", error="boom")
        conn.close()
        # FAILED → deletable, files included.
        assert client.delete(f"/api/runs/{run_id}", headers=AUTH).status_code == 204
        assert not any(Path(settings.uploads_dir).glob("*.pdf"))
        assert client.delete("/api/runs/nope", headers=AUTH).status_code == 404

        # A corrupted job payload must never wedge deletion (404-forever class).
        poisoned = client.post("/api/runs", headers=AUTH, files=_upload_files()).json()["run_id"]
        conn = dbmod.connect(settings.db_path, settings.embedding_dim)
        job = dbmod.get_run_job(conn, poisoned)
        dbmod.finish_job_and_run(conn, job["id"], poisoned, "FAILED", error="boom")
        conn.execute("UPDATE jobs SET payload = 'not json' WHERE run_id = ?", (poisoned,))
        conn.commit()
        conn.close()
        assert client.delete(f"/api/runs/{poisoned}", headers=AUTH).status_code == 204


@pytest.mark.parametrize("recorded", ["absolute", "dot-dot", "symlinked-folder", "sibling-prefix"])
def test_delete_run_never_removes_a_file_outside_the_uploads_folder(tmp_path, recorded):
    """A CLI ingest records the user's ORIGINAL file (ingest_parsed with no upload
    id stores its absolute path): deleting that run from the gallery removes its
    rows, never the user's own PDF — nor one reached by climbing out of uploads,
    through a folder inside uploads that links out, or kept in a sibling folder
    whose name merely starts with the uploads folder's."""
    settings = _settings(tmp_path)
    # "uploads-archive" shares the uploads folder's name as a string prefix, so
    # only a whole-component check (not a string prefix) keeps it out.
    folder = "uploads-archive" if recorded == "sibling-prefix" else "library"
    library = tmp_path / folder / "my_only_copy.pdf"
    library.parent.mkdir()
    library.write_bytes(PDF_BYTES)
    settings.uploads_dir.mkdir()  # so the dot-dot path really reaches the library
    if recorded == "symlinked-folder":  # inside uploads by spelling, outside on disk
        (settings.uploads_dir / "linked").symlink_to(library.parent, target_is_directory=True)
    path = {
        "absolute": str(library.resolve()),
        "dot-dot": str(settings.uploads_dir / ".." / "library" / library.name),
        "symlinked-folder": str(settings.uploads_dir / "linked" / library.name),
        "sibling-prefix": str(library.resolve()),
    }[recorded]
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    run_id = dbmod.create_run(conn)
    upload = dbmod.add_upload(conn, "REPORT", library.name, path)
    dbmod.add_document(conn, run_id, "REPORT", upload_id=upload)
    conn.close()

    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        assert client.delete(f"/api/runs/{run_id}", headers=AUTH).status_code == 204
        assert client.get(f"/api/runs/{run_id}", headers=AUTH).status_code == 404
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    assert conn.execute("SELECT count(*) FROM uploads WHERE id = ?", (upload,)).fetchone()[0] == 0
    conn.close()
    assert library.read_bytes() == PDF_BYTES


def test_delete_run_keeps_a_stored_file_another_run_still_names(tmp_path, monkeypatch):
    """The CLI can ingest a PDF straight out of the uploads folder, recording it by
    its absolute path while the uploading run's row names it relative to the
    server's working directory (the real default, data/uploads). Deleting the CLI
    run leaves the file to that run; deleting the last run naming it removes it."""
    monkeypatch.chdir(tmp_path)
    settings = _settings(tmp_path, uploads_dir=Path("uploads"))
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        api_run = client.post("/api/runs", headers=AUTH, files=_upload_files()).json()["run_id"]
        conn = dbmod.connect(settings.db_path, settings.embedding_dim)
        [stored] = [
            Path(row["path"])
            for row in conn.execute("SELECT path FROM uploads WHERE kind = 'REPORT'")
        ]
        assert not stored.is_absolute()
        cli_run = dbmod.create_run(conn)
        cli_upload = dbmod.add_upload(conn, "REPORT", "report.pdf", str(stored.resolve()))
        dbmod.add_document(conn, cli_run, "REPORT", upload_id=cli_upload)

        assert client.delete(f"/api/runs/{cli_run}", headers=AUTH).status_code == 204
        assert stored.read_bytes() == PDF_BYTES

        job = dbmod.get_run_job(conn, api_run)
        dbmod.finish_job_and_run(conn, job["id"], api_run, "FAILED", error="boom")
        assert client.delete(f"/api/runs/{api_run}", headers=AUTH).status_code == 204
        assert not stored.exists()
        conn.close()


@pytest.mark.parametrize("deleted", ["uploading", "cli"])
def test_delete_run_keeps_a_stored_file_another_run_names_in_other_letter_case(
    tmp_path, monkeypatch, deleted
):
    """A case-insensitive volume (macOS APFS, the default) opens a stored PDF under
    its name in any letter case, so a CLI row spelling it <ID>.PDF names the same
    file as the uploading run's <id>.pdf. Deleting either run leaves the other's
    file served; deleting the last run naming it removes it."""
    monkeypatch.chdir(tmp_path)
    settings = _settings(tmp_path, uploads_dir=Path("uploads"))
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        api_run = client.post("/api/runs", headers=AUTH, files=_upload_files()).json()["run_id"]
        conn = dbmod.connect(settings.db_path, settings.embedding_dim)
        [(upload_id, stored)] = [
            (row["id"], Path(row["path"]))
            for row in conn.execute("SELECT id, path FROM uploads WHERE kind = 'REPORT'")
        ]
        alias = stored.resolve().with_name(stored.name.upper())
        if not alias.exists():
            conn.close()
            pytest.skip("case-sensitive volume: names differing in case are different files")
        api_doc = dbmod.add_document(conn, api_run, "REPORT", upload_id=upload_id)
        cli_run = dbmod.create_run(conn)
        cli_upload = dbmod.add_upload(conn, "REPORT", "report.pdf", str(alias))
        cli_doc = dbmod.add_document(conn, cli_run, "REPORT", upload_id=cli_upload)
        job = dbmod.get_run_job(conn, api_run)
        dbmod.finish_job_and_run(conn, job["id"], api_run, "FAILED", error="boom")
        conn.close()
        runs = {"uploading": (api_run, api_doc), "cli": (cli_run, cli_doc)}
        [(kept_run, kept_doc)] = [ids for name, ids in runs.items() if name != deleted]

        assert client.delete(f"/api/runs/{runs[deleted][0]}", headers=AUTH).status_code == 204
        served = client.get(f"/api/runs/{kept_run}/documents/{kept_doc}/file", headers=AUTH)
        assert served.status_code == 200
        assert served.content == PDF_BYTES

        assert client.delete(f"/api/runs/{kept_run}", headers=AUTH).status_code == 204
        assert not stored.exists()


@pytest.mark.parametrize("broken", ["missing", "symlink-loop", "nul-byte"])
def test_a_stored_path_naming_no_readable_file_never_blocks_deleting_a_run(tmp_path, broken):
    """Deleting a run checks its files against every other upload's path. One path
    that names no readable file (a CLI-ingested PDF since removed, a folder later
    replaced by a symlink loop, a tampered row with a NUL byte) must not fail that
    check and make every run undeletable — nor its own."""
    settings = _settings(tmp_path)
    library = tmp_path / "library"
    library.mkdir()
    (library / "a").symlink_to(library / "b")
    (library / "b").symlink_to(library / "a")
    path = {
        "missing": str(library / "gone.pdf"),
        "symlink-loop": str(library / "a" / "looped.pdf"),
        "nul-byte": str(library / "tampered\x00.pdf"),
    }[broken]
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    broken_run = dbmod.create_run(conn)
    upload = dbmod.add_upload(conn, "REPORT", "broken.pdf", path)
    dbmod.add_document(conn, broken_run, "REPORT", upload_id=upload)
    [stored] = [row["path"] for row in conn.execute("SELECT path FROM uploads")]
    assert stored == path  # the row really holds the unreadable spelling
    conn.close()
    other_run = _seed_scored_run(settings)

    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        assert client.delete(f"/api/runs/{other_run}", headers=AUTH).status_code == 204
        assert client.delete(f"/api/runs/{broken_run}", headers=AUTH).status_code == 204
        assert client.get("/api/runs", headers=AUTH).json()["runs"] == []
    assert sorted(p.name for p in settings.uploads_dir.iterdir()) == []


@pytest.mark.parametrize("stored", [".json", ".pdf"], ids=["page", "pdf"])
def test_delete_run_removes_every_file_a_link_upload_can_leave(tmp_path, stored):
    """A link's files all share its planned name: the page or PDF its row names,
    plus what an interrupted fetch left behind (a PDF stored but never recorded,
    a half-written .part). Deleting the run reclaims every one of them."""
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        run_id = client.post(
            "/api/runs",
            headers=AUTH,
            files=_upload_files(source_count=0),
            data={"source_urls": ["https://example.org/report"]},
        ).json()["run_id"]
        conn = dbmod.connect(settings.db_path, settings.embedding_dim)
        link = conn.execute("SELECT id, path FROM uploads WHERE url IS NOT NULL").fetchone()
        planned = Path(link["path"])
        if stored == ".pdf":
            dbmod.record_fetch(
                conn,
                link["id"],
                path=str(planned.with_suffix(".pdf")),
                source_type="pdf",
                content_hash="abc",
            )
        for suffix in (".json", ".pdf", ".json.part", ".pdf.part"):
            planned.with_name(planned.stem + suffix).write_bytes(b"left behind")
        job = dbmod.get_run_job(conn, run_id)
        dbmod.finish_job_and_run(conn, job["id"], run_id, "FAILED", error="database is locked")
        conn.close()

        assert client.delete(f"/api/runs/{run_id}", headers=AUTH).status_code == 204
    assert sorted(p.name for p in settings.uploads_dir.iterdir()) == []


def test_report_sources_include_extracted_metadata(tmp_path):
    settings = _settings(tmp_path)
    run_id = _seed_scored_run(settings)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        [source] = client.get(f"/api/runs/{run_id}/report", headers=AUTH).json()["sources"]
        assert source["metadata"] == {"title": "The Source Report"}


def test_runs_list_is_enriched(tmp_path):
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        run_id = client.post(
            "/api/runs",
            headers=AUTH,
            files=_upload_files(source_count=2),
            data={"title": "Fresh"},
        ).json()["run_id"]
        [item] = client.get("/api/runs", headers=AUTH).json()["runs"]
        assert item["id"] == run_id
        assert item["title"] == "Fresh"
        assert item["source_count"] == 2
        assert item["scores"] is None


def test_runs_list_scores_match_report_and_jobless_run_degrades(tmp_path):
    settings = _settings(tmp_path)
    seeded = _seed_scored_run(settings)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        [item] = client.get("/api/runs", headers=AUTH).json()["runs"]
        report = client.get(f"/api/runs/{seeded}/report", headers=AUTH).json()
        assert item["scores"] == report["scores"]
        # The seeded run has no jobs row: source_count and title degrade, not 500.
        assert item["source_count"] is None
        assert item["title"] is None


def test_run_detail_includes_uploads_report_first(tmp_path):
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        run_id = client.post("/api/runs", headers=AUTH, files=_upload_files(source_count=2)).json()[
            "run_id"
        ]
        uploads = client.get(f"/api/runs/{run_id}", headers=AUTH).json()["uploads"]
        assert [u["kind"] for u in uploads] == ["REPORT", "SOURCE", "SOURCE"]
        assert uploads[0]["file_name"] == "report.pdf"
        assert [u["file_name"] for u in uploads[1:]] == ["source0.pdf", "source1.pdf"]


def test_run_detail_uploads_empty_for_jobless_run(tmp_path):
    settings = _settings(tmp_path)
    seeded = _seed_scored_run(settings)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        assert client.get(f"/api/runs/{seeded}", headers=AUTH).json()["uploads"] == []


def test_post_runs_queues_job_and_worker_completes_it(tmp_path):
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        response = client.post("/api/runs", headers=AUTH, files=_upload_files(source_count=2))
        assert response.status_code == 202
        body = response.json()
        run_id, job_id = body["run_id"], body["job_id"]

        detail = client.get(f"/api/runs/{run_id}", headers=AUTH).json()
        assert detail["run"]["status"] == "CREATED"
        assert detail["job"]["id"] == job_id
        assert detail["job"]["status"] == "QUEUED"
        assert len(detail["job"]["payload"]["source_upload_ids"]) == 2

        # Drain the queue synchronously with fake steps (the real worker was
        # replaced by the noop), then the run and job must both read DONE.
        record: list[str] = []
        conn = dbmod.connect(settings.db_path, settings.embedding_dim)

        def _step(name):
            def step(context, run_id_, payload):
                record.append(name)
                return f"{name} ok"

            return step

        steps = {name: _step(name) for name in ("ingest", "extract", "verify", "score")}
        assert Worker(settings, steps=steps).run_pending(conn) == 1
        conn.close()
        assert record == ["ingest", "extract", "verify", "score"]

        detail = client.get(f"/api/runs/{run_id}", headers=AUTH).json()
        assert detail["run"]["status"] == "DONE"
        assert detail["job"]["status"] == "DONE"
        runs = client.get("/api/runs", headers=AUTH).json()["runs"]
        assert [r["id"] for r in runs] == [run_id]


@pytest.mark.parametrize(
    ("file_name", "content", "expected_status"),
    [
        ("report.txt", PDF_BYTES, 400),  # wrong extension
        ("report.pdf", b"not a pdf at all", 400),  # wrong magic
        ("report.pdf", b"%PDF-" + b"x" * 2000, 413),  # over the size cap
    ],
)
def test_bad_uploads_are_rejected_before_any_rows(tmp_path, file_name, content, expected_status):
    settings = _settings(tmp_path, max_upload_bytes=1000)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        files = [
            ("report", (file_name, content, "application/pdf")),
            ("sources", ("source.pdf", PDF_BYTES, "application/pdf")),
        ]
        assert client.post("/api/runs", headers=AUTH, files=files).status_code == expected_status

    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM uploads").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    conn.close()
    # No orphaned blob on disk either — validation happens before any write.
    assert not any((settings.uploads_dir).glob("*.pdf")) if settings.uploads_dir.exists() else True


def test_too_many_sources_is_rejected(tmp_path):
    settings = _settings(tmp_path, max_source_files=2)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        assert (
            client.post("/api/runs", headers=AUTH, files=_upload_files(source_count=3)).status_code
            == 400
        )
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    conn.close()


def test_rejected_source_rejects_the_whole_request(tmp_path):
    """A valid report + one bad source must write NOTHING — validation is
    all-files-first, then rows (v1 stranded queue rows on partial failures)."""
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        files = [
            ("report", ("report.pdf", PDF_BYTES, "application/pdf")),
            ("sources", ("good.pdf", PDF_BYTES, "application/pdf")),
            ("sources", ("bad.exe", PDF_BYTES, "application/pdf")),
        ]
        assert client.post("/api/runs", headers=AUTH, files=files).status_code == 400

    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    assert conn.execute("SELECT count(*) FROM uploads").fetchone()[0] == 0
    conn.close()


def test_unknown_ids_404(client):
    assert client.get("/api/runs/nope", headers=AUTH).status_code == 404
    assert client.get("/api/jobs/nope", headers=AUTH).status_code == 404
    assert client.get("/api/runs/nope/report", headers=AUTH).status_code == 404
    assert client.post("/api/runs/nope/retry", headers=AUTH).status_code == 404


def test_retry_requeues_only_failed_runs(client):
    conn = dbmod.connect(client.app.state.settings.db_path, DIM)
    run_id = dbmod.create_run(conn)
    job_id = dbmod.create_job(conn, run_id, {"report_upload_id": "r", "source_upload_ids": []})

    # QUEUED (not FAILED) — refused, nothing double-runs.
    assert client.post(f"/api/runs/{run_id}/retry", headers=AUTH).status_code == 409

    dbmod.finish_job_and_run(conn, job_id, run_id, "FAILED", error="Connection error.")
    response = client.post(f"/api/runs/{run_id}/retry", headers=AUTH)
    assert response.status_code == 202
    assert response.json() == {"run_id": run_id, "job_id": job_id, "status": "QUEUED"}
    assert dbmod.get_job(conn, job_id)["status"] == "QUEUED"
    assert dbmod.get_run(conn, run_id)["status"] == "RUNNING"
    assert dbmod.get_run(conn, run_id)["error"] is None

    # Already requeued — a second retry is refused too.
    assert client.post(f"/api/runs/{run_id}/retry", headers=AUTH).status_code == 409
    conn.close()


def test_retry_race_loser_gets_409_not_500(client, monkeypatch):
    """Two overlapping retries both read the job as FAILED; the loser's
    guarded UPDATE matches nothing and raises ValueError — which must map to
    a 409 conflict, not a 500."""
    from authorai import api as apimod

    conn = dbmod.connect(client.app.state.settings.db_path, DIM)
    run_id = dbmod.create_run(conn)
    job_id = dbmod.create_job(conn, run_id, {"report_upload_id": "r", "source_upload_ids": []})
    dbmod.finish_job_and_run(conn, job_id, run_id, "FAILED", error="boom")

    def racing_requeue(*args, **kwargs):
        raise ValueError(f"Job {job_id!r} is not FAILED — nothing to retry")

    monkeypatch.setattr(apimod.dbmod, "requeue_job", racing_requeue)
    response = client.post(f"/api/runs/{run_id}/retry", headers=AUTH)
    assert response.status_code == 409
    conn.close()


def _seed_scored_run(settings) -> str:
    """A run with one supported + one unverifiable verdict and stored scores.

    Both documents are backed by real uploaded PDF files on disk so the
    document-file endpoint can stream them.
    """
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    run_id = dbmod.create_run(conn)
    report_pdf = settings.uploads_dir / f"{dbmod.new_id()}.pdf"
    report_pdf.write_bytes(PDF_BYTES)
    source_pdf = settings.uploads_dir / f"{dbmod.new_id()}.pdf"
    source_pdf.write_bytes(PDF_BYTES)
    report_upload = dbmod.add_upload(conn, "REPORT", "report.pdf", str(report_pdf))
    source_upload = dbmod.add_upload(conn, "SOURCE", "the-source.pdf", str(source_pdf))
    report = dbmod.add_document(conn, run_id, "REPORT", upload_id=report_upload)
    source = dbmod.add_document(
        conn, run_id, "SOURCE", upload_id=source_upload, title="The Source Report"
    )
    from authorai.embeddings import FakeEmbedder

    embedder = FakeEmbedder(dim=DIM)
    [chunk_id] = dbmod.add_chunks(
        conn, run_id, source, [{"text": "hunger fell", "page": 3}], embedder.embed(["hunger fell"])
    )
    [claim_a, claim_b] = dbmod.add_claims(
        conn,
        run_id,
        report,
        [{"text": "hunger fell", "page": 1}, {"text": "made up", "page": 2}],
    )
    dbmod.add_verdicts(
        conn,
        run_id,
        [
            {
                "claim_id": claim_a,
                "verdict": "SUPPORTED",
                "raw_verdict": "SUPPORTED",
                "quote": "hunger fell",
                "quote_verified": 1,
                "quoted_chunk_id": chunk_id,
                "rationale": "verbatim",
                "model": "fake",
            },
            {
                "claim_id": claim_b,
                "verdict": "UNVERIFIABLE",
                "raw_verdict": "SUPPORTED",  # downgraded by the quote check
                "quote": "nowhere",
                "quote_verified": 0,
                "quoted_chunk_id": None,
                "rationale": "quote failed verification",
                "model": "fake",
            },
        ],
    )
    dbmod.save_run_scores(
        conn,
        run_id,
        accuracy={
            "supported": 1,
            "contradicted": 0,
            "unverifiable": 1,
            "total": 2,
            "correct": 1,
            "incorrect": 0,
            "disavowed": 0,
            "accuracy": 1.0,
            "coverage": 0.5,
        },
        credibility={
            "score": 62.5,
            "method": "usage_weighted_mean",
            "sources": [{"doc_id": source, "total": 62.5, "tier": "VERIFIED_DOI", "usage": 1}],
        },
        validity={
            "score": 71.0,
            "components": {
                "coverage": {
                    "score": 70,
                    "justification": "treats its stated scope",
                    "quote": "hunger fell",
                    "quote_verified": 1,
                }
            },
            "weights_used": {"coverage": 1.0},
        },
    )
    dbmod.save_source_credibility(
        conn,
        run_id,
        [
            {
                "doc_id": source,
                "metadata": {"title": "The Source Report"},
                "components": {"authority": 30.0},
                "total": 62.5,
                "tier": "VERIFIED_DOI",
            }
        ],
    )
    conn.close()
    return run_id


def test_report_shape_scored_run(tmp_path):
    settings = _settings(tmp_path)
    run_id = _seed_scored_run(settings)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        report = client.get(f"/api/runs/{run_id}/report", headers=AUTH).json()

    # All scores leave the API as 0–1 fractions.
    assert report["scores"] == {
        "accuracy": 1.0,
        "coverage": 0.5,
        "credibility": 0.625,
        "validity": 0.71,
    }
    assert report["stats"] == {
        "claims_total": 2,
        "claims_supported": 1,
        "claims_contradicted": 0,
        "claims_unverifiable": 1,
    }

    supported, downgraded = report["claims"]
    assert supported["verdict"] == "SUPPORTED"
    assert supported["downgraded"] is False
    # The quoted chunk resolves to its source document and page.
    assert supported["evidence_source"]["title"] == "The Source Report"
    assert supported["evidence_source"]["page"] == 3
    assert downgraded["verdict"] == "UNVERIFIABLE"
    assert downgraded["downgraded"] is True
    assert downgraded["evidence_source"] is None

    [source] = report["sources"]
    assert source["tier"] == "VERIFIED_DOI"
    assert source["total"] == 62.5

    # The report exposes its own REPORT doc id for the report-PDF pane.
    assert report["report_doc_id"]


def test_document_file_streams_the_pdf(tmp_path):
    settings = _settings(tmp_path)
    run_id = _seed_scored_run(settings)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        report = client.get(f"/api/runs/{run_id}/report", headers=AUTH).json()
        report_doc = report["report_doc_id"]
        source_doc = report["sources"][0]["doc_id"]
        for doc_id in (report_doc, source_doc):
            resp = client.get(f"/api/runs/{run_id}/documents/{doc_id}/file", headers=AUTH)
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "application/pdf"
            assert resp.content.startswith(b"%PDF-")


def test_document_file_cross_run_is_404(tmp_path):
    """A doc id from another run must not resolve — the run_id is the boundary."""
    settings = _settings(tmp_path)
    run_a = _seed_scored_run(settings)
    run_b = _seed_scored_run(settings)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        report_a = client.get(f"/api/runs/{run_a}/report", headers=AUTH).json()
        doc_a = report_a["report_doc_id"]
        # Ask for run A's doc under run B's id.
        resp = client.get(f"/api/runs/{run_b}/documents/{doc_a}/file", headers=AUTH)
        assert resp.status_code == 404


def test_document_file_unknown_doc_is_404(client):
    # /api/runs/x/... — the run itself doesn't exist either.
    assert client.get("/api/runs/x/documents/y/file", headers=AUTH).status_code == 404


def test_report_before_scoring_returns_null_scores(tmp_path):
    settings = _settings(tmp_path)
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    run_id = dbmod.create_run(conn)
    conn.close()

    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        report = client.get(f"/api/runs/{run_id}/report", headers=AUTH).json()
    assert report["scores"] is None
    assert report["stats"]["claims_total"] == 0
    assert report["claims"] == []


def test_chat_unknown_run_404(client):
    assert (
        client.post("/api/runs/nope/chat", headers=AUTH, json={"question": "hi"}).status_code == 404
    )


def test_chat_rejects_a_run_that_is_not_done(tmp_path):
    """The seeded run is CREATED (not scored); chat must 409, not answer."""
    settings = _settings(tmp_path)
    run_id = _seed_scored_run(settings)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(f"/api/runs/{run_id}/chat", headers=AUTH, json={"question": "hi"})
    assert resp.status_code == 409


def test_chat_rejects_an_oversized_question(tmp_path):
    """A chat body must not be a memory-amplification vector — an overlong
    question is a 422, not forwarded whole to the paid model."""
    settings = _settings(tmp_path)
    run_id = _seed_scored_run(settings)
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    dbmod.set_run_status(conn, run_id, "DONE")
    conn.close()
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(f"/api/runs/{run_id}/chat", headers=AUTH, json={"question": "x" * 5000})
    assert resp.status_code == 422


def test_chat_answers_a_done_run(tmp_path, monkeypatch):
    from authorai import api as apimod
    from tests.conftest import FakeLLM

    settings = _settings(tmp_path)
    run_id = _seed_scored_run(settings)
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    dbmod.set_run_status(conn, run_id, "DONE")
    conn.close()

    fake = FakeLLM(chat_answer="One claim is unverifiable.")
    monkeypatch.setattr(apimod, "AnthropicClient", lambda key: fake)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(
            f"/api/runs/{run_id}/chat",
            headers=AUTH,
            json={"question": "summary?", "mode": "evidence"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"answer": "One claim is unverifiable.", "mode": "evidence"}
    # The endpoint actually built the context and called chat with the cache block.
    assert fake.chat_calls[0]["system_blocks"][0]["cache_control"] == {"type": "ephemeral"}


def test_report_sources_list_every_source_document_with_type_and_scorability(tmp_path):
    settings = _settings(tmp_path)
    run_id = _seed_scored_run(settings)
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    image_upload = dbmod.add_upload(
        conn, "SOURCE", "chart.png", str(settings.uploads_dir / "c.png"), source_type="image"
    )
    image_doc = dbmod.add_document(conn, run_id, "SOURCE", upload_id=image_upload, title="chart")
    web_upload = dbmod.add_upload(
        conn,
        "SOURCE",
        "who.int",
        str(settings.uploads_dir / "w.json"),
        source_type="web",
        url="https://www.who.int/facts",
    )
    web_doc = dbmod.add_document(
        conn,
        run_id,
        "SOURCE",
        upload_id=web_upload,
        title="Drinking-water",
        metadata=json.dumps(
            {
                "sections": [],
                "provenance": {
                    "url": "https://www.who.int/facts",
                    "truncated": {"kept_chars": 200_000, "dropped_chars": 51_234},
                },
            }
        ),
    )
    stored = conn.execute("SELECT credibility FROM run_scores WHERE run_id = ?", (run_id,))
    credibility = json.loads(stored.fetchone()["credibility"])
    credibility["excluded"] = [{"doc_id": image_doc, "reason": "image", "usage": 0}]
    with conn:
        conn.execute(
            "UPDATE run_scores SET credibility = ? WHERE run_id = ?",
            (json.dumps(credibility), run_id),
        )
    conn.close()

    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        report = client.get(f"/api/runs/{run_id}/report", headers=AUTH).json()

    by_id = {s["doc_id"]: s for s in report["sources"]}
    assert len(by_id) == 3  # every SOURCE document, scored or not
    pdf = next(s for s in report["sources"] if s["title"] == "The Source Report")
    assert (pdf["source_type"], pdf["url"], pdf["scorable"]) == ("pdf", None, True)
    assert pdf["total"] is not None
    assert report["sources"][0]["doc_id"] == pdf["doc_id"]  # scored first
    assert by_id[image_doc] == {
        "doc_id": image_doc,
        "title": "chart",
        "source_type": "image",
        "url": None,
        "scorable": False,
        "total": None,
        "tier": None,
        "components": None,
        "metadata": None,
        "truncated": None,
    }
    web = by_id[web_doc]
    assert (web["source_type"], web["url"], web["scorable"], web["total"]) == (
        "web",
        "https://www.who.int/facts",
        True,
        None,
    )
    # A page read only in part says so where the user reads its standing, not
    # only in the server log.
    assert web["truncated"] == {"kept_chars": 200_000, "dropped_chars": 51_234}
    assert pdf["truncated"] is None  # a PDF is never capped
    assert report["credibility_detail"]["excluded"] == credibility["excluded"]


def test_run_detail_uploads_carry_source_type_and_url(tmp_path):
    settings = _settings(tmp_path)
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    run_id, _ = dbmod.create_run_with_uploads_and_job(
        conn,
        [
            dbmod.UploadSpec(kind="REPORT", file_name="report.pdf", path=str(tmp_path / "r.pdf")),
            dbmod.UploadSpec(
                kind="SOURCE",
                file_name="www.who.int",
                path=str(tmp_path / "u.json"),
                source_type="web",
                url="https://www.who.int/facts",
            ),
        ],
    )
    conn.close()
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        uploads = client.get(f"/api/runs/{run_id}", headers=AUTH).json()["uploads"]
    keys = ("kind", "file_name", "source_type", "url")
    assert [{k: u[k] for k in keys} for u in uploads] == [
        {"kind": "REPORT", "file_name": "report.pdf", "source_type": "pdf", "url": None},
        {
            "kind": "SOURCE",
            "file_name": "www.who.int",
            "source_type": "web",
            "url": "https://www.who.int/facts",
        },
    ]


def _cite_chunk(conn, run_id, chunk_id, text):
    report_doc = dbmod.get_report_doc_id(conn, run_id)
    [claim] = dbmod.add_claims(conn, run_id, report_doc, [{"text": text, "page": 1}])
    dbmod.add_verdicts(
        conn,
        run_id,
        [
            {
                "claim_id": claim,
                "verdict": "SUPPORTED",
                "raw_verdict": "SUPPORTED",
                "quote": text,
                "quote_verified": 1,
                "quoted_chunk_id": chunk_id,
                "rationale": "r",
                "model": "fake",
            }
        ],
    )


def test_report_evidence_carries_the_locator_for_each_source_type(tmp_path):
    from authorai.embeddings import FakeEmbedder

    settings = _settings(tmp_path)
    run_id = _seed_scored_run(settings)
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    embedder = FakeEmbedder(dim=DIM)

    def source(source_type, title, url, chunk):
        upload = dbmod.add_upload(
            conn,
            "SOURCE",
            title,
            str(settings.uploads_dir / f"{title}.json"),
            source_type=source_type,
            url=url,
        )
        doc = dbmod.add_document(conn, run_id, "SOURCE", upload_id=upload, title=title)
        [chunk_id] = dbmod.add_chunks(conn, run_id, doc, [chunk], embedder.embed([chunk["text"]]))
        _cite_chunk(conn, run_id, chunk_id, chunk["text"])
        return doc, chunk_id

    web_doc, web_chunk = source(
        "web",
        "Drinking-water",
        "https://www.who.int/facts",
        {"text": "73 percent safely managed", "section": "Access to services"},
    )
    video_url = "https://www.youtube.com/watch?v=abc123def45"
    video_doc, video_chunk = source(
        "youtube",
        "Water talk",
        video_url,
        {"text": "two billion lack water", "start_seconds": 754.0, "end_seconds": 829.0},
    )
    conn.close()

    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        claims = client.get(f"/api/runs/{run_id}/report", headers=AUTH).json()["claims"]
    evidence = {c["text"]: c["evidence_source"] for c in claims if c["evidence_source"]}
    assert evidence["73 percent safely managed"] == {
        "doc_id": web_doc,
        "title": "Drinking-water",
        "page": None,
        "source_type": "web",
        "url": "https://www.who.int/facts",
        "section": "Access to services",
        "start_seconds": None,
        "chunk_id": web_chunk,
    }
    assert evidence["two billion lack water"] == {
        "doc_id": video_doc,
        "title": "Water talk",
        "page": None,
        "source_type": "youtube",
        "url": video_url,
        "section": None,
        "start_seconds": 754.0,
        "chunk_id": video_chunk,
    }
    pdf = evidence["hunger fell"]
    assert (pdf["source_type"], pdf["page"], pdf["url"], pdf["start_seconds"]) == (
        "pdf",
        3,
        None,
        None,
    )


def test_report_types_a_source_document_without_an_upload_row_as_pdf(tmp_path):
    """CLI and hand-seeded runs can hold SOURCE documents with no upload row.
    SourceType is never null in the frontend contract, so both the cited evidence
    and the sources list say "pdf" for such a document — a null evidence type
    would send the viewer to 'no preview'."""
    from authorai.embeddings import FakeEmbedder

    settings = _settings(tmp_path)
    run_id = _seed_scored_run(settings)
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    legacy = dbmod.add_document(conn, run_id, "SOURCE", title="Legacy CLI Source")
    [chunk_id] = dbmod.add_chunks(
        conn,
        run_id,
        legacy,
        [{"text": "wheat yields rose", "page": 7}],
        FakeEmbedder(dim=DIM).embed(["wheat yields rose"]),
    )
    _cite_chunk(conn, run_id, chunk_id, "wheat yields rose")
    conn.close()

    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        report = client.get(f"/api/runs/{run_id}/report", headers=AUTH).json()

    [claim] = [c for c in report["claims"] if c["text"] == "wheat yields rose"]
    evidence = claim["evidence_source"]
    assert (evidence["doc_id"], evidence["source_type"], evidence["page"], evidence["url"]) == (
        legacy,
        "pdf",
        7,
        None,
    )
    [source] = [s for s in report["sources"] if s["doc_id"] == legacy]
    assert (source["source_type"], source["url"], source["scorable"]) == ("pdf", None, True)


def test_document_file_serves_each_source_type_with_its_media_type(tmp_path):
    settings = _settings(tmp_path)
    run_id = _seed_scored_run(settings)
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    snapshot = settings.uploads_dir / f"{dbmod.new_id()}.json"
    snapshot.write_text('{"schema": 1}', encoding="utf-8")
    png = settings.uploads_dir / f"{dbmod.new_id()}.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    jpg = settings.uploads_dir / f"{dbmod.new_id()}.jpg"
    jpg.write_bytes(b"\xff\xd8\xff" + b"0" * 16)
    cases = [
        ("web", snapshot, "web", "https://www.who.int/facts", "application/json"),
        ("video", snapshot, "youtube", "https://youtu.be/abc123def45", "application/json"),
        ("png", png, "image", None, "image/png"),
        ("jpg", jpg, "image", None, "image/jpeg"),
    ]
    docs = {}
    for name, path, source_type, url, _media in cases:
        upload = dbmod.add_upload(conn, "SOURCE", name, str(path), source_type=source_type, url=url)
        docs[name] = dbmod.add_document(conn, run_id, "SOURCE", upload_id=upload, title=name)
    conn.close()
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        for name, _path, _type, _url, media in cases:
            resp = client.get(f"/api/runs/{run_id}/documents/{docs[name]}/file", headers=AUTH)
            assert resp.status_code == 200, name
            assert resp.headers["content-type"].split(";")[0] == media, name


def test_link_only_run_records_planned_page_uploads(tmp_path):
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(
            "/api/runs",
            headers=AUTH,
            files=[("report", ("report.pdf", PDF_BYTES, "application/pdf"))],
            data={"source_urls": ["  https://www.who.int/facts#top  "]},
        )
        assert resp.status_code == 202, resp.text
        uploads = client.get(f"/api/runs/{resp.json()['run_id']}", headers=AUTH).json()["uploads"]
    assert [(u["kind"], u["source_type"], u["url"], u["file_name"]) for u in uploads] == [
        ("REPORT", "pdf", None, "report.pdf"),
        ("SOURCE", "web", "https://www.who.int/facts", "https://www.who.int/facts"),
    ]
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    row = conn.execute(
        "SELECT path, content_hash FROM uploads WHERE source_type = 'web'"
    ).fetchone()
    conn.close()
    planned = Path(row["path"])
    assert planned.suffix == ".json"
    assert planned.parent.resolve() == settings.uploads_dir.resolve()
    assert not planned.exists()  # the ingest step fetches it, not the upload request
    assert row["content_hash"] is None


def test_files_and_links_are_recorded_files_first(tmp_path):
    settings = _settings(tmp_path)
    links = ["https://example.org/a", "https://example.org/b"]
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(
            "/api/runs",
            headers=AUTH,
            files=_upload_files(source_count=1),
            data={"source_urls": links},
        )
        assert resp.status_code == 202, resp.text
        uploads = client.get(f"/api/runs/{resp.json()['run_id']}", headers=AUTH).json()["uploads"]
    assert [(u["kind"], u["source_type"], u["url"]) for u in uploads] == [
        ("REPORT", "pdf", None),
        ("SOURCE", "pdf", None),
        ("SOURCE", "web", links[0]),
        ("SOURCE", "web", links[1]),
    ]


@pytest.mark.parametrize(
    ("links", "detail"),
    [
        (["ftp://example.org/file"], "not a usable link"),
        (["not a url"], "not a usable link"),
        ([""], "not a usable link"),
        (["https://user:pass@example.org/a"], "not a usable link"),
        (["https://www.youtube.com/watch?v=abc123def45"], "YouTube links are not supported yet"),
        (["https://youtu.be/abc123def45"], "YouTube links are not supported yet"),
        (["https://example.org/a", "https://example.org/a#section"], "added twice"),
    ],
)
def test_bad_links_reject_the_whole_request(tmp_path, links, detail):
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(
            "/api/runs",
            headers=AUTH,
            files=_upload_files(source_count=1),
            data={"source_urls": links},
        )
    assert resp.status_code == 400, resp.text
    assert detail in resp.json()["detail"]
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM uploads").fetchone()[0] == 0
    conn.close()
    assert not settings.uploads_dir.exists() or not any(settings.uploads_dir.iterdir())


def test_a_run_needs_at_least_one_source(tmp_path):
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(
            "/api/runs",
            headers=AUTH,
            files=[("report", ("report.pdf", PDF_BYTES, "application/pdf"))],
        )
    assert resp.status_code == 400, resp.text
    assert "at least one source" in resp.json()["detail"]


def test_links_count_toward_the_source_cap(tmp_path):
    settings = _settings(tmp_path, max_source_files=2)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(
            "/api/runs",
            headers=AUTH,
            files=_upload_files(source_count=1),
            data={"source_urls": ["https://example.org/a", "https://example.org/b"]},
        )
    assert resp.status_code == 400, resp.text
    assert "too many sources" in resp.json()["detail"]


# --- POST /api/references/scan ---------------------------------------------------

SCAN = "/api/references/scan"
CITED = ReferenceList(
    references=[
        Reference(entry="Open, A. (2021). Free paper. J. 1.", doi="10.1000/open", year=2021),
        Reference(entry="Closed, B. (2019). Paid paper. J. 2.", doi="10.1000/closed", year=2019),
        Reference(entry="Web, C. (2020). A page. https://x.org/page", url="https://x.org/page#top"),
        Reference(entry="Plain, D. (2018). A book.", title="A book", authors=["Plain, D."]),
    ]
)


def _scan_report(pages=("Body.", "References\nOpen, A. (2021). Free paper. J. 1.")):
    return {"report": ("report.pdf", pdf_with_pages(list(pages)), "application/pdf")}


def _no_llm(monkeypatch) -> None:
    """The scan under test never reaches the model: constructing a client
    is already the failure."""
    from authorai import api as apimod

    monkeypatch.setattr(
        apimod, "AnthropicClient", lambda key: pytest.fail("constructed an LLM client")
    )


def _nothing_written(settings) -> None:
    """The scan is structurally unable to write rows (no DB dependency) and
    must leave nothing on disk either — it reads the spooled part in place."""
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    for table in ("runs", "uploads", "jobs"):
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0, table
    conn.close()
    assert not settings.uploads_dir.exists() or not any(settings.uploads_dir.iterdir())


@respx.mock
def test_reference_scan_lists_cited_works_and_writes_nothing(tmp_path, monkeypatch):
    from authorai import api as apimod

    respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/open").mock(
        return_value=httpx.Response(
            200, json={"is_oa": True, "best_oa_location": {"url_for_pdf": "https://x.org/o.pdf"}}
        )
    )
    respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/closed").mock(
        return_value=httpx.Response(200, json={"is_oa": False, "best_oa_location": None})
    )
    settings = _settings(tmp_path, crossref_mailto="checker@example.org")
    fake = FakeLLM({ReferenceList: CITED})
    monkeypatch.setattr(apimod, "AnthropicClient", lambda key: fake)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(SCAN, headers=AUTH, files=_scan_report())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["text_source"] == "heading"
    assert body["lookup"] == {"status": "ok", "detail": None}
    assert body["limits"] == {
        "text_truncated": False,
        "references_dropped": 0,
        "possibly_incomplete": False,
    }
    assert [(r["retrievability"], r["suggested_url"]) for r in body["references"]] == [
        ("pdf", "https://x.org/o.pdf"),
        ("paywalled", None),
        ("unknown", "https://x.org/page"),  # printed in the entry, not checked
        ("unknown", None),
    ]
    assert body["references"][0] == {
        "title": None,
        "authors": [],
        "year": 2021,
        "doi": "10.1000/open",
        "url": None,
        "entry": "Open, A. (2021). Free paper. J. 1.",
        "retrievability": "pdf",
        "suggested_url": "https://x.org/o.pdf",
    }
    # The model saw the closing text under the references contract, on the
    # configured model.
    call = fake.parse_calls[0]
    assert call["model"] == settings.references_model
    assert "Open, A. (2021)" in call["prompt"]
    assert call["output_type"] is ReferenceList
    _nothing_written(settings)


@pytest.mark.parametrize(
    ("file_name", "content", "expected_status", "expected_detail"),
    [
        ("report.txt", b"%PDF-1.4 x", 400, "'report.txt' is not a .pdf file"),
        ("report.pdf", b"not a pdf at all", 400, "'report.pdf' is not PDF content"),
        ("report.pdf", b"%PDF-1.4 fake pdf content", 400, "'report.pdf': could not read"),
        ("report.pdf", b"%PDF-" + b"x" * 2000, 413, "exceeds the 1000 byte per-file limit"),
    ],
)
def test_reference_scan_rejects_what_it_cannot_read(
    tmp_path, monkeypatch, file_name, content, expected_status, expected_detail
):
    """The same validation an upload gets, plus pypdf's verdict: a file that
    passes the magic check but cannot be opened is a 400 naming the file,
    never a 500 — and no model is constructed for any of them."""
    _no_llm(monkeypatch)
    settings = _settings(tmp_path, max_upload_bytes=1000)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(
            SCAN, headers=AUTH, files={"report": (file_name, content, "application/pdf")}
        )
    assert resp.status_code == expected_status, resp.text
    assert expected_detail in resp.json()["detail"]
    _nothing_written(settings)


def test_reference_scan_of_a_report_too_costly_to_read_is_a_400_not_a_500(tmp_path, monkeypatch):
    """The PDF is read in a bounded child (references.read_pages); a file
    that holds the reader past Settings.extract_timeout_seconds is refused
    like any file pypdf cannot open — a 400 naming the file and saying it
    was too costly, never a 500 — and no model is constructed."""
    from authorai import references as refsmod

    monkeypatch.setattr(refsmod, "_read_in_child", reader_that_never_answers)
    _no_llm(monkeypatch)
    settings = _settings(tmp_path, extract_timeout_seconds=0.5)
    app = create_app(settings, worker=_NoopWorker())
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(SCAN, headers=AUTH, files=_scan_report())
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"].startswith(
        "'report.pdf': could not read the PDF: it was too costly to read"
    )
    _nothing_written(settings)


def test_reference_scan_reads_the_last_pages_of_a_long_report_promptly(tmp_path, monkeypatch):
    """5,000 textless pages: the page-tree bound is linear in the tree, the
    read covers the last 600 pages, and the answer is 'none' without a
    model call — in seconds, not the minutes a hostile tree would take."""
    _no_llm(monkeypatch)
    settings = _settings(tmp_path)
    started = time.perf_counter()
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(SCAN, headers=AUTH, files=_scan_report(pages=[""] * 5_000))
    assert resp.status_code == 200, resp.text
    assert resp.json()["text_source"] == "none"
    assert time.perf_counter() - started < 20
    _nothing_written(settings)


def test_reference_scan_reports_a_reference_list_cut_by_the_text_cap(tmp_path, monkeypatch):
    """A bibliography longer than REFERENCE_MAX_CHARS is read in part; the
    response says so (`limits.text_truncated`), so the dialog does not
    present the works it never read as absent from the user's sources."""
    from authorai import api as apimod
    from authorai.references import REFERENCE_MAX_CHARS

    fake = FakeLLM({ReferenceList: CITED})
    monkeypatch.setattr(apimod, "AnthropicClient", lambda key: fake)
    settings = _settings(tmp_path)
    # Prose-shaped lines: the test is about the character cap, and a list of
    # two thousand entry-shaped lines answered with four references would
    # rightly trip the completeness guard (which has its own test above).
    lines = ("A line of closing text, no entry.\n" * (REFERENCE_MAX_CHARS // 34 + 100)).rstrip()
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(SCAN, headers=AUTH, files=_scan_report(pages=["References\n" + lines]))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["text_source"] == "heading"
    assert body["limits"] == {
        "text_truncated": True,
        "references_dropped": 0,
        "possibly_incomplete": False,
    }
    assert all(len(call["prompt"]) < REFERENCE_MAX_CHARS for call in fake.parse_calls)


@respx.mock
def test_reference_scan_reports_an_answer_that_still_looks_incomplete(
    tmp_path, monkeypatch, references_log
):
    """The live '1 reference' shape, end to end: a page with four lines
    shaped like an entry's start and a model answering one entry twice.
    The part is asked once more (two calls), the answer kept, and the
    dialog is told the scan may have missed entries — the flag in
    `limits`, next to the caps, with the warning in the log."""
    from authorai import api as apimod

    cited = [("Open", 2021), ("Closed", 2019), ("Web", 2020), ("Plain", 2018)]
    page = "References\n" + "\n".join(
        f"{name}, A. ({year}). A paper. J. {i}." for i, (name, year) in enumerate(cited)
    )
    one = ReferenceList(references=[Reference(entry="Open, A. (2021). A paper. J. 0.")])
    settings = _settings(tmp_path, crossref_mailto="checker@example.org")
    fake = FakeLLM({ReferenceList: [one, one]})
    monkeypatch.setattr(apimod, "AnthropicClient", lambda key: fake)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(SCAN, headers=AUTH, files=_scan_report(pages=("Body.", page)))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["limits"] == {
        "text_truncated": False,
        "references_dropped": 0,
        "possibly_incomplete": True,
    }
    assert [r["entry"] for r in body["references"]] == ["Open, A. (2021). A paper. J. 0."]
    assert len(fake.parse_calls) == 2
    assert "1 references for about 4 entry starts" in references_log.text
    assert "still looks incomplete after a retry" in references_log.text
    _nothing_written(settings)


def test_reference_scan_of_a_textless_pdf_makes_no_model_call(tmp_path, monkeypatch):
    """A scanned (image-only) PDF has no text to read: the answer is 'none'
    and an empty list — the model is never asked, so it cannot invent a
    bibliography."""
    _no_llm(monkeypatch)
    settings = _settings(tmp_path, crossref_mailto="checker@example.org")
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(SCAN, headers=AUTH, files=_scan_report(pages=["", ""]))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "text_source": "none",
        "lookup": {"status": "ok", "detail": None},
        "limits": {"text_truncated": False, "references_dropped": 0, "possibly_incomplete": False},
        "references": [],
    }
    _nothing_written(settings)


@respx.mock
def test_reference_scan_without_a_contact_email_looks_nothing_up(tmp_path, monkeypatch):
    """No route is mocked: any Unpaywall request would make respx raise. The
    list is still returned, flagged 'unconfigured', every DOI 'unknown' — and
    a printed address is still offered, since it needs no lookup."""
    from authorai import api as apimod

    # Explicit, not "unset": a backend/.env that names a contact email (the
    # dev setup, the live scratch server's symlink) would otherwise supply
    # one, and this test would depend on the machine it runs on.
    settings = _settings(tmp_path, crossref_mailto=None)
    assert settings.crossref_mailto is None
    fake = FakeLLM({ReferenceList: CITED})
    monkeypatch.setattr(apimod, "AnthropicClient", lambda key: fake)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(SCAN, headers=AUTH, files=_scan_report())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["lookup"] == {"status": "unconfigured", "detail": None}
    assert [(r["retrievability"], r["suggested_url"]) for r in body["references"]] == [
        ("unknown", None),
        ("unknown", None),
        ("unknown", "https://x.org/page"),
        ("unknown", None),
    ]


def test_reference_scan_survives_a_blank_entry(tmp_path, monkeypatch):
    """A model slip — `entry` "" or whitespace where the prompt asked for
    the entry's first characters — is an ordinary answer, not a 500: the
    handler re-validates each reference as a ScannedReference, and a blank
    entry must arrive there as "" (a str), never as None. The row with a
    title is listed with entry "", the row with nothing to act on is
    dropped. raise_server_exceptions=False so a handler failure shows as
    the HTTP 500 the dialog would see."""
    from authorai import api as apimod

    settings = _settings(tmp_path)  # crossref_mailto unset: no lookup, no network
    answer = ReferenceList(
        references=[
            Reference(entry=""),
            Reference(entry="   ", title="Titled work", authors=["Titled, T."], year=2020),
            Reference(entry="Printed, P. (2019). As printed."),
        ]
    )
    fake = FakeLLM({ReferenceList: answer})
    monkeypatch.setattr(apimod, "AnthropicClient", lambda key: fake)
    app = create_app(settings, worker=_NoopWorker())
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(SCAN, headers=AUTH, files=_scan_report())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [(r["entry"], r["title"]) for r in body["references"]] == [
        ("", "Titled work"),
        ("Printed, P. (2019). As printed.", None),
    ]
    assert body["limits"] == {
        "text_truncated": False,
        "references_dropped": 1,
        "possibly_incomplete": False,
    }
    assert body["references"][0] == {
        "title": "Titled work",
        "authors": ["Titled, T."],
        "year": 2020,
        "doi": None,
        "url": None,
        "entry": "",
        "retrievability": "unknown",
        "suggested_url": None,
    }
    _nothing_written(settings)


@respx.mock
def test_reference_scan_survives_an_unpaywall_outage(tmp_path, monkeypatch):
    """A registry outage is not a scan failure: the citation list is still
    useful, so it comes back 200 with an explicit 'unavailable' flag and the
    reason — never a silent 'unknown', never a 500."""
    from authorai import api as apimod

    monkeypatch.setattr("authorai.credibility.time.sleep", lambda seconds: None)
    respx.get(url__regex=rf"{UNPAYWALL_BASE}/v2/.*").mock(return_value=httpx.Response(503))
    settings = _settings(tmp_path, crossref_mailto="checker@example.org")
    fake = FakeLLM({ReferenceList: CITED})
    monkeypatch.setattr(apimod, "AnthropicClient", lambda key: fake)
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        resp = client.post(SCAN, headers=AUTH, files=_scan_report())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["lookup"]["status"] == "unavailable"
    assert "HTTP 503" in body["lookup"]["detail"]
    assert len(body["references"]) == len(CITED.references)
    assert all(r["retrievability"] == "unknown" for r in body["references"])
    assert body["references"][2]["suggested_url"] == "https://x.org/page"
    _nothing_written(settings)


@respx.mock
def test_reference_scan_survives_a_200_whose_body_is_not_json(tmp_path, monkeypatch):
    """A captive portal or a CDN error page answers 200 with HTML. That is a
    registry malfunction like the 200-non-object case, and the contract is
    the same: 200 with lookup.status "unavailable" and the cause in
    `detail` — never a 500 from a JSON decode error escaping the lookup."""
    from authorai import api as apimod

    monkeypatch.setattr("authorai.credibility.time.sleep", lambda seconds: None)
    respx.get(url__regex=rf"{UNPAYWALL_BASE}/v2/.*").mock(
        return_value=httpx.Response(
            200,
            text="<html><body>Service degraded</body></html>",
            headers={"content-type": "text/html"},
        )
    )
    settings = _settings(tmp_path, crossref_mailto="checker@example.org")
    monkeypatch.setattr(apimod, "AnthropicClient", lambda key: FakeLLM({ReferenceList: CITED}))
    with TestClient(
        create_app(settings, worker=_NoopWorker()), raise_server_exceptions=False
    ) as client:
        resp = client.post(SCAN, headers=AUTH, files=_scan_report())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["lookup"]["status"] == "unavailable"
    assert "not JSON" in body["lookup"]["detail"]
    assert "Unpaywall" in body["lookup"]["detail"]
    assert len(body["references"]) == len(CITED.references)
    _nothing_written(settings)


def test_reference_scan_needs_the_report_part(client):
    assert client.post(SCAN, headers=AUTH).status_code == 422


class _HeldLLM(FakeLLM):
    """A FakeLLM whose parse holds until released, counting the scans that
    reached it — so a test can keep SCAN_CONCURRENCY scans in flight."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reached = threading.Semaphore(0)
        self.release = threading.Event()

    def parse(self, **kwargs):
        self.reached.release()
        assert self.release.wait(20), "the held scan was never released"
        return super().parse(**kwargs)


def _post_in_threads(client, count: int) -> tuple[list, list[threading.Thread]]:
    results: list = []
    threads = [
        threading.Thread(
            target=lambda: results.append(client.post(SCAN, headers=AUTH, files=_scan_report()))
        )
        for _ in range(count)
    ]
    for thread in threads:
        thread.start()
    return results, threads


def test_scans_past_the_concurrency_bound_are_refused_until_one_finishes(tmp_path, monkeypatch):
    """SCAN_CONCURRENCY scans run at once; the next is a 429 with a plain
    message, before any reader process or model call of its own; once a
    scan finishes, a new one is a 200. A scan the dialog abandons runs on
    (a threadpool thread cannot be cancelled), each holding a reader child
    and the model calls, so overlapping picks queue on the client instead
    of stacking on the server."""
    from authorai import api as apimod
    from authorai.api import SCAN_CONCURRENCY

    fake = _HeldLLM({ReferenceList: CITED})
    monkeypatch.setattr(apimod, "AnthropicClient", lambda key: fake)
    settings = _settings(tmp_path)  # no contact email: no lookups
    with TestClient(create_app(settings, worker=_NoopWorker())) as client:
        results, threads = _post_in_threads(client, SCAN_CONCURRENCY)
        try:
            for _ in range(SCAN_CONCURRENCY):
                assert fake.reached.acquire(timeout=20), "a scan never reached the model"
            refused = client.post(SCAN, headers=AUTH, files=_scan_report())
        finally:
            fake.release.set()
            for thread in threads:
                thread.join(20)
        assert refused.status_code == 429, refused.text
        assert refused.json() == {
            "detail": "a reference scan is already running — try again in a moment"
        }
        assert [resp.status_code for resp in results] == [200] * SCAN_CONCURRENCY
        assert len(fake.parse_calls) == SCAN_CONCURRENCY  # the refused scan asked nothing
        again = client.post(SCAN, headers=AUTH, files=_scan_report())
        assert again.status_code == 200, again.text
        assert len(fake.parse_calls) == SCAN_CONCURRENCY + 1
    _nothing_written(settings)


def test_a_scan_that_fails_frees_its_slot(tmp_path, monkeypatch):
    """A slot is held for the scan's whole run and released on every exit:
    after a 400 (a read too costly) and a 500 (a model failure), a full
    set of SCAN_CONCURRENCY scans still runs at once."""
    from authorai import api as apimod
    from authorai.api import SCAN_CONCURRENCY

    class FailingLLM(FakeLLM):
        def parse(self, **kwargs):
            raise RuntimeError("the model is down")

    settings = _settings(tmp_path)
    app = create_app(settings, worker=_NoopWorker())
    with TestClient(app, raise_server_exceptions=False) as client:
        # Past the file checks (the magic bytes are there), refused by pypdf.
        unreadable = {"report": ("report.pdf", b"%PDF-1.4 fake pdf content", "application/pdf")}
        resp = client.post(SCAN, headers=AUTH, files=unreadable)
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"].startswith("'report.pdf': could not read the PDF")
        monkeypatch.setattr(apimod, "AnthropicClient", lambda key: FailingLLM())
        assert client.post(SCAN, headers=AUTH, files=_scan_report()).status_code == 500

        fake = _HeldLLM({ReferenceList: CITED})
        monkeypatch.setattr(apimod, "AnthropicClient", lambda key: fake)
        results, threads = _post_in_threads(client, SCAN_CONCURRENCY)
        try:
            for _ in range(SCAN_CONCURRENCY):
                assert fake.reached.acquire(timeout=20), "a slot was not freed"
        finally:
            fake.release.set()
            for thread in threads:
                thread.join(20)
        assert [resp.status_code for resp in results] == [200] * SCAN_CONCURRENCY
