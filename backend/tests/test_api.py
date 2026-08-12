"""API tests: fail-closed startup, structural auth, upload validation, report shape.

TestClient is always used as a context manager — otherwise the lifespan
(where fail-closed and recovery live) never runs and the tests would pass
against an app that can't actually start.
"""

import pytest
from fastapi.testclient import TestClient

from authorai import db as dbmod
from authorai.config import Settings
from authorai.jobs import Worker
from authorai.main import create_app
from tests.conftest import DIM

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
    """Structural regression test for v1's per-route auth: EVERY route under
    /api must 401 without a key and with a wrong key — a new endpoint that
    forgets auth fails here automatically. Routes are enumerated from the
    OpenAPI schema, which lists everything the app actually serves."""
    app = create_app(_settings(tmp_path), worker=_NoopWorker())
    api_paths = {
        path: ops for path, ops in app.openapi()["paths"].items() if path.startswith("/api")
    }
    assert len(api_paths) >= 4  # sanity: the router is actually mounted
    with TestClient(app) as client:
        for path, ops in api_paths.items():
            url = path.replace("{run_id}", "x").replace("{job_id}", "x")
            for method in ops:
                assert client.request(method, url).status_code == 401, f"{method} {url}"
                wrong = client.request(method, url, headers={"X-API-Key": "wrong"})
                assert wrong.status_code == 401, f"{method} {url} with wrong key"


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


def _seed_scored_run(settings) -> str:
    """A run with one supported + one unverifiable verdict and stored scores."""
    conn = dbmod.connect(settings.db_path, settings.embedding_dim)
    run_id = dbmod.create_run(conn)
    report = dbmod.add_document(conn, run_id, "REPORT")
    source = dbmod.add_document(conn, run_id, "SOURCE", title="The Source Report")
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
            "accuracy": 1.0,
            "coverage": 0.5,
        },
        credibility={"score": 62.5, "weighting": "usage"},
        validity={"score": 71.0, "components": {}},
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
