import sqlite3

import pytest

from authorai import db as dbmod
from authorai.embeddings import FakeEmbedder
from authorai.search import keyword_search
from tests.conftest import DIM

_EMBEDDER = FakeEmbedder(dim=DIM)


def _one_chunk(conn, run_id, doc_id, text):
    return dbmod.add_chunks(conn, run_id, doc_id, [{"text": text}], _EMBEDDER.embed([text]))[0]


def test_migrations_are_idempotent(tmp_path):
    path = tmp_path / "db.sqlite"
    first = dbmod.connect(path, embedding_dim=DIM)
    first.close()
    second = dbmod.connect(path, embedding_dim=DIM)
    version = second.execute("PRAGMA user_version").fetchone()[0]
    assert version == dbmod.SCHEMA_VERSION
    second.close()


def test_embedding_dim_mismatch_fails_loudly(tmp_path):
    path = tmp_path / "db.sqlite"
    dbmod.connect(path, embedding_dim=DIM).close()
    with pytest.raises(RuntimeError, match="embedding_dim"):
        dbmod.connect(path, embedding_dim=DIM * 2)


def test_run_crud(conn):
    run_id = dbmod.create_run(conn)
    run = dbmod.get_run(conn, run_id)
    assert run["status"] == "CREATED"
    dbmod.set_run_status(conn, run_id, "FAILED", error="boom")
    run = dbmod.get_run(conn, run_id)
    assert run["status"] == "FAILED"
    assert run["error"] == "boom"
    assert [r["id"] for r in dbmod.list_runs(conn)] == [run_id]


def test_set_run_status_rejects_bad_input(conn):
    run_id = dbmod.create_run(conn)
    with pytest.raises(ValueError, match="status"):
        dbmod.set_run_status(conn, run_id, "FALIED")
    with pytest.raises(ValueError, match="Unknown run"):
        dbmod.set_run_status(conn, "does-not-exist", "DONE")


def test_chunk_embedding_count_mismatch_raises(conn):
    run_id = dbmod.create_run(conn)
    doc_id = dbmod.add_document(conn, run_id, "SOURCE")
    with pytest.raises(ValueError, match="embedding"):
        dbmod.add_chunks(conn, run_id, doc_id, [{"text": "one"}], [])


def test_chunk_text_is_immutable(conn):
    run_id = dbmod.create_run(conn)
    doc_id = dbmod.add_document(conn, run_id, "SOURCE")
    chunk_id = _one_chunk(conn, run_id, doc_id, "wheat production statistics")

    # Editing text in place would desync the stored embedding — must be refused.
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with conn:
            conn.execute("UPDATE chunks SET text = ? WHERE id = ?", ("rice yields", chunk_id))
    assert keyword_search(conn, run_id, "wheat") == [chunk_id]

    # Non-text columns stay editable.
    with conn:
        conn.execute("UPDATE chunks SET page = 5 WHERE id = ?", (chunk_id,))


def _claim(conn, run_id, doc_id, text="Hunger rose in 2023."):
    return dbmod.add_claims(conn, run_id, doc_id, [{"text": text}])[0]


def _verdict_row(claim_id, **overrides):
    row = {
        "claim_id": claim_id,
        "verdict": "SUPPORTED",
        "raw_verdict": "SUPPORTED",
        "quote": "hunger rose sharply in 2023",
        "quote_verified": 1,
        "quoted_chunk_id": None,
        "evidence_chunk_ids": [1, 2],
        "year_flag": None,
        "rationale": "The source states it.",
        "model": "test-model",
    }
    row.update(overrides)
    return row


def test_verdicts_roundtrip_joins_claim_fields(conn):
    run_id = dbmod.create_run(conn)
    doc_id = dbmod.add_document(conn, run_id, "REPORT")
    claim_id = _claim(conn, run_id, doc_id)
    dbmod.add_verdicts(conn, run_id, [_verdict_row(claim_id)])

    [row] = dbmod.list_verdicts(conn, run_id)
    assert row["claim_id"] == claim_id
    assert row["verdict"] == "SUPPORTED"
    assert row["text"] == "Hunger rose in 2023."  # joined from claims
    assert row["evidence_chunk_ids"] == "[1, 2]"  # stored as JSON


def test_verdicts_replace_is_atomic(conn):
    run_id = dbmod.create_run(conn)
    doc_id = dbmod.add_document(conn, run_id, "REPORT")
    claim_id = _claim(conn, run_id, doc_id)
    dbmod.add_verdicts(conn, run_id, [_verdict_row(claim_id, rationale="old")])

    # A failed replacement must roll the DELETE back — never leave the run
    # with neither its old verdicts nor new ones.
    with pytest.raises(sqlite3.IntegrityError):
        dbmod.add_verdicts(conn, run_id, [_verdict_row(claim_id, verdict="MAYBE")], replace=True)
    assert [v["rationale"] for v in dbmod.list_verdicts(conn, run_id)] == ["old"]

    dbmod.add_verdicts(conn, run_id, [_verdict_row(claim_id, rationale="new")], replace=True)
    assert [v["rationale"] for v in dbmod.list_verdicts(conn, run_id)] == ["new"]


def test_reextract_after_verify_cascades_verdicts(conn):
    # Re-extraction deletes the document's claims; with foreign_keys=ON the
    # verdicts must CASCADE with them or the delete itself fails.
    run_id = dbmod.create_run(conn)
    doc_id = dbmod.add_document(conn, run_id, "REPORT")
    claim_id = _claim(conn, run_id, doc_id)
    dbmod.add_verdicts(conn, run_id, [_verdict_row(claim_id)])

    dbmod.add_claims(conn, run_id, doc_id, [{"text": "A fresh claim."}], replace=True)
    assert dbmod.list_verdicts(conn, run_id) == []


def test_verdict_rows_carry_prompt_hash(conn):
    run_id = dbmod.create_run(conn)
    doc_id = dbmod.add_document(conn, run_id, "REPORT")
    claim_id = _claim(conn, run_id, doc_id)
    dbmod.add_verdicts(conn, run_id, [_verdict_row(claim_id, prompt_hash="abc123")])
    [row] = dbmod.list_verdicts(conn, run_id)
    assert row["prompt_hash"] == "abc123"


# Everything migration 12 added (no indexes, so plain column drops), shared by
# every rewind below 12 — prefix it BEFORE V11_REWIND.
V12_REWIND = (
    "ALTER TABLE uploads DROP COLUMN source_type; ALTER TABLE uploads DROP COLUMN url;"
    " ALTER TABLE chunks DROP COLUMN start_seconds; ALTER TABLE chunks DROP COLUMN end_seconds;"
)

# Everything migration 11 added, dropped in one prefix shared by every rewind
# below 11 (each index drop precedes its column drop — SQLite refuses to drop
# an indexed column).
V11_REWIND = (
    "DROP INDEX idx_uploads_content_hash; ALTER TABLE uploads DROP COLUMN content_hash;"
    " DROP INDEX idx_chunks_doc; DROP INDEX idx_figures_doc;"
    " ALTER TABLE documents DROP COLUMN embedding_model;"
)


def test_migration_4_to_5_adds_prompt_hash(tmp_path):
    path = tmp_path / "db.sqlite"
    conn = dbmod.connect(path, embedding_dim=DIM)
    # Rewind: drop the column the way a v4 database lacks it.
    conn.executescript(
        V12_REWIND
        + V11_REWIND
        + " DROP TABLE run_scores; DROP TABLE source_credibility; DROP TABLE jobs;"
        " ALTER TABLE claims DROP COLUMN stance;"
        " ALTER TABLE claims DROP COLUMN extraction_prompt_hash;"
        " ALTER TABLE runs DROP COLUMN title;"
        " ALTER TABLE verdicts DROP COLUMN prompt_hash; PRAGMA user_version = 4;"
    )
    conn.close()
    conn = dbmod.connect(path, embedding_dim=DIM)
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 5
    conn.execute("SELECT prompt_hash FROM verdicts")  # column exists again
    conn.close()


def test_migration_5_to_6_rebuilds_chunks_vec_preserving_data(tmp_path):
    path = tmp_path / "db.sqlite"
    conn = dbmod.connect(path, embedding_dim=DIM)
    run_id = dbmod.create_run(conn)
    source = dbmod.add_document(conn, run_id, "SOURCE")
    report = dbmod.add_document(conn, run_id, "REPORT")
    embedder = FakeEmbedder(dim=DIM)
    [source_chunk] = dbmod.add_chunks(
        conn, run_id, source, [{"text": "source text"}], embedder.embed(["source text"])
    )
    [report_chunk] = dbmod.add_chunks(
        conn, run_id, report, [{"text": "report text"}], embedder.embed(["report text"])
    )

    # Rewind to the v5 single-partition schema, preserving the vectors.
    conn.executescript(
        f"""
        CREATE TEMP TABLE b AS SELECT chunk_id, run_id, embedding FROM chunks_vec;
        DROP TABLE chunks_vec;
        CREATE VIRTUAL TABLE chunks_vec USING vec0(
          chunk_id INTEGER PRIMARY KEY, run_id TEXT PARTITION KEY, embedding FLOAT[{DIM}]
        );
        INSERT INTO chunks_vec(chunk_id, run_id, embedding) SELECT * FROM b;
        DROP TABLE b;
        {V12_REWIND}
        {V11_REWIND}
        DROP TABLE run_scores;
        DROP TABLE source_credibility;
        DROP TABLE jobs;
        ALTER TABLE claims DROP COLUMN stance;
        ALTER TABLE claims DROP COLUMN extraction_prompt_hash;
        ALTER TABLE runs DROP COLUMN title;
        PRAGMA user_version = 5;
        """
    )
    conn.close()

    conn = dbmod.connect(path, embedding_dim=DIM)  # migration 6 runs here
    rows = conn.execute("SELECT chunk_id, doc_kind FROM chunks_vec ORDER BY chunk_id").fetchall()
    assert [(r["chunk_id"], r["doc_kind"]) for r in rows] == [
        (source_chunk, "SOURCE"),
        (report_chunk, "REPORT"),
    ]
    # The rebuilt index still answers filtered KNN queries.
    from authorai.search import vector_search

    assert vector_search(conn, run_id, [1.0] + [0.0] * (DIM - 1), k=5, doc_kind="SOURCE") == [
        source_chunk
    ]
    conn.close()


def test_migration_9_to_10_adds_run_title(tmp_path):
    path = tmp_path / "db.sqlite"
    conn = dbmod.connect(path, embedding_dim=DIM)
    # Rewind: a v9 database has no runs.title.
    conn.executescript(
        V12_REWIND + V11_REWIND + " ALTER TABLE runs DROP COLUMN title; PRAGMA user_version = 9;"
    )
    conn.close()
    conn = dbmod.connect(path, embedding_dim=DIM)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == dbmod.SCHEMA_VERSION
    conn.execute("SELECT title FROM runs")  # column exists again
    conn.close()


def test_migration_10_to_11_adds_upload_content_hash(tmp_path):
    path = tmp_path / "db.sqlite"
    conn = dbmod.connect(path, embedding_dim=DIM)
    # Rewind: a v10 database has none of migration 11's columns or indexes.
    conn.executescript(V12_REWIND + V11_REWIND + " PRAGMA user_version = 10;")
    conn.close()
    conn = dbmod.connect(path, embedding_dim=DIM)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == dbmod.SCHEMA_VERSION
    conn.execute("SELECT content_hash FROM uploads")  # columns exist again
    conn.execute("SELECT embedding_model FROM documents")
    indexes = {
        row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    }
    assert {"idx_uploads_content_hash", "idx_chunks_doc", "idx_figures_doc"} <= indexes
    conn.close()


def test_create_run_with_uploads_stores_title(conn):
    run_id, _job_id = dbmod.create_run_with_uploads_and_job(
        conn,
        [
            dbmod.UploadSpec(
                kind="REPORT", file_name="r.pdf", path="/tmp/r.pdf", content_hash="hash-a"
            )
        ],
        title="My Study",
    )
    assert dbmod.get_run(conn, run_id)["title"] == "My Study"
    [stored] = conn.execute(
        "SELECT content_hash FROM uploads WHERE content_hash IS NOT NULL"
    ).fetchall()
    assert stored["content_hash"] == "hash-a"
    untitled_id, _ = dbmod.create_run_with_uploads_and_job(
        conn, [dbmod.UploadSpec(kind="REPORT", file_name="r.pdf", path="/tmp/r.pdf")]
    )
    assert dbmod.get_run(conn, untitled_id)["title"] is None


def test_add_chunks_rejects_unknown_document(conn):
    run_id = dbmod.create_run(conn)
    with pytest.raises(ValueError, match="Unknown document"):
        dbmod.add_chunks(conn, run_id, "no-such-doc", [{"text": "x"}], _EMBEDDER.embed(["x"]))


def test_migration_3_to_4_adds_verdicts(tmp_path):
    path = tmp_path / "db.sqlite"
    conn = dbmod.connect(path, embedding_dim=DIM)
    # Rewind to a v3 state and reconnect — the v4 block must re-run cleanly.
    conn.executescript(
        V12_REWIND
        + V11_REWIND
        + " DROP TABLE verdicts; DROP TABLE run_scores; DROP TABLE source_credibility;"
        " DROP TABLE jobs; ALTER TABLE claims DROP COLUMN stance;"
        " ALTER TABLE claims DROP COLUMN extraction_prompt_hash;"
        " ALTER TABLE runs DROP COLUMN title; PRAGMA user_version = 3;"
    )
    conn.close()
    conn = dbmod.connect(path, embedding_dim=DIM)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == dbmod.SCHEMA_VERSION
    conn.execute("SELECT id, prompt_hash FROM verdicts")  # table exists, later ALTERs applied
    conn.close()


def test_delete_cleans_both_indexes(conn):
    run_id = dbmod.create_run(conn)
    doc_id = dbmod.add_document(conn, run_id, "SOURCE")
    chunk_id = _one_chunk(conn, run_id, doc_id, "wheat production statistics")
    assert keyword_search(conn, run_id, "wheat") == [chunk_id]

    with conn:
        conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
    assert keyword_search(conn, run_id, "wheat") == []
    assert conn.execute("SELECT count(*) FROM chunks_vec").fetchone()[0] == 0


def test_delete_run_data_reads_no_file_while_holding_the_write_lock(conn, tmp_path, monkeypatch):
    """Deleting a run compares its files with every other upload's, which reads the
    filesystem once per stored path. Under BEGIN IMMEDIATE, one stored path on a
    hung mount would hold the database write lock for the whole stall: the worker's
    writes fail with 'database is locked' and its job can be left RUNNING."""
    import os

    doomed_file, kept_file = tmp_path / "doomed.pdf", tmp_path / "kept.pdf"
    for file in (doomed_file, kept_file):
        file.write_bytes(b"%PDF-1.4")
    doomed = dbmod.create_run(conn)
    upload = dbmod.add_upload(conn, "REPORT", "doomed.pdf", str(doomed_file))
    dbmod.add_document(conn, doomed, "REPORT", upload_id=upload)
    other = dbmod.create_run(conn)
    kept = dbmod.add_upload(conn, "REPORT", "kept.pdf", str(kept_file))
    dbmod.add_document(conn, other, "REPORT", upload_id=kept)

    locked = False
    calls: list[tuple[str, str, bool]] = []  # (function, path, write lock held)

    def trace(statement: str) -> None:
        nonlocal locked
        keyword = statement.lstrip().upper()
        if keyword.startswith("BEGIN"):
            locked = True
        elif keyword.startswith(("COMMIT", "ROLLBACK")):
            locked = False

    def recording(name, real):
        def call(path, *args, **kwargs):
            calls.append((name, str(path), locked))
            return real(path, *args, **kwargs)

        return call

    monkeypatch.setattr(os.path, "realpath", recording("os.path.realpath", os.path.realpath))
    monkeypatch.setattr(os, "stat", recording("os.stat", os.stat))
    monkeypatch.setattr(os, "lstat", recording("os.lstat", os.lstat))
    conn.set_trace_callback(trace)
    try:
        assert dbmod.delete_run_data(conn, doomed) == [str(doomed_file)]
    finally:
        conn.set_trace_callback(None)
    assert [call for call in calls if call[2]] == []
    assert str(kept_file) in {path for _, path, _ in calls}  # the comparison still ran


# --- ingest dedup primitives ---------------------------------------------


EMB = "emb-model-x"  # the embedding model donors in this block are stamped with


def _donor_run(conn, kind="SOURCE", content_hash="cafe01", embedding_model=EMB, title="Donor Doc"):
    """A complete donor: upload + document + figure + three chunks."""
    run_id = dbmod.create_run(conn)
    upload_id = dbmod.add_upload(conn, kind, "donor.pdf", "/tmp/donor.pdf", content_hash)
    doc_id = dbmod.add_document(
        conn, run_id, kind, upload_id=upload_id, title=title, embedding_model=embedding_model
    )
    figure_id = dbmod.add_figure(conn, run_id, doc_id, "/tmp/fig-1.png", page=2, caption="cap")
    chunks = [
        {
            "text": "alpha wheat statistics",
            "page": 1,
            "section": "Intro",
            "start_seconds": 75.0,
            "end_seconds": 150.0,
        },
        {"text": "beta table of yields", "page": 2, "kind": "table"},
        {"text": "figure about crops", "page": 2, "kind": "figure", "figure_id": figure_id},
    ]
    embedder = FakeEmbedder(dim=DIM)
    chunk_ids = dbmod.add_chunks(
        conn, run_id, doc_id, chunks, embedder.embed([c["text"] for c in chunks])
    )
    return run_id, upload_id, doc_id, figure_id, chunk_ids


def test_find_ingest_donor_picks_newest_complete_and_excludes_self(conn):
    _, old_upload, old_doc, _, _ = _donor_run(conn, content_hash="samehash")
    conn.execute(
        "UPDATE uploads SET created_at = '2020-01-01T00:00:00' WHERE id = ?", (old_upload,)
    )
    conn.commit()
    _, new_upload, new_doc, _, _ = _donor_run(conn, content_hash="samehash")

    donor = dbmod.find_ingest_donor(
        conn, "samehash", embedding_model=EMB, exclude_upload_id="someone-else"
    )
    assert donor is not None and donor["doc_id"] == new_doc  # newest wins
    # Excluding the newest upload falls back to the older complete donor.
    donor = dbmod.find_ingest_donor(
        conn, "samehash", embedding_model=EMB, exclude_upload_id=new_upload
    )
    assert donor is not None and donor["doc_id"] == old_doc


def test_find_ingest_donor_filters_by_embedding_model(conn):
    """The per-document model gate: vectors made under another model — or
    never stamped at all (legacy/pre-dedup docs) — must not donate, even when
    the bytes match. A global stamp could be flip-flopped (A → B → A) into
    reusing model-B vectors under model A; per-document cannot."""
    _donor_run(conn, content_hash="modelhash", embedding_model="other-model")
    _donor_run(conn, content_hash="modelhash", embedding_model=None)  # pre-dedup doc
    assert (
        dbmod.find_ingest_donor(conn, "modelhash", embedding_model=EMB, exclude_upload_id="x")
        is None
    )

    _, _, matching_doc, _, _ = _donor_run(conn, content_hash="modelhash")
    donor = dbmod.find_ingest_donor(conn, "modelhash", embedding_model=EMB, exclude_upload_id="x")
    assert donor is not None and donor["doc_id"] == matching_doc


def test_find_ingest_donor_rejects_torn_and_null(conn):
    run_id = dbmod.create_run(conn)
    upload_id = dbmod.add_upload(conn, "SOURCE", "torn.pdf", "/tmp/torn.pdf", "tornhash")
    dbmod.add_document(conn, run_id, "SOURCE", upload_id=upload_id, embedding_model=EMB)
    torn = dbmod.find_ingest_donor(conn, "tornhash", embedding_model=EMB, exclude_upload_id="x")
    assert torn is None  # zero chunks — torn
    assert dbmod.find_ingest_donor(conn, None, embedding_model=EMB, exclude_upload_id="x") is None
    # NULL-hash uploads never match anything, not even each other.
    assert dbmod.find_ingest_donor(conn, "", embedding_model=EMB, exclude_upload_id="x") is None


def test_copy_document_data_equivalence(conn):
    donor_run, _, donor_doc, donor_figure, donor_chunk_ids = _donor_run(conn)

    new_run = dbmod.create_run(conn)
    new_upload = dbmod.add_upload(conn, "REPORT", "again.pdf", "/tmp/again.pdf", "cafe01")
    new_doc = dbmod.new_id()
    new_figure = dbmod.new_id()
    copied = dbmod.copy_document_data(
        conn,
        donor_doc,
        run_id=new_run,
        doc_id=new_doc,
        upload_id=new_upload,
        kind="REPORT",
        figure_map={donor_figure: (new_figure, "/tmp/new/fig-1.png")},
        fallback_title="again",
    )
    assert copied == 3

    columns = "page, section, kind, text, start_seconds, end_seconds"
    donor_rows = conn.execute(
        f"SELECT {columns} FROM chunks WHERE doc_id = ? ORDER BY id", (donor_doc,)
    ).fetchall()
    copy_rows = conn.execute(
        f"SELECT {columns} FROM chunks WHERE doc_id = ? ORDER BY id", (new_doc,)
    ).fetchall()
    assert [tuple(r) for r in donor_rows] == [tuple(r) for r in copy_rows]
    # The time locator really travelled (a copy dropping it would still match
    # a donor that never had one).
    assert (copy_rows[0]["start_seconds"], copy_rows[0]["end_seconds"]) == (75.0, 150.0)

    # Embedding blobs are byte-identical, order-aligned.
    copy_chunk_ids = [
        r["id"]
        for r in conn.execute("SELECT id FROM chunks WHERE doc_id = ? ORDER BY id", (new_doc,))
    ]
    for old_id, new_id_ in zip(donor_chunk_ids, copy_chunk_ids, strict=True):
        old_blob = conn.execute(
            "SELECT embedding FROM chunks_vec WHERE chunk_id = ?", (old_id,)
        ).fetchone()[0]
        new_blob = conn.execute(
            "SELECT embedding FROM chunks_vec WHERE chunk_id = ?", (new_id_,)
        ).fetchone()[0]
        assert bytes(old_blob) == bytes(new_blob)

    # Both indexes answer on the NEW run under the NEW kind.
    assert keyword_search(conn, new_run, "wheat") == [copy_chunk_ids[0]]
    from authorai.search import vector_search

    query = FakeEmbedder(dim=DIM).embed(["alpha wheat statistics"])[0]
    assert copy_chunk_ids[0] in vector_search(conn, new_run, query, k=3, doc_kind="REPORT")
    assert vector_search(conn, new_run, query, k=3, doc_kind="SOURCE") == []

    # Figure remapped; document carries donor title under the new identity.
    figure = conn.execute("SELECT * FROM figures WHERE doc_id = ?", (new_doc,)).fetchone()
    assert figure["id"] == new_figure and figure["image_path"] == "/tmp/new/fig-1.png"
    assert (
        conn.execute(
            "SELECT figure_id FROM chunks WHERE doc_id = ? AND kind = 'figure'", (new_doc,)
        ).fetchone()[0]
        == new_figure
    )
    document = conn.execute("SELECT * FROM documents WHERE id = ?", (new_doc,)).fetchone()
    assert document["title"] == "Donor Doc"  # a real (parsed-style) title is kept
    assert document["kind"] == "REPORT"
    assert document["upload_id"] == new_upload
    assert document["run_id"] == new_run
    assert document["embedding_model"] == EMB  # the vectors ARE the donor's
    # Donor untouched.
    assert (
        conn.execute("SELECT count(*) FROM chunks WHERE run_id = ?", (donor_run,)).fetchone()[0]
        == 3
    )


def test_copy_retitles_when_donor_title_was_its_filename_stem(conn):
    """A donor titled by the filename-stem FALLBACK carries the previous
    upload's name — the copy must retitle with the new upload's stem, exactly
    as a fresh ingest of these bytes would. A parsed title (≠ the donor's
    stem) is kept: identical bytes parse to identical titles."""
    _, _, fallback_doc, fig_a, _ = _donor_run(conn, content_hash="t1", title="donor")
    new_run = dbmod.create_run(conn)
    upload = dbmod.add_upload(conn, "SOURCE", "papers_v2.pdf", "/tmp/p2.pdf", "t1")
    doc = dbmod.new_id()
    dbmod.copy_document_data(
        conn,
        fallback_doc,
        run_id=new_run,
        doc_id=doc,
        upload_id=upload,
        kind="SOURCE",
        figure_map={fig_a: (dbmod.new_id(), "/tmp/new/a.png")},
        fallback_title="papers_v2",
    )
    assert (
        conn.execute("SELECT title FROM documents WHERE id = ?", (doc,)).fetchone()[0]
        == "papers_v2"
    )

    # An absent donor title also takes the new fallback.
    _, _, untitled_doc, fig_b, _ = _donor_run(conn, content_hash="t2", title=None)
    doc2 = dbmod.new_id()
    dbmod.copy_document_data(
        conn,
        untitled_doc,
        run_id=new_run,
        doc_id=doc2,
        upload_id=dbmod.add_upload(conn, "SOURCE", "other.pdf", "/tmp/o.pdf", "t2"),
        kind="SOURCE",
        figure_map={fig_b: (dbmod.new_id(), "/tmp/new/b.png")},
        fallback_title="other",
    )
    assert (
        conn.execute("SELECT title FROM documents WHERE id = ?", (doc2,)).fetchone()[0] == "other"
    )


def test_copy_document_data_is_atomic(conn):
    _, _, donor_doc, _donor_figure, _ = _donor_run(conn)
    new_run = dbmod.create_run(conn)
    new_upload = dbmod.add_upload(conn, "SOURCE", "again.pdf", "/tmp/a.pdf", "cafe01")
    new_doc = dbmod.new_id()
    with pytest.raises(KeyError):
        dbmod.copy_document_data(
            conn,
            donor_doc,
            run_id=new_run,
            doc_id=new_doc,
            upload_id=new_upload,
            kind="SOURCE",
            figure_map={},  # violates the files-first contract → abort
        )
    # Single transaction: nothing of the new document exists.
    for table, col in (("documents", "id"), ("figures", "doc_id"), ("chunks", "doc_id")):
        assert (
            conn.execute(f"SELECT count(*) FROM {table} WHERE {col} = ?", (new_doc,)).fetchone()[0]
            == 0
        )
    assert (
        conn.execute("SELECT count(*) FROM chunks_vec WHERE run_id = ?", (new_run,)).fetchone()[0]
        == 0
    )


def test_copy_document_data_refuses_incomplete_donor(conn):
    run_id = dbmod.create_run(conn)
    upload_id = dbmod.add_upload(conn, "SOURCE", "d.pdf", "/tmp/d.pdf", "h")
    doc_id = dbmod.add_document(conn, run_id, "SOURCE", upload_id=upload_id)
    with pytest.raises(ValueError, match="incomplete"):
        dbmod.copy_document_data(
            conn,
            doc_id,
            run_id=dbmod.create_run(conn),
            doc_id=dbmod.new_id(),
            upload_id=upload_id,
            kind="SOURCE",
            figure_map={},
        )


# --- migration 12: source types --------------------------------------------


def test_migration_12_adds_source_type_url_and_time_locators(conn):
    conn.execute("SELECT source_type, url FROM uploads")
    conn.execute("SELECT start_seconds, end_seconds FROM chunks")
    assert dbmod.SCHEMA_VERSION == 12


def test_migration_11_to_12_defaults_legacy_uploads_to_pdf(tmp_path):
    path = tmp_path / "db.sqlite"
    conn = dbmod.connect(path, embedding_dim=DIM)
    # Rewind: a v11 database has none of migration 12's columns. A row written
    # then is an uploaded PDF by construction.
    conn.executescript(V12_REWIND + " PRAGMA user_version = 11;")
    conn.execute(
        "INSERT INTO uploads(id, kind, file_name, path, created_at)"
        " VALUES ('legacy', 'SOURCE', 'old.pdf', '/tmp/old.pdf', '2026-01-01')"
    )
    conn.commit()
    conn.close()
    conn = dbmod.connect(path, embedding_dim=DIM)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == dbmod.SCHEMA_VERSION
    row = conn.execute("SELECT source_type, url FROM uploads WHERE id = 'legacy'").fetchone()
    assert (row["source_type"], row["url"]) == ("pdf", None)
    conn.execute("SELECT start_seconds, end_seconds FROM chunks")
    conn.close()


def test_add_upload_records_source_type_and_url(conn):
    web = dbmod.add_upload(
        conn, "SOURCE", "example.org", "/tmp/u.json", source_type="web", url="https://example.org/a"
    )
    pdf = dbmod.add_upload(conn, "SOURCE", "a.pdf", "/tmp/a.pdf")
    rows = {
        r["id"]: (r["source_type"], r["url"])
        for r in conn.execute("SELECT id, source_type, url FROM uploads")
    }
    assert rows[web] == ("web", "https://example.org/a")
    assert rows[pdf] == ("pdf", None)


@pytest.mark.parametrize("bad", ["ftp", "PDF", "", "html"])
def test_unknown_source_type_is_refused_before_any_write(conn, bad):
    with pytest.raises(ValueError, match="source type"):
        dbmod.add_upload(conn, "SOURCE", "x", "/tmp/x", source_type=bad)
    with pytest.raises(ValueError, match="source type"):
        dbmod.create_run_with_uploads_and_job(
            conn,
            [
                dbmod.UploadSpec(kind="REPORT", file_name="r.pdf", path="/tmp/r.pdf"),
                dbmod.UploadSpec(kind="SOURCE", file_name="x", path="/tmp/x", source_type=bad),
            ],
        )
    assert conn.execute("SELECT count(*) FROM uploads").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


def test_add_chunks_stores_time_locators(conn):
    run_id = dbmod.create_run(conn)
    doc_id = dbmod.add_document(conn, run_id, "SOURCE")
    chunks = [
        {"text": "transcript window one", "start_seconds": 0.0, "end_seconds": 75.0},
        {"text": "a page chunk", "page": 3},
    ]
    dbmod.add_chunks(conn, run_id, doc_id, chunks, _EMBEDDER.embed([c["text"] for c in chunks]))
    rows = conn.execute(
        "SELECT page, start_seconds, end_seconds FROM chunks WHERE doc_id = ? ORDER BY id",
        (doc_id,),
    ).fetchall()
    assert [tuple(r) for r in rows] == [(None, 0.0, 75.0), (3, None, None)]


def test_create_run_with_upload_specs_records_source_fields(conn):
    run_id, job_id = dbmod.create_run_with_uploads_and_job(
        conn,
        [
            dbmod.UploadSpec(
                kind="REPORT", file_name="r.pdf", path="/tmp/r.pdf", content_hash="h1"
            ),
            dbmod.UploadSpec(
                kind="SOURCE",
                file_name="example.org",
                path="/tmp/u.json",
                source_type="web",
                url="https://example.org/facts",
            ),
        ],
        title="Mixed",
    )
    payload = dbmod.get_job(conn, job_id)["payload"]
    [source_id] = payload["source_upload_ids"]
    row = conn.execute("SELECT * FROM uploads WHERE id = ?", (source_id,)).fetchone()
    assert (row["source_type"], row["url"], row["content_hash"]) == (
        "web",
        "https://example.org/facts",
        None,
    )
    report = conn.execute(
        "SELECT source_type, url FROM uploads WHERE id = ?", (payload["report_upload_id"],)
    ).fetchone()
    assert (report["source_type"], report["url"]) == ("pdf", None)


def test_record_fetch_moves_a_link_upload_to_its_stored_pdf_in_one_update(conn):
    """One UPDATE, and a refused type writes nothing: a path moved while the type
    still says web (or the reverse) is a state no retry can recover from — the
    stored PDF would be read as a page, or a missing file never fetched again."""
    url = "https://example.org/report.pdf"
    upload = dbmod.add_upload(conn, "SOURCE", url, "/tmp/u.json", source_type="web", url=url)

    def stored():
        row = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload,)).fetchone()
        return (row["path"], row["source_type"], row["content_hash"], row["url"])

    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        dbmod.record_fetch(conn, upload, path="/tmp/u.pdf", source_type="pdf", content_hash="abc")
    finally:
        conn.set_trace_callback(None)
    assert sum(s.lstrip().upper().startswith("UPDATE") for s in statements) == 1
    assert stored() == ("/tmp/u.pdf", "pdf", "abc", url)
    with pytest.raises(ValueError, match="source type"):
        dbmod.record_fetch(conn, upload, path="/tmp/x", source_type="ftp", content_hash=None)
    assert stored() == ("/tmp/u.pdf", "pdf", "abc", url)
    with pytest.raises(ValueError, match="Unknown upload"):
        dbmod.record_fetch(conn, "nope", path="/tmp/x", source_type="pdf", content_hash=None)


def test_list_run_sources_orders_scored_by_total_then_unscored_in_ingest_order(conn):
    """The report's source list and the chat context show sources in exactly this
    order (the frontend never re-sorts, and the credibility view opens on the
    first): scored sources by total, highest first, then the rest as ingested."""
    run_id = dbmod.create_run(conn)
    dbmod.add_document(conn, run_id, "REPORT", title="The Report")  # never listed
    unscored_first = dbmod.add_document(conn, run_id, "SOURCE", title="unscored first")
    scored_low = dbmod.add_document(conn, run_id, "SOURCE", title="scored low")
    unscored_second = dbmod.add_document(conn, run_id, "SOURCE", title="unscored second")
    scored_high = dbmod.add_document(conn, run_id, "SOURCE", title="scored high")
    dbmod.save_source_credibility(
        conn,
        run_id,
        [
            {"doc_id": doc, "metadata": {}, "components": {}, "total": total, "tier": "NONE"}
            for doc, total in ((scored_low, 40.0), (scored_high, 80.0))
        ],
    )
    order = [row["doc_id"] for row in dbmod.list_run_sources(conn, run_id)]
    assert order == [scored_high, scored_low, unscored_first, unscored_second]
