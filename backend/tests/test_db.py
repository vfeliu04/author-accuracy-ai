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


def test_migration_4_to_5_adds_prompt_hash(tmp_path):
    path = tmp_path / "db.sqlite"
    conn = dbmod.connect(path, embedding_dim=DIM)
    # Rewind: drop the column the way a v4 database lacks it.
    conn.executescript("ALTER TABLE verdicts DROP COLUMN prompt_hash; PRAGMA user_version = 4;")
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


def test_add_chunks_rejects_unknown_document(conn):
    run_id = dbmod.create_run(conn)
    with pytest.raises(ValueError, match="Unknown document"):
        dbmod.add_chunks(conn, run_id, "no-such-doc", [{"text": "x"}], _EMBEDDER.embed(["x"]))


def test_migration_3_to_4_adds_verdicts(tmp_path):
    path = tmp_path / "db.sqlite"
    conn = dbmod.connect(path, embedding_dim=DIM)
    # Rewind to a v3 state and reconnect — the v4 block must re-run cleanly.
    conn.executescript("DROP TABLE verdicts; PRAGMA user_version = 3;")
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
