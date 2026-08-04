import pytest

from authorai import db as dbmod
from authorai.search import keyword_search
from tests.conftest import DIM


def _one_chunk(conn, run_id, doc_id, text):
    return dbmod.add_chunks(conn, run_id, doc_id, [{"text": text}], [[1.0] + [0.0] * (DIM - 1)])[0]


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


def test_chunk_embedding_count_mismatch_raises(conn):
    run_id = dbmod.create_run(conn)
    doc_id = dbmod.add_document(conn, run_id, "SOURCE")
    with pytest.raises(ValueError, match="embedding"):
        dbmod.add_chunks(conn, run_id, doc_id, [{"text": "one"}], [])


def test_fts_stays_in_sync_through_insert_update_delete(conn):
    run_id = dbmod.create_run(conn)
    doc_id = dbmod.add_document(conn, run_id, "SOURCE")
    chunk_id = _one_chunk(conn, run_id, doc_id, "wheat production statistics")

    assert keyword_search(conn, run_id, "wheat") == [chunk_id]

    with conn:
        conn.execute("UPDATE chunks SET text = ? WHERE id = ?", ("rice yields", chunk_id))
    assert keyword_search(conn, run_id, "wheat") == []
    assert keyword_search(conn, run_id, "rice") == [chunk_id]

    with conn:
        conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
    assert keyword_search(conn, run_id, "rice") == []
    vec_rows = conn.execute("SELECT count(*) FROM chunks_vec").fetchone()[0]
    assert vec_rows == 0
