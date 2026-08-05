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


def test_delete_cleans_both_indexes(conn):
    run_id = dbmod.create_run(conn)
    doc_id = dbmod.add_document(conn, run_id, "SOURCE")
    chunk_id = _one_chunk(conn, run_id, doc_id, "wheat production statistics")
    assert keyword_search(conn, run_id, "wheat") == [chunk_id]

    with conn:
        conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
    assert keyword_search(conn, run_id, "wheat") == []
    assert conn.execute("SELECT count(*) FROM chunks_vec").fetchone()[0] == 0
