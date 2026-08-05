"""SQLite storage: run-scoped schema, migrations, and repository functions.

One database file holds everything: relational tables, the FTS5 keyword index,
and the sqlite-vec vector index. Every pipeline table carries a `run_id` so
runs never overwrite each other — there is no reset step, ever.
"""

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32

from authorai.embeddings import normalize

SCHEMA_VERSION = 1


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def connect(db_path: Path | str, embedding_dim: int) -> sqlite3.Connection:
    """Open (creating/migrating if needed) the database.

    Fails loudly if the database was created with a different embedding
    dimension than the one configured — a dimension mismatch would otherwise
    corrupt every similarity search silently.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    _migrate(conn, embedding_dim)
    _check_embedding_dim(conn, embedding_dim)
    return conn


def _migrate(conn: sqlite3.Connection, embedding_dim: int) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version >= SCHEMA_VERSION:
        return
    if version < 1:
        # The whole migration — DDL, meta row, and version bump — runs in ONE
        # transaction so an interruption can never leave a half-created schema
        # with user_version still 0 (which would brick every later connect).
        conn.executescript(
            f"""
            BEGIN;

            CREATE TABLE meta(
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            CREATE TABLE runs(
              id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'CREATED',
              error TEXT
            );

            CREATE TABLE uploads(
              id TEXT PRIMARY KEY,
              kind TEXT NOT NULL CHECK(kind IN ('SOURCE', 'REPORT')),
              file_name TEXT NOT NULL,
              path TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE documents(
              id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL REFERENCES runs(id),
              upload_id TEXT REFERENCES uploads(id),
              kind TEXT NOT NULL CHECK(kind IN ('SOURCE', 'REPORT')),
              title TEXT,
              metadata TEXT NOT NULL DEFAULT '{{}}'
            );
            CREATE INDEX idx_documents_run ON documents(run_id);

            CREATE TABLE chunks(
              id INTEGER PRIMARY KEY,
              run_id TEXT NOT NULL REFERENCES runs(id),
              doc_id TEXT NOT NULL REFERENCES documents(id),
              page INTEGER,
              section TEXT,
              kind TEXT NOT NULL DEFAULT 'text' CHECK(kind IN ('text', 'table', 'figure')),
              text TEXT NOT NULL
            );
            CREATE INDEX idx_chunks_run ON chunks(run_id);

            CREATE VIRTUAL TABLE chunks_fts USING fts5(
              text,
              content='chunks',
              content_rowid='id'
            );

            CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
              INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER chunks_fts_delete AFTER DELETE ON chunks BEGIN
              INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
              DELETE FROM chunks_vec WHERE chunk_id = old.id;
            END;
            -- Chunk text is immutable: an in-place edit would desync the stored
            -- embedding (there is no way to re-embed from inside SQL). Delete
            -- the chunk and re-add it instead.
            CREATE TRIGGER chunks_text_immutable BEFORE UPDATE OF text ON chunks BEGIN
              SELECT RAISE(ABORT, 'chunk text is immutable — delete the chunk and re-add it');
            END;

            CREATE VIRTUAL TABLE chunks_vec USING vec0(
              chunk_id INTEGER PRIMARY KEY,
              run_id TEXT PARTITION KEY,
              embedding FLOAT[{embedding_dim}]
            );

            INSERT INTO meta(key, value) VALUES ('embedding_dim', '{int(embedding_dim)}');
            PRAGMA user_version = {SCHEMA_VERSION};

            COMMIT;
            """
        )


def _check_embedding_dim(conn: sqlite3.Connection, embedding_dim: int) -> None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'embedding_dim'").fetchone()
    stored = int(row["value"])
    if stored != embedding_dim:
        raise RuntimeError(
            f"Database was created with embedding_dim={stored} but settings say "
            f"{embedding_dim}. Refusing to mix incompatible vectors — use a new "
            f"database file or restore the original setting."
        )


# --- runs ---------------------------------------------------------------


def create_run(conn: sqlite3.Connection) -> str:
    run_id = new_id()
    with conn:
        conn.execute("INSERT INTO runs(id, created_at) VALUES (?, ?)", (run_id, now_iso()))
    return run_id


def get_run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def list_runs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def set_run_status(
    conn: sqlite3.Connection, run_id: str, status: str, error: str | None = None
) -> None:
    with conn:
        conn.execute(
            "UPDATE runs SET status = ?, error = ? WHERE id = ?",
            (status, error, run_id),
        )


# --- uploads and documents ----------------------------------------------


def add_upload(conn: sqlite3.Connection, kind: str, file_name: str, path: str) -> str:
    upload_id = new_id()
    with conn:
        conn.execute(
            "INSERT INTO uploads(id, kind, file_name, path, created_at) VALUES (?, ?, ?, ?, ?)",
            (upload_id, kind, file_name, path, now_iso()),
        )
    return upload_id


def add_document(
    conn: sqlite3.Connection,
    run_id: str,
    kind: str,
    upload_id: str | None = None,
    title: str | None = None,
    metadata: str = "{}",
) -> str:
    doc_id = new_id()
    with conn:
        conn.execute(
            "INSERT INTO documents(id, run_id, upload_id, kind, title, metadata)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, run_id, upload_id, kind, title, metadata),
        )
    return doc_id


# --- chunks --------------------------------------------------------------


def add_chunks(
    conn: sqlite3.Connection,
    run_id: str,
    doc_id: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> list[int]:
    """Insert chunks and their embeddings in one transaction.

    Each chunk dict may carry `text` (required), `page`, `section`, `kind`.
    Embeddings are L2-normalized on write so vector distances behave as cosine.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"{len(chunks)} chunks but {len(embeddings)} embeddings — "
            "every chunk needs exactly one embedding"
        )
    chunk_ids: list[int] = []
    with conn:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            cursor = conn.execute(
                "INSERT INTO chunks(run_id, doc_id, page, section, kind, text)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    doc_id,
                    chunk.get("page"),
                    chunk.get("section"),
                    chunk.get("kind", "text"),
                    chunk["text"],
                ),
            )
            chunk_id = cursor.lastrowid
            conn.execute(
                "INSERT INTO chunks_vec(chunk_id, run_id, embedding) VALUES (?, ?, ?)",
                (chunk_id, run_id, serialize_float32(normalize(embedding))),
            )
            chunk_ids.append(chunk_id)
    return chunk_ids


def get_chunks(conn: sqlite3.Connection, chunk_ids: list[int]) -> dict[int, dict]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = conn.execute(f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids).fetchall()
    return {row["id"]: dict(row) for row in rows}
