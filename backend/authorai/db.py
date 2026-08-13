"""SQLite storage: run-scoped schema, migrations, and repository functions.

One database file holds everything: relational tables, the FTS5 keyword index,
and the sqlite-vec vector index. Every pipeline table carries a `run_id` so
runs never overwrite each other — there is no reset step, ever.
"""

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

import sqlite_vec
from sqlite_vec import serialize_float32

from authorai.embeddings import normalize

SCHEMA_VERSION = 8

JOB_STATUSES = ("QUEUED", "RUNNING", "DONE", "FAILED")

RUN_STATUSES = frozenset({"CREATED", "RUNNING", "DONE", "FAILED"})

DOC_KINDS = ("SOURCE", "REPORT")

# THE verdict vocabulary: the Verdict model annotates with the Literal and the
# SQL CHECK interpolates the derived tuple, so the two cannot drift apart.
VerdictLiteral = Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]
VERDICTS: tuple[str, ...] = get_args(VerdictLiteral)


def is_downgraded(verdict_row: dict) -> bool:
    """Did the code checks downgrade this verdict from what the model said?

    The one definition both the pipeline summary and the eval scorer use —
    quote_verified==0 alone is NOT it (a raw-UNVERIFIABLE verdict whose
    volunteered quote failed was never downgraded).
    """
    return verdict_row.get("raw_verdict") not in (None, verdict_row.get("verdict"))


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def connect(
    db_path: Path | str, embedding_dim: int, *, check_same_thread: bool = True
) -> sqlite3.Connection:
    """Open (creating/migrating if needed) the database.

    Fails loudly if the database was created with a different embedding
    dimension than the one configured — a dimension mismatch would otherwise
    corrupt every similarity search silently.

    `check_same_thread=False` is for the API's per-request connections only:
    FastAPI runs a request's dependencies and its endpoint on different
    threadpool threads, but strictly one at a time, so cross-thread use is
    sequential and safe. Everything else keeps the thread guard.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    # busy_timeout FIRST: the WAL switch and the first-open migration both need
    # an exclusive lock, so the timeout must already be in force or a racing
    # opener throws SQLITE_BUSY instantly instead of waiting the 5 s.
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    _migrate(conn, embedding_dim)
    _check_embedding_dim(conn, embedding_dim)
    return conn


def _migrate(conn: sqlite3.Connection, embedding_dim: int) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        # A database written by a NEWER build. Returning silently would let an
        # old checkout write through a schema it doesn't understand — e.g. the
        # pre-partition add_chunks stores doc_kind=NULL into the two-key
        # chunks_vec, and every SOURCE-filtered search then silently misses it.
        raise RuntimeError(
            f"Database schema version {version} is newer than this build supports "
            f"({SCHEMA_VERSION}) — refusing to open it with an older Author AI."
        )
    if version == SCHEMA_VERSION:
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
            PRAGMA user_version = 1;

            COMMIT;
            """
        )
    if version < 2:
        conn.executescript(
            """
            BEGIN;

            CREATE TABLE figures(
              id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL REFERENCES runs(id),
              doc_id TEXT NOT NULL REFERENCES documents(id),
              page INTEGER,
              image_path TEXT NOT NULL,
              caption TEXT
            );
            CREATE INDEX idx_figures_run ON figures(run_id);

            ALTER TABLE chunks ADD COLUMN figure_id TEXT REFERENCES figures(id);

            PRAGMA user_version = 2;

            COMMIT;
            """
        )
    if version < 3:
        conn.executescript(
            """
            BEGIN;

            CREATE TABLE claims(
              id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL REFERENCES runs(id),
              doc_id TEXT NOT NULL REFERENCES documents(id),
              page INTEGER,
              text TEXT NOT NULL,
              value REAL,
              unit TEXT,
              year INTEGER,
              subject TEXT,
              created_at TEXT NOT NULL
            );
            CREATE INDEX idx_claims_run ON claims(run_id);

            ALTER TABLE figures ADD COLUMN description TEXT;

            PRAGMA user_version = 3;

            COMMIT;
            """
        )
    if version < 4:
        verdict_check = "('" + "', '".join(VERDICTS) + "')"
        conn.executescript(
            f"""
            BEGIN;

            CREATE TABLE verdicts(
              id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL REFERENCES runs(id),
              -- CASCADE: re-extraction replaces a document's claims inside a
              -- transaction; with foreign_keys=ON a plain FK would make that
              -- DELETE fail once verdicts exist. Verdicts on deleted claims
              -- are meaningless, so they go with them.
              claim_id TEXT NOT NULL UNIQUE REFERENCES claims(id) ON DELETE CASCADE,
              verdict TEXT NOT NULL CHECK(verdict IN {verdict_check}),
              raw_verdict TEXT NOT NULL CHECK(raw_verdict IN {verdict_check}),
              quote TEXT,
              quote_verified INTEGER,          -- NULL = no quote applicable, 1 = ok, 0 = FAILED
              quoted_chunk_id INTEGER REFERENCES chunks(id),
              evidence_chunk_ids TEXT NOT NULL, -- JSON array of chunk ids shown to the judge
              year_flag INTEGER,               -- NULL = n/a, 1 = claim year absent from cited chunk
              rationale TEXT NOT NULL,
              model TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE INDEX idx_verdicts_run ON verdicts(run_id);

            PRAGMA user_version = 4;

            COMMIT;
            """
        )
    if version < 5:
        conn.executescript(
            """
            BEGIN;

            -- Which VERDICT_SYSTEM produced each row. eval-verdict refuses to
            -- score rows whose hash differs from the current prompt: scoring
            -- stale verdicts as if they were fresh already caused one wrong
            -- conclusion (MISTAKES.md 2026-08-08). NULL (pre-migration rows)
            -- counts as stale.
            ALTER TABLE verdicts ADD COLUMN prompt_hash TEXT;

            PRAGMA user_version = 5;

            COMMIT;
            """
        )
    if version < 6:
        # Rebuild chunks_vec with doc_kind as a second PARTITION KEY so
        # verification's SOURCE-only retrieval is index-native instead of the
        # over-fetch workaround (which capped at sqlite-vec's k=4096 and could
        # starve the vector channel on large runs). Data-preserving: vec0
        # embeddings SELECT out as bytes and re-insert cleanly.
        # The dim MUST come from meta — _migrate runs before the embedding-dim
        # check, and using the connect() parameter here would rebuild the table
        # at a mismatched dimension before that check could refuse it.
        dim = int(conn.execute("SELECT value FROM meta WHERE key = 'embedding_dim'").fetchone()[0])
        conn.executescript(
            f"""
            BEGIN;

            CREATE TEMP TABLE vec_backup AS
              SELECT v.chunk_id, v.run_id, d.kind AS doc_kind, v.embedding
              FROM chunks_vec v
              JOIN chunks c ON c.id = v.chunk_id
              JOIN documents d ON d.id = c.doc_id;

            DROP TABLE chunks_vec;

            CREATE VIRTUAL TABLE chunks_vec USING vec0(
              chunk_id INTEGER PRIMARY KEY,
              run_id TEXT PARTITION KEY,
              doc_kind TEXT PARTITION KEY,
              embedding FLOAT[{dim}]
            );

            INSERT INTO chunks_vec(chunk_id, run_id, doc_kind, embedding)
              SELECT chunk_id, run_id, doc_kind, embedding FROM vec_backup;

            DROP TABLE vec_backup;

            PRAGMA user_version = 6;

            COMMIT;
            """
        )
    if version < 7:
        conn.executescript(
            """
            BEGIN;

            -- Derived scores, persisted so the API serves them without
            -- recomputing (and so score history survives — run-scoped, never
            -- reset). The three run-level scores are only ever fetched whole,
            -- so JSON columns; per-source credibility gets real rows because
            -- the UI drills into individual sources.
            CREATE TABLE run_scores(
              run_id TEXT PRIMARY KEY REFERENCES runs(id),
              accuracy TEXT NOT NULL,
              credibility TEXT NOT NULL,
              validity TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE source_credibility(
              id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL REFERENCES runs(id),
              doc_id TEXT NOT NULL UNIQUE REFERENCES documents(id),
              metadata TEXT NOT NULL,
              components TEXT NOT NULL,
              total REAL NOT NULL,
              tier TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE INDEX idx_source_credibility_run ON source_credibility(run_id);

            PRAGMA user_version = 7;

            COMMIT;
            """
        )
    if version < 8:
        status_check = "('" + "', '".join(JOB_STATUSES) + "')"
        conn.executescript(
            f"""
            BEGIN;

            -- Background pipeline jobs. `payload` carries the work order
            -- (upload ids) so recovery never has to guess what a run was
            -- meant to contain; `progress` is a JSON array of
            -- {{step,label,status,ts}} entries upserted by step name.
            CREATE TABLE jobs(
              id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL REFERENCES runs(id),
              kind TEXT NOT NULL DEFAULT 'full_pipeline',
              status TEXT NOT NULL DEFAULT 'QUEUED' CHECK(status IN {status_check}),
              payload TEXT NOT NULL,
              progress TEXT NOT NULL DEFAULT '[]',
              error TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE INDEX idx_jobs_run ON jobs(run_id);

            PRAGMA user_version = 8;

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
    if status not in RUN_STATUSES:
        raise ValueError(f"Unknown run status {status!r}; expected one of {sorted(RUN_STATUSES)}")
    with conn:
        cursor = conn.execute(
            "UPDATE runs SET status = ?, error = ? WHERE id = ?",
            (status, error, run_id),
        )
    if cursor.rowcount == 0:
        raise ValueError(f"Unknown run {run_id!r}")


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
    doc_id: str | None = None,
) -> str:
    doc_id = doc_id or new_id()
    with conn:
        conn.execute(
            "INSERT INTO documents(id, run_id, upload_id, kind, title, metadata)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, run_id, upload_id, kind, title, metadata),
        )
    return doc_id


def get_document_path(conn: sqlite3.Connection, run_id: str, doc_id: str) -> tuple[str, str] | None:
    """The stored PDF path + original file name for a document, scoped to its run.

    Returns None when the (run_id, doc_id) pair does not exist or the document
    has no upload backing it — the file endpoint turns None into a 404. The
    run_id predicate is the access boundary: a doc_id from another run resolves
    to nothing.
    """
    row = conn.execute(
        """
        SELECT u.path, u.file_name
        FROM documents d JOIN uploads u ON u.id = d.upload_id
        WHERE d.id = ? AND d.run_id = ?
        """,
        (doc_id, run_id),
    ).fetchone()
    return (row["path"], row["file_name"]) if row else None


def get_report_doc_id(conn: sqlite3.Connection, run_id: str) -> str | None:
    """The run's REPORT document id (for the report-PDF pane). None if unset."""
    row = conn.execute(
        "SELECT id FROM documents WHERE run_id = ? AND kind = 'REPORT' LIMIT 1", (run_id,)
    ).fetchone()
    return row["id"] if row else None


def add_figure(
    conn: sqlite3.Connection,
    run_id: str,
    doc_id: str,
    image_path: str,
    page: int | None = None,
    caption: str | None = None,
    description: str | None = None,
    figure_id: str | None = None,
) -> str:
    figure_id = figure_id or new_id()
    with conn:
        conn.execute(
            "INSERT INTO figures(id, run_id, doc_id, page, image_path, caption, description)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (figure_id, run_id, doc_id, page, image_path, caption, description),
        )
    return figure_id


def add_claims(
    conn: sqlite3.Connection,
    run_id: str,
    doc_id: str,
    claims: list[dict],
    *,
    replace: bool = False,
) -> list[str]:
    """Insert extracted claims in one transaction; returns their ids.

    With `replace`, the document's existing claims are deleted inside that SAME
    transaction — deleting first and inserting after would leave the document
    holding neither the old claims nor the new ones if the insert failed.
    """
    claim_ids: list[str] = []
    with conn:
        if replace:
            conn.execute("DELETE FROM claims WHERE doc_id = ?", (doc_id,))
        for claim in claims:
            claim_id = new_id()
            conn.execute(
                "INSERT INTO claims(id, run_id, doc_id, page, text, value, unit, year,"
                " subject, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    claim_id,
                    run_id,
                    doc_id,
                    claim.get("page"),
                    claim["text"],
                    claim.get("value"),
                    claim.get("unit"),
                    claim.get("year"),
                    claim.get("subject"),
                    now_iso(),
                ),
            )
            claim_ids.append(claim_id)
    return claim_ids


def add_verdicts(
    conn: sqlite3.Connection,
    run_id: str,
    verdicts: list[dict],
    *,
    replace: bool = False,
) -> list[str]:
    """Insert verdicts in one transaction; returns their ids.

    With `replace`, the run's existing verdicts are deleted inside that SAME
    transaction (the add_claims lesson: delete-then-insert across two
    transactions loses the old rows when the insert fails).
    """
    verdict_ids: list[str] = []
    with conn:
        if replace:
            conn.execute("DELETE FROM verdicts WHERE run_id = ?", (run_id,))
        for verdict in verdicts:
            verdict_id = new_id()
            conn.execute(
                "INSERT INTO verdicts(id, run_id, claim_id, verdict, raw_verdict, quote,"
                " quote_verified, quoted_chunk_id, evidence_chunk_ids, year_flag, rationale,"
                " model, prompt_hash, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    verdict_id,
                    run_id,
                    verdict["claim_id"],
                    verdict["verdict"],
                    verdict["raw_verdict"],
                    verdict.get("quote"),
                    verdict.get("quote_verified"),
                    verdict.get("quoted_chunk_id"),
                    json.dumps(verdict.get("evidence_chunk_ids", [])),
                    verdict.get("year_flag"),
                    verdict["rationale"],
                    verdict["model"],
                    verdict.get("prompt_hash"),
                    now_iso(),
                ),
            )
            verdict_ids.append(verdict_id)
    return verdict_ids


def list_verdicts(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """A run's verdicts joined with their claims, in document order."""
    rows = conn.execute(
        """
        SELECT v.*, c.text, c.value, c.unit, c.year, c.page
        FROM verdicts v JOIN claims c ON c.id = v.claim_id
        WHERE v.run_id = ? ORDER BY c.page, c.id
        """,
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def list_verdicts_with_evidence(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """list_verdicts plus where each quoted evidence chunk came from.

    LEFT JOINs (a verdict may cite no chunk) resolve quoted_chunk_id to its
    source document and page, so the report endpoint can show the reader which
    source backs each verdict without a second round of queries.
    """
    rows = conn.execute(
        """
        SELECT v.*, c.text, c.value, c.unit, c.year, c.page,
               ch.page AS evidence_page, d.id AS evidence_doc_id, d.title AS evidence_doc_title
        FROM verdicts v
        JOIN claims c ON c.id = v.claim_id
        LEFT JOIN chunks ch ON ch.id = v.quoted_chunk_id
        LEFT JOIN documents d ON d.id = ch.doc_id
        WHERE v.run_id = ? ORDER BY c.page, c.id
        """,
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def create_job(
    conn: sqlite3.Connection, run_id: str, payload: dict, kind: str = "full_pipeline"
) -> str:
    job_id = new_id()
    with conn:
        conn.execute(
            "INSERT INTO jobs(id, run_id, kind, payload, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, run_id, kind, json.dumps(payload), now_iso(), now_iso()),
        )
    return job_id


def create_run_with_uploads_and_job(
    conn: sqlite3.Connection, uploads: list[tuple[str, str, str]]
) -> tuple[str, str]:
    """Create a run, its upload rows, and its pipeline job in ONE transaction.

    `uploads` is a list of (kind, file_name, path); exactly one must be REPORT.
    All-or-nothing so a failure part-way never strands a run with no job, or
    uploads with no run (the API's atomicity guarantee — v1 committed each
    separately and left orphans on the first failure).
    """
    report_upload_id: str | None = None
    source_upload_ids: list[str] = []
    now = now_iso()
    run_id = new_id()
    with conn:
        conn.execute("INSERT INTO runs(id, created_at) VALUES (?, ?)", (run_id, now))
        for kind, file_name, path in uploads:
            upload_id = new_id()
            conn.execute(
                "INSERT INTO uploads(id, kind, file_name, path, created_at) VALUES (?, ?, ?, ?, ?)",
                (upload_id, kind, file_name, path, now),
            )
            if kind == "REPORT":
                report_upload_id = upload_id
            else:
                source_upload_ids.append(upload_id)
        if report_upload_id is None:
            raise ValueError("create_run_with_uploads_and_job needs exactly one REPORT upload")
        job_id = new_id()
        payload = {"report_upload_id": report_upload_id, "source_upload_ids": source_upload_ids}
        conn.execute(
            "INSERT INTO jobs(id, run_id, kind, payload, created_at, updated_at)"
            " VALUES (?, ?, 'full_pipeline', ?, ?, ?)",
            (job_id, run_id, json.dumps(payload), now, now),
        )
    return run_id, job_id


def _job_row(row: sqlite3.Row) -> dict:
    job = dict(row)
    job["payload"] = json.loads(job["payload"])
    job["progress"] = json.loads(job["progress"])
    return job


def get_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _job_row(row) if row else None


def get_run_job(conn: sqlite3.Connection, run_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM jobs WHERE run_id = ? ORDER BY created_at DESC LIMIT 1", (run_id,)
    ).fetchone()
    return _job_row(row) if row else None


def claim_next_job(conn: sqlite3.Connection) -> dict | None:
    """Atomically claim the oldest QUEUED job (compare-and-set via one UPDATE)."""
    with conn:
        row = conn.execute(
            """
            UPDATE jobs SET status = 'RUNNING', updated_at = ?
            WHERE id = (SELECT id FROM jobs WHERE status = 'QUEUED' ORDER BY created_at LIMIT 1)
            RETURNING *
            """,
            (now_iso(),),
        ).fetchone()
    return _job_row(row) if row else None


def push_job_progress(
    conn: sqlite3.Connection, job_id: str, step: str, label: str, status: str = "done"
) -> None:
    """Upsert one step entry in the job's progress array (keyed by step name).

    This is a read-modify-write of the whole JSON array, so it takes the write
    lock with BEGIN IMMEDIATE BEFORE the SELECT — two writers reading the same
    array and both appending would otherwise silently lose an entry (and a lost
    'done' entry would make a completed step re-run on resume, re-billing an
    LLM batch). Callers never hold an open transaction here.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT progress FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown job {job_id!r}")
        progress = json.loads(row["progress"])
        entry = {"step": step, "label": label, "status": status, "ts": now_iso()}
        for i, existing in enumerate(progress):
            if existing["step"] == step:
                progress[i] = entry
                break
        else:
            progress.append(entry)
        conn.execute(
            "UPDATE jobs SET progress = ?, updated_at = ? WHERE id = ?",
            (json.dumps(progress), now_iso(), job_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def finish_job_and_run(
    conn: sqlite3.Connection, job_id: str, run_id: str, status: str, error: str | None = None
) -> None:
    """Terminal state for a job AND its run in ONE transaction.

    Two separate writes would leave a crash window where the job reads DONE
    but the run stays RUNNING forever — startup recovery keys off RUNNING
    jobs, so nothing would ever repair the divergence.
    """
    if status not in ("DONE", "FAILED"):
        raise ValueError(f"finish_job_and_run only accepts DONE/FAILED, got {status!r}")
    with conn:
        job_cursor = conn.execute(
            "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            (status, error, now_iso(), job_id),
        )
        run_cursor = conn.execute(
            "UPDATE runs SET status = ?, error = ? WHERE id = ?", (status, error, run_id)
        )
        if job_cursor.rowcount == 0 or run_cursor.rowcount == 0:
            raise ValueError(f"Unknown job {job_id!r} or run {run_id!r}")


def requeue_running_jobs(conn: sqlite3.Connection) -> list[str]:
    """Startup recovery: a RUNNING job after a restart is an interrupted job —
    re-queue it (steps are idempotent / reconciled) instead of stranding it
    RUNNING forever, which is what v1 did."""
    with conn:
        rows = conn.execute(
            "UPDATE jobs SET status = 'QUEUED', updated_at = ? WHERE status = 'RUNNING'"
            " RETURNING id",
            (now_iso(),),
        ).fetchall()
    job_ids = [row["id"] for row in rows]
    for job_id in job_ids:
        push_job_progress(conn, job_id, "recovered", "Re-queued after restart", status="done")
    return job_ids


def save_run_scores(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    accuracy: dict,
    credibility: dict,
    validity: dict,
) -> None:
    """Persist a run's three scores, replacing any prior row atomically."""
    with conn:
        conn.execute("DELETE FROM run_scores WHERE run_id = ?", (run_id,))
        conn.execute(
            "INSERT INTO run_scores(run_id, accuracy, credibility, validity, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                run_id,
                json.dumps(accuracy),
                json.dumps(credibility),
                json.dumps(validity),
                now_iso(),
            ),
        )


def get_run_scores(conn: sqlite3.Connection, run_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM run_scores WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    return {
        "accuracy": json.loads(row["accuracy"]),
        "credibility": json.loads(row["credibility"]),
        "validity": json.loads(row["validity"]),
        "created_at": row["created_at"],
    }


def save_source_credibility(conn: sqlite3.Connection, run_id: str, rows: list[dict]) -> None:
    """Persist per-source credibility rows, replacing the run's prior set atomically."""
    with conn:
        conn.execute("DELETE FROM source_credibility WHERE run_id = ?", (run_id,))
        for row in rows:
            conn.execute(
                "INSERT INTO source_credibility(id, run_id, doc_id, metadata, components,"
                " total, tier, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id(),
                    run_id,
                    row["doc_id"],
                    json.dumps(row["metadata"]),
                    json.dumps(row["components"]),
                    row["total"],
                    row["tier"],
                    now_iso(),
                ),
            )


def list_source_credibility(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT s.*, d.title AS doc_title
        FROM source_credibility s JOIN documents d ON d.id = s.doc_id
        WHERE s.run_id = ? ORDER BY s.total DESC
        """,
        (run_id,),
    ).fetchall()
    out = []
    for row in rows:
        record = dict(row)
        record["metadata"] = json.loads(record["metadata"])
        record["components"] = json.loads(record["components"])
        out.append(record)
    return out


def list_claims(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM claims WHERE run_id = ? ORDER BY page, id", (run_id,)
    ).fetchall()
    return [dict(row) for row in rows]


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
    doc = conn.execute("SELECT kind FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if doc is None:
        raise ValueError(f"Unknown document {doc_id!r} — add the document before its chunks")
    doc_kind = doc["kind"]
    chunk_ids: list[int] = []
    with conn:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            cursor = conn.execute(
                "INSERT INTO chunks(run_id, doc_id, page, section, kind, text, figure_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    doc_id,
                    chunk.get("page"),
                    chunk.get("section"),
                    chunk.get("kind", "text"),
                    chunk["text"],
                    chunk.get("figure_id"),
                ),
            )
            chunk_id = cursor.lastrowid
            conn.execute(
                "INSERT INTO chunks_vec(chunk_id, run_id, doc_kind, embedding) VALUES (?, ?, ?, ?)",
                (chunk_id, run_id, doc_kind, serialize_float32(normalize(embedding))),
            )
            chunk_ids.append(chunk_id)
    return chunk_ids


def list_chunks_by_kind(conn: sqlite3.Connection, doc_id: str, kind: str) -> list[dict]:
    """A document's chunks of one kind, in document order (the rowid is that order)."""
    rows = conn.execute(
        "SELECT page, section, text FROM chunks WHERE doc_id = ? AND kind = ? ORDER BY id",
        (doc_id, kind),
    ).fetchall()
    return [dict(row) for row in rows]


def get_chunks(conn: sqlite3.Connection, chunk_ids: list[int]) -> dict[int, dict]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = conn.execute(f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids).fetchall()
    return {row["id"]: dict(row) for row in rows}
