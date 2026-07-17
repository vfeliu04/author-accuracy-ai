"""SQLite helpers for persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from ..config import get_settings
from ..services.logger import setup_logger


logger = setup_logger(__name__)

_thread_local = threading.local()

_METADATA_SCHEMA_VERSION = 1


def _stamp_metadata(metadata: dict) -> dict:
    """Return a copy of *metadata* with schema_version set, for migration detection."""
    stamped = dict(metadata)
    stamped.setdefault("schema_version", _METADATA_SCHEMA_VERSION)
    return stamped


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    conn = getattr(_thread_local, "connection", None)
    if conn is not None:
        try:
            conn.cursor()  # cheap liveness check
        except sqlite3.ProgrammingError:
            conn = None  # closed — create a fresh one
    if conn is None:
        settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(settings.sqlite_path)
        conn.row_factory = sqlite3.Row
        _thread_local.connection = conn
    return conn


def init_db() -> None:
    conn = _connect()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            doc_type TEXT NOT NULL,
            path TEXT NOT NULL,
            metadata TEXT,
            body_text TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            text TEXT NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            chunk_type TEXT,
            chart_id TEXT,
            x_value TEXT,
            y_value REAL,
            series_name TEXT,
            metadata TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        );

        CREATE TABLE IF NOT EXISTS charts (
            id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            page INTEGER NOT NULL,
            figure_label TEXT,
            bbox_json TEXT,
            chart_type TEXT,
            raw_json TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        );

        CREATE TABLE IF NOT EXISTS claims (
            claim_id TEXT PRIMARY KEY,
            report_id TEXT NOT NULL,
            text TEXT NOT NULL,
            verdict TEXT,
            confidence REAL,
            confidence_band TEXT,
            explanation TEXT,
            processing_mode TEXT,
            metadata TEXT,
            FOREIGN KEY (report_id) REFERENCES documents(doc_id)
        );

        CREATE TABLE IF NOT EXISTS claim_evidence (
            evidence_id TEXT PRIMARY KEY,
            claim_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            chunk_id TEXT,
            verdict_label TEXT,
            metadata TEXT,
            FOREIGN KEY (claim_id) REFERENCES claims(claim_id),
            FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
        );

        CREATE TABLE IF NOT EXISTS credibility_scores (
            source_id TEXT PRIMARY KEY,
            score REAL NOT NULL,
            metadata_confidence TEXT,
            components TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS validity_scores (
            report_id TEXT PRIMARY KEY,
            overall REAL,
            coverage REAL,
            consistency REAL,
            methodology REAL,
            context REAL,
            recency REAL,
            diagnostics TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            report_id TEXT,
            role TEXT,
            message TEXT,
            timestamp TEXT,
            context_ids TEXT
        );

        CREATE TABLE IF NOT EXISTS uploads (
            upload_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            path TEXT NOT NULL,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            report_id TEXT,
            source_ids TEXT,
            result_json TEXT,
            error_message TEXT,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents (doc_id);
        CREATE INDEX IF NOT EXISTS idx_claims_report_id ON claims (report_id);
        CREATE INDEX IF NOT EXISTS idx_credibility_scores_source_id ON credibility_scores (source_id);
        CREATE INDEX IF NOT EXISTS idx_chat_logs_session_id ON chat_logs (session_id);
        CREATE INDEX IF NOT EXISTS idx_chat_logs_report_id ON chat_logs (report_id);
        """
    )
    # Add processing_mode column if missing (for existing DBs)
    cur.execute("PRAGMA table_info(claims);")
    cols = [row[1] for row in cur.fetchall()]
    if "processing_mode" not in cols:
        try:
            cur.execute("ALTER TABLE claims ADD COLUMN processing_mode TEXT;")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc):
                raise
    cur.execute("PRAGMA table_info(chunks);")
    chunk_cols = [row[1] for row in cur.fetchall()]
    chunk_alters = {
        "chunk_type": "ALTER TABLE chunks ADD COLUMN chunk_type TEXT;",
        "chart_id": "ALTER TABLE chunks ADD COLUMN chart_id TEXT;",
        "x_value": "ALTER TABLE chunks ADD COLUMN x_value TEXT;",
        "y_value": "ALTER TABLE chunks ADD COLUMN y_value REAL;",
        "series_name": "ALTER TABLE chunks ADD COLUMN series_name TEXT;",
    }
    for col_name, stmt in chunk_alters.items():
        if col_name not in chunk_cols:
            try:
                cur.execute(stmt)
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc):
                    raise
    # Add progress_json column to jobs if missing (existing DBs)
    cur.execute("PRAGMA table_info(jobs);")
    job_cols = [row[1] for row in cur.fetchall()]
    if "progress_json" not in job_cols:
        try:
            cur.execute("ALTER TABLE jobs ADD COLUMN progress_json TEXT;")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc):
                raise
    conn.commit()
    conn.close()


class Repository:
    def __init__(self):
        init_db()
        self.settings = get_settings()

    def execute(self, query: str, params: Iterable[Any] | Mapping[str, Any] = ()):  # type: ignore[override]
        with _connect() as conn:
            conn.execute(query, params)
            conn.commit()

    def executemany(self, query: str, rows):
        with _connect() as conn:
            conn.executemany(query, rows)
            conn.commit()

    def fetchall(self, query: str, params=()) -> list[sqlite3.Row]:
        with _connect() as conn:
            cur = conn.execute(query, params)
            return cur.fetchall()

    def fetchone(self, query: str, params=()) -> Optional[sqlite3.Row]:
        with _connect() as conn:
            cur = conn.execute(query, params)
            return cur.fetchone()

    # Convenience helpers -------------------------------------------------

    def upsert_document(self, doc_id: str, doc_type: str, path: str, metadata: dict, body_text: str, created_at: str):
        self.execute(
            """
            INSERT INTO documents (doc_id, doc_type, path, metadata, body_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                doc_type=excluded.doc_type,
                path=excluded.path,
                metadata=excluded.metadata,
                body_text=excluded.body_text,
                created_at=excluded.created_at
            """,
            (doc_id, doc_type, path, json.dumps(_stamp_metadata(metadata)), body_text, created_at),
        )

    def update_document_metadata(self, doc_id: str, updates: dict):
        doc = self.get_document(doc_id)
        if not doc:
            return
        metadata = doc.get("metadata") or {}
        metadata.update({k: v for k, v in updates.items() if v is not None})
        self.execute(
            "UPDATE documents SET metadata = ? WHERE doc_id = ?",
            (json.dumps(_stamp_metadata(metadata)), doc_id),
        )

    def insert_chunks(self, chunks: list[dict]):
        rows = [
            (
                chunk["chunk_id"],
                chunk["doc_id"],
                chunk["text"],
                chunk.get("page_start"),
                chunk.get("page_end"),
                chunk.get("chunk_type"),
                chunk.get("chart_id"),
                chunk.get("x_value"),
                chunk.get("y_value"),
                chunk.get("series_name"),
                json.dumps(_stamp_metadata(chunk.get("metadata") or {})),
            )
            for chunk in chunks
        ]
        self.executemany(
            """
            INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, page_start, page_end, chunk_type, chart_id, x_value, y_value, series_name, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def insert_charts(self, charts: list[dict]):
        if not charts:
            return
        rows = [
            (
                chart["id"],
                chart["doc_id"],
                chart["page"],
                chart.get("figure_label"),
                json.dumps(chart.get("bbox") if isinstance(chart.get("bbox"), (list, tuple)) else chart.get("bbox")),
                chart.get("chart_type"),
                json.dumps(chart.get("raw_json") or {}),
            )
            for chart in charts
        ]
        self.executemany(
            """
            INSERT OR REPLACE INTO charts (id, doc_id, page, figure_label, bbox_json, chart_type, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def insert_claims(self, claims: list[dict]):
        rows = [
            (
                claim["claim_id"],
                claim["report_id"],
                claim["text"],
                claim["verdict"],
                claim["confidence"],
                claim["confidence_band"],
                claim["explanation"],
                claim.get("processing_mode"),
                json.dumps(_stamp_metadata(claim.get("metadata") or {})),
            )
            for claim in claims
        ]
        self.executemany(
            """
            INSERT OR REPLACE INTO claims (claim_id, report_id, text, verdict, confidence, confidence_band, explanation, processing_mode, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def insert_evidence(self, evidence: list[dict]):
        rows = [
            (
                e["evidence_id"],
                e["claim_id"],
                e["source_id"],
                e.get("chunk_id"),
                e.get("verdict_label"),
                json.dumps(_stamp_metadata(e.get("metadata") or {})),
            )
            for e in evidence
        ]
        self.executemany(
            """
            INSERT OR REPLACE INTO claim_evidence (evidence_id, claim_id, source_id, chunk_id, verdict_label, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def upsert_credibility(self, score: dict):
        self.execute(
            """
            INSERT INTO credibility_scores (source_id, score, metadata_confidence, components, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                score=excluded.score,
                metadata_confidence=excluded.metadata_confidence,
                components=excluded.components,
                updated_at=excluded.updated_at
            """,
            (
                score["source_id"],
                score["score"],
                score.get("metadata_confidence"),
                json.dumps(score.get("components") or {}),
                score.get("last_refreshed_at"),
            ),
        )

    def upsert_validity(self, score: dict):
        self.execute(
            """
            INSERT INTO validity_scores (report_id, overall, coverage, consistency, methodology, context, recency, diagnostics, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(report_id) DO UPDATE SET
                overall=excluded.overall,
                coverage=excluded.coverage,
                consistency=excluded.consistency,
                methodology=excluded.methodology,
                context=excluded.context,
                recency=excluded.recency,
                diagnostics=excluded.diagnostics,
                updated_at=excluded.updated_at
            """,
            (
                score["report_id"],
                score["overall"],
                score["coverage"],
                score["consistency"],
                score["methodology"],
                score["context"],
                score["recency"],
                json.dumps(score.get("diagnostics") or {}),
                score.get("last_calculated_at"),
            ),
        )

    def list_claims(self, report_id: Optional[str] = None) -> list[dict]:
        rows = self.fetchall(
            "SELECT * FROM claims WHERE (? IS NULL OR report_id = ?) ORDER BY rowid",
            (report_id, report_id),
        )
        return [dict(row) | {"metadata": json.loads(row["metadata"] or "{}")}
                for row in rows]

    def list_claims_by_report(self, report_id: str, limit: int | None = None, offset: int = 0) -> list[dict]:
        query = "SELECT * FROM claims WHERE report_id = ? ORDER BY rowid"
        params: list[Any] = [report_id]
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        rows = self.fetchall(query, tuple(params))
        return [dict(row) | {"metadata": json.loads(row["metadata"] or "{}")}
                for row in rows]

    def count_claims_by_report(self, report_id: str) -> int:
        row = self.fetchone("SELECT COUNT(*) as total FROM claims WHERE report_id = ?", (report_id,))
        return int(row["total"]) if row else 0

    def list_chunks(self, doc_id: Optional[str] = None) -> list[dict]:
        rows = self.fetchall(
            "SELECT * FROM chunks WHERE (? IS NULL OR doc_id = ?)",
            (doc_id, doc_id),
        )
        return [dict(row) | {"metadata": json.loads(row["metadata"] or "{}")}
                for row in rows]

    def list_charts(self, doc_id: Optional[str] = None) -> list[dict]:
        rows = self.fetchall(
            "SELECT * FROM charts WHERE (? IS NULL OR doc_id = ?)",
            (doc_id, doc_id),
        )
        charts = []
        for row in rows:
            entry = dict(row)
            try:
                entry["bbox"] = json.loads(entry.get("bbox_json") or "null")
            except json.JSONDecodeError:
                entry["bbox"] = None
            entry["raw_json"] = json.loads(entry.get("raw_json") or "{}")
            charts.append(entry)
        return charts

    def get_document(self, doc_id: str) -> Optional[dict]:
        row = self.fetchone("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        if not row:
            return None
        return dict(row) | {"metadata": json.loads(row["metadata"] or "{}")}

    def list_documents(self, doc_ids: list[str]) -> dict[str, dict]:
        if not doc_ids:
            return {}
        placeholders = ",".join(["?"] * len(doc_ids))
        rows = self.fetchall(
            f"SELECT * FROM documents WHERE doc_id IN ({placeholders})",
            tuple(doc_ids),
        )
        documents = {}
        for row in rows:
            documents[row["doc_id"]] = dict(row) | {"metadata": json.loads(row["metadata"] or "{}")}
        return documents

    def list_source_doc_ids(self) -> list[str]:
        """Return doc_ids for all documents with doc_type='SOURCE'."""
        rows = self.fetchall("SELECT doc_id FROM documents WHERE doc_type = 'SOURCE'")
        return [row["doc_id"] for row in rows]

    def list_evidence_for_claim(self, claim_id: str) -> list[dict]:
        """Return all evidence rows for a given claim_id."""
        rows = self.fetchall(
            "SELECT evidence_id, claim_id, source_id, chunk_id, verdict_label, metadata "
            "FROM claim_evidence WHERE claim_id = ? ORDER BY rowid",
            (claim_id,),
        )
        result = []
        for row in rows:
            meta = row["metadata"]
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            result.append({
                "evidence_id": row["evidence_id"],
                "claim_id": row["claim_id"],
                "source_id": row["source_id"],
                "chunk_id": row["chunk_id"],
                "verdict_label": row["verdict_label"],
                "metadata": meta,
            })
        return result

    def list_evidence_for_claims(self, claim_ids: list[str]) -> list[dict]:
        if not claim_ids:
            return []
        placeholders = ",".join(["?"] * len(claim_ids))
        rows = self.fetchall(
            f"""
            SELECT e.*, u.file_name
            FROM claim_evidence e
            LEFT JOIN uploads u ON u.upload_id = e.source_id
            WHERE e.claim_id IN ({placeholders})
            """,
            tuple(claim_ids),
        )
        return [
            dict(row) | {"metadata": json.loads(row["metadata"] or "{}")}
            for row in rows
        ]

    def record_chat_turn(self, turn: dict):
        self.execute(
            """
            INSERT INTO chat_logs (session_id, report_id, role, message, timestamp, context_ids)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                turn.get("session_id"),
                turn.get("report_id"),
                turn.get("role"),
                turn.get("message"),
                turn.get("timestamp"),
                json.dumps(turn.get("context_ids") or {}),
            ),
        )

    def get_chat_history(self, report_id: str, limit: Optional[int] = None) -> list[dict]:
        query = "SELECT * FROM chat_logs WHERE report_id = ? ORDER BY id DESC"
        params: tuple[Any, ...] = (report_id,)
        if limit:
            query += " LIMIT ?"
            params = (report_id, limit)
        rows = self.fetchall(query, params)
        rows.reverse()
        return [
            {
                "session_id": row["session_id"],
                "role": row["role"],
                "message": row["message"],
                "timestamp": row["timestamp"],
                "context_ids": json.loads(row["context_ids"] or "{}")
            }
            for row in rows
        ]

    def get_credibility(self, source_id: str) -> Optional[dict]:
        row = self.fetchone(
            "SELECT * FROM credibility_scores WHERE source_id = ?",
            (source_id,),
        )
        if not row:
            return None
        return dict(row) | {"components": json.loads(row["components"] or "{}")}

    def get_validity(self, report_id: str) -> Optional[dict]:
        row = self.fetchone(
            "SELECT * FROM validity_scores WHERE report_id = ?",
            (report_id,),
        )
        if not row:
            return None
        return dict(row) | {"diagnostics": json.loads(row["diagnostics"] or "{}")}

    def source_usage(self, report_id: str) -> list[dict]:
        rows = self.fetchall(
            """
            SELECT e.source_id, COUNT(*) AS usage_count
            FROM claim_evidence e
            JOIN claims c ON c.claim_id = e.claim_id
            WHERE c.report_id = ? AND e.verdict_label = 'SUPPORTED'
            GROUP BY e.source_id
            """,
            (report_id,),
        )
        return [dict(row) for row in rows]

    def get_claims_for_source(self, source_id: str, limit: int | None = None, offset: int = 0) -> list[dict]:
        query = (
            "SELECT DISTINCT c.* FROM claim_evidence e "
            "JOIN claims c ON c.claim_id = e.claim_id "
            "WHERE e.source_id = ? ORDER BY c.rowid"
        )
        params: list[Any] = [source_id]
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        rows = self.fetchall(query, tuple(params))
        return [dict(row) | {"metadata": json.loads(row["metadata"] or "{}")}
                for row in rows]

    def count_claims_for_source(self, source_id: str) -> int:
        row = self.fetchone(
            "SELECT COUNT(DISTINCT claim_id) as total FROM claim_evidence WHERE source_id = ?",
            (source_id,),
        )
        return int(row["total"]) if row else 0

    # Upload helpers -----------------------------------------------------

    def add_upload(self, upload: dict):
        self.execute(
            """
            INSERT INTO uploads (upload_id, file_name, file_type, path, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(upload_id) DO UPDATE SET
                file_name=excluded.file_name,
                file_type=excluded.file_type,
                path=excluded.path,
                created_at=excluded.created_at
            """,
            (
                upload["upload_id"],
                upload["file_name"],
                upload["file_type"],
                upload["path"],
                upload.get("created_at"),
            ),
        )

    def get_upload(self, upload_id: str) -> Optional[dict]:
        row = self.fetchone("SELECT * FROM uploads WHERE upload_id = ?", (upload_id,))
        return dict(row) if row else None

    def list_uploads(self, file_type: Optional[str] = None) -> list[dict]:
        rows = self.fetchall(
            "SELECT * FROM uploads WHERE (? IS NULL OR file_type = ?) ORDER BY created_at DESC",
            (file_type, file_type),
        )
        return [dict(row) for row in rows]

    def delete_upload(self, upload_id: str):
        self.execute("DELETE FROM uploads WHERE upload_id = ?", (upload_id,))

    # Job helpers ---------------------------------------------------------

    def create_job(self, job: dict):
        self.execute(
            """
            INSERT INTO jobs (job_id, status, report_id, source_ids, result_json, error_message, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job["job_id"],
                job["status"],
                job.get("report_id"),
                json.dumps(job.get("source_ids") or []),
                json.dumps(job.get("result") or {}),
                job.get("error_message"),
                job.get("created_at"),
                job.get("updated_at"),
            ),
        )

    def update_job(self, job_id: str, **fields):
        allowed = {"status", "result_json", "error_message", "updated_at", "progress_json"}
        columns = []
        values = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            columns.append(f"{key} = ?")
            if key in {"result_json", "progress_json"} and not isinstance(value, str):
                values.append(json.dumps(value))
            else:
                values.append(value)
        if not columns:
            return
        values.append(job_id)
        self.execute(
            f"UPDATE jobs SET {', '.join(columns)} WHERE job_id = ?",
            tuple(values),
        )

    def push_job_progress(self, job_id: str, step: str, label: str, status: str = "done") -> None:
        """Append or update a progress step on the job's progress_json array."""
        import datetime
        row = self.fetchone("SELECT progress_json FROM jobs WHERE job_id = ?", (job_id,))
        if row is None:
            return
        entries: list = json.loads(row["progress_json"] or "[]")
        ts = datetime.datetime.utcnow().isoformat()
        for entry in entries:
            if entry["step"] == step:
                entry["status"] = status
                entry["label"] = label
                entry["ts"] = ts
                break
        else:
            entries.append({"step": step, "label": label, "status": status, "ts": ts})
        self.execute(
            "UPDATE jobs SET progress_json = ?, updated_at = ? WHERE job_id = ?",
            (json.dumps(entries), ts, job_id),
        )

    def get_job(self, job_id: str) -> Optional[dict]:
        row = self.fetchone("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        if not row:
            return None
        return self._row_to_job(row)

    def get_latest_job(self) -> Optional[dict]:
        row = self.fetchone(
            "SELECT * FROM jobs WHERE status = 'DONE' ORDER BY updated_at DESC LIMIT 1"
        )
        if not row:
            return None
        return self._row_to_job(row)

    def get_latest_job_for_report(self, report_id: str) -> Optional[dict]:
        row = self.fetchone(
            """
            SELECT * FROM jobs
            WHERE report_id = ? AND status = 'DONE'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (report_id,),
        )
        if not row:
            return None
        return self._row_to_job(row)

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["source_ids"] = json.loads(result.get("source_ids") or "[]")
        result["result_json"] = json.loads(result.get("result_json") or "{}")
        result["progress_json"] = json.loads(result.get("progress_json") or "[]")
        return result
