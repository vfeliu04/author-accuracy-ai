"""SQLite helpers for persistence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from ..config import get_settings
from ..services.logger import setup_logger


logger = setup_logger(__name__)


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
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
            metadata TEXT,
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
        """
    )
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
            (doc_id, doc_type, path, json.dumps(metadata), body_text, created_at),
        )

    def update_document_metadata(self, doc_id: str, updates: dict):
        doc = self.get_document(doc_id)
        if not doc:
            return
        metadata = doc.get("metadata") or {}
        metadata.update({k: v for k, v in updates.items() if v is not None})
        self.execute(
            "UPDATE documents SET metadata = ? WHERE doc_id = ?",
            (json.dumps(metadata), doc_id),
        )

    def insert_chunks(self, chunks: list[dict]):
        rows = [
            (
                chunk["chunk_id"],
                chunk["doc_id"],
                chunk["text"],
                chunk.get("page_start"),
                chunk.get("page_end"),
                json.dumps(chunk.get("metadata") or {}),
            )
            for chunk in chunks
        ]
        self.executemany(
            """
            INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, page_start, page_end, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
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
                json.dumps(claim.get("metadata") or {}),
            )
            for claim in claims
        ]
        self.executemany(
            """
            INSERT OR REPLACE INTO claims (claim_id, report_id, text, verdict, confidence, confidence_band, explanation, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(e.get("metadata") or {}),
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

    def list_claims_by_report(self, report_id: str) -> list[dict]:
        rows = self.fetchall(
            "SELECT * FROM claims WHERE report_id = ? ORDER BY rowid",
            (report_id,),
        )
        return [dict(row) | {"metadata": json.loads(row["metadata"] or "{}")}
                for row in rows]

    def list_chunks(self, doc_id: Optional[str] = None) -> list[dict]:
        rows = self.fetchall(
            "SELECT * FROM chunks WHERE (? IS NULL OR doc_id = ?)",
            (doc_id, doc_id),
        )
        return [dict(row) | {"metadata": json.loads(row["metadata"] or "{}")}
                for row in rows]

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

    def get_claims_for_source(self, source_id: str) -> list[dict]:
        rows = self.fetchall(
            """
            SELECT c.* FROM claim_evidence e
            JOIN claims c ON c.claim_id = e.claim_id
            WHERE e.source_id = ?
            ORDER BY c.rowid
            """,
            (source_id,),
        )
        return [dict(row) | {"metadata": json.loads(row["metadata"] or "{}")}
                for row in rows]

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
        allowed = {"status", "result_json", "error_message", "updated_at"}
        columns = []
        values = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            columns.append(f"{key} = ?")
            if key == "result_json" and not isinstance(value, str):
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
        return result
