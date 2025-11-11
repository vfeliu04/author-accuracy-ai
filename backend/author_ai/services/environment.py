"""Utilities for resetting the local verification environment."""

from __future__ import annotations

import shutil
from pathlib import Path

from ..config import get_settings
from ..storage.database import init_db
from .logger import setup_logger
import sqlite3


logger = setup_logger(__name__)


def _clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _wipe_pipeline_tables(db_path: Path) -> None:
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = OFF;")
        tables = [
            "documents",
            "chunks",
            "claims",
            "claim_evidence",
            "credibility_scores",
            "validity_scores",
            "chat_logs",
        ]
        for table in tables:
            conn.execute(f"DELETE FROM {table}")
        conn.commit()
        logger.info("Cleared pipeline tables in %s", db_path)
    finally:
        conn.close()


def reset_environment() -> None:
    """Clear pipeline state (documents, claims, indexes, cache) before a fresh run."""

    settings = get_settings()

    if settings.sqlite_path.exists():
        _wipe_pipeline_tables(settings.sqlite_path)
    else:
        init_db()

    _clean_directory(settings.faiss_index_dir)
    _clean_directory(settings.cache_dir)

    logger.info("Environment reset complete; pipeline state cleared.")
