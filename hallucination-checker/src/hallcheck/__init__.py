"""Hallucination checker package."""

from __future__ import annotations

import sys
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(_MODULE_ROOT), str(_REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

__all__ = [
    "config",
    "db",
    "models",
    "ingest_pdf",
    "chunking",
    "embeddings",
    "extract_claims",
    "tables",
    "gpt",
    "retrieval",
    "verify",
    "cli",
]
