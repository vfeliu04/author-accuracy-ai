import os
import pytest
from pathlib import Path
import logging

from author_ai.config import get_settings

# Ensure environment variables from the project .env are loaded for tests.
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


@pytest.fixture(scope="session", autouse=True)
def enable_debug_logging():
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("author_ai").setLevel(logging.DEBUG)


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    base = tmp_path
    monkeypatch.setenv("DATA_ROOT", str(base / "data"))
    monkeypatch.setenv("SQLITE_PATH", str(base / "data" / "accuracy.db"))
    monkeypatch.setenv("FAISS_INDEX_DIR", str(base / "indexes"))
    monkeypatch.setenv("CACHE_DIR", str(base / "cache"))
    monkeypatch.setenv("CLAIM_VECTOR_PATH", str(base / "claim_indexes"))
    monkeypatch.setenv("SOURCE_VECTOR_PATH", str(base / "source_indexes"))
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("RETRIEVAL_SUPPORT_THRESHOLD", "0.35")
    get_settings.cache_clear()
    settings = get_settings()
    yield settings
    get_settings.cache_clear()
