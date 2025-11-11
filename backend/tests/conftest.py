import os
import pytest

from author_ai.config import get_settings


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    base = tmp_path
    monkeypatch.setenv("DATA_ROOT", str(base / "data"))
    monkeypatch.setenv("SQLITE_PATH", str(base / "data" / "accuracy.db"))
    monkeypatch.setenv("FAISS_INDEX_DIR", str(base / "indexes"))
    monkeypatch.setenv("CACHE_DIR", str(base / "cache"))
    monkeypatch.setenv("API_KEY", "test-key")
    get_settings.cache_clear()
    settings = get_settings()
    yield settings
    get_settings.cache_clear()
