from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pytest

from mvp_rag.app import FALLBACK_ANSWER, create_app, get_llm_client, get_vector_store


class FakeScoredPoint:
    def __init__(self, payload: Dict[str, object]):
        self.payload = payload


class FakeVectorStore:
    def __init__(self) -> None:
        self._dedup: set[Tuple[str, int, str]] = set()
        self._payloads: List[Dict[str, object]] = []

    def upsert_chunks(
        self, doc_title: str, chunks: Sequence[Dict[str, int | str]], embeddings: Sequence[Sequence[float]]
    ) -> Tuple[int, int]:
        inserted = 0
        skipped = 0
        for chunk in chunks:
            key = (doc_title, int(chunk["page_start"]), str(chunk["chunk_id"]))
            if key in self._dedup:
                skipped += 1
                continue
            payload = {
                "doc_title": doc_title,
                "page_start": int(chunk["page_start"]),
                "page_end": int(chunk["page_end"]),
                "chunk_text": str(chunk["chunk_text"]),
                "chunk_id": str(chunk["chunk_id"]),
                "token_count": int(chunk["token_count"]),
            }
            self._dedup.add(key)
            self._payloads.append(payload)
            inserted += 1
        return inserted, skipped

    def query(self, embedding: Sequence[float], top_k: int = 4) -> List[FakeScoredPoint]:
        payloads = self._payloads[:top_k]
        return [FakeScoredPoint(payload) for payload in payloads]

    def clear(self) -> None:
        self._payloads.clear()
        self._dedup.clear()


class FakeLLMClient:
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, query: str) -> List[float]:
        return [float(len(query))]

    def ask_llm(self, context_blocks: Sequence[str], question: str) -> str:
        if not context_blocks:
            return FALLBACK_ANSWER
        return f"{question} :: answered from context."


@pytest.fixture(name="client_with_stubs")
def fixture_client_with_stubs(monkeypatch):
    get_vector_store.cache_clear()
    get_llm_client.cache_clear()
    fake_store = FakeVectorStore()
    fake_llm = FakeLLMClient()

    def _get_vector_store(app):  # type: ignore[override]
        return fake_store

    def _get_llm_client(app):  # type: ignore[override]
        return fake_llm

    monkeypatch.setattr("mvp_rag.app.get_vector_store", _get_vector_store)
    monkeypatch.setattr("mvp_rag.app.get_llm_client", _get_llm_client)

    app = create_app(testing=True)
    app.config["FAKE_VECTOR_STORE"] = fake_store
    with app.test_client() as client:
        yield client, fake_store


@pytest.fixture
def dummy_pdf_bytes() -> bytes:
    return Path("tests/fixtures/dummy.pdf").read_bytes()
