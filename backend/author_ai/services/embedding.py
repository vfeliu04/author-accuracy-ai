"""
Wrapper around the embedding provider so pipelines can stay provider agnostic.
"""

from __future__ import annotations

from typing import Iterable, List

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency for now
    OpenAI = None  # type: ignore

from ..config import get_settings
from .logger import setup_logger


logger = setup_logger(__name__)


class EmbeddingService:
    def __init__(self):
        self.settings = get_settings()
        if OpenAI and self.settings.openai_api_key:
            self.client = OpenAI(api_key=self.settings.openai_api_key)
        else:
            self.client = None
            logger.warning("OpenAI client not configured; embeddings will use deterministic fallback.")

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        texts = [t.strip() for t in texts]
        if not texts:
            return []

        if not self.client:
            logger.warning("Using fallback embedding; set OPENAI_API_KEY for production use.")
            return [self._fallback_embedding(text) for text in texts]

        response = self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        logger.debug("Generated %d embeddings via %s", len(vectors), self.settings.embedding_model)
        return vectors

    @staticmethod
    def _fallback_embedding(text: str, dim: int = 64) -> List[float]:
        vec = [0.0] * dim
        for idx, char in enumerate(text.lower()):
            bucket = ord(char) % dim
            vec[bucket] += 1
        norm = max(1.0, sum(value * value for value in vec) ** 0.5)
        return [value / norm for value in vec]


_EMBEDDINGS = EmbeddingService()


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    """Convenience function for callers that do not want to instantiate the service."""

    return _EMBEDDINGS.embed(texts)


# Extend EmbeddingService to batch requests when using OpenAI to avoid oversized payloads.
def _batch(iterable, size: int):
    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# Wrap embed_texts with batching for safety when OpenAI is enabled.
def embed_texts(texts: Iterable[str], batch_size: int = 64) -> List[List[float]]:  # type: ignore[override]
    texts = list(texts)
    if not texts:
        return []
    if not _EMBEDDINGS.client:
        return _EMBEDDINGS.embed(texts)

    vectors: list[list[float]] = []
    for chunk in _batch(texts, batch_size):
        vectors.extend(_EMBEDDINGS.embed(chunk))
    return vectors
