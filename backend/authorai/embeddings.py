"""Text embedding providers.

`OpenAIEmbedder` is the real provider and refuses to construct without an API
key — a missing key must never silently degrade into meaningless vectors.
`FakeEmbedder` is for tests: deterministic, tiny dimensions.
"""

import zlib
from typing import Protocol

import numpy as np

BATCH_SIZE = 64


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def normalize(vector: list[float]) -> list[float]:
    """L2-normalize so L2 distance ordering equals cosine ordering."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector")
    return (arr / norm).tolist()


class OpenAIEmbedder:
    def __init__(self, api_key: str | None, model: str, dim: int):
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Embeddings require a real provider — "
                "refusing to start rather than producing meaningless vectors."
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = self._client.embeddings.create(
                model=self.model, input=batch, dimensions=self.dim
            )
            out.extend(item.embedding for item in response.data)
        return out


class FakeEmbedder:
    """Deterministic embedder for tests.

    Known texts map to fixed vectors via `mapping`; unknown texts get a
    reproducible pseudo-random vector seeded by a stable checksum of the text
    (not `hash()`, which is salted per process).
    """

    def __init__(self, dim: int = 8, mapping: dict[str, list[float]] | None = None):
        self.dim = dim
        # Same attribute the real embedder exposes — ingest stamps it onto
        # documents.embedding_model (the dedup donor filter).
        self.model = "fake-embedder"
        self._mapping = mapping or {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._mapping.get(text, self._pseudo(text)) for text in texts]

    def _pseudo(self, text: str) -> list[float]:
        rng = np.random.default_rng(zlib.crc32(text.encode()))
        return normalize(rng.normal(size=self.dim).tolist())
