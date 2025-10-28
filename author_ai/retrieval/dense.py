"""Deterministic dense retriever stand-in.

Real systems would use sentence transformers + FAISS or similar.  For offline
testing we emulate this behaviour by turning each document into a hashed term
vector and computing cosine similarity.  The interface mirrors what an actual
embedding store would expose so the calling code can stay unchanged later on.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List


TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class DenseResult:
    doc_id: str
    content: str
    score: float


class DenseRetriever:
    """Small cosine-similarity helper used as our dense retriever."""

    def __init__(self) -> None:
        self._vectors: dict[str, dict[str, float]] = {}
        self._documents: dict[str, str] = {}

    def index(self, docs: Iterable[tuple[str, str]]) -> None:
        """Turn each document into a unit-length TF vector."""

        for doc_id, content in docs:
            tokens = _tokenize(content)
            term_counts: dict[str, float] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0.0) + 1.0
            norm = math.sqrt(sum(value * value for value in term_counts.values())) or 1.0
            self._vectors[doc_id] = {token: value / norm for token, value in term_counts.items()}
            self._documents[doc_id] = content

    def search(self, query: str, k: int = 5) -> List[DenseResult]:
        """Return highest-similarity documents for the given query string."""

        query_tokens = _tokenize(query)
        query_counts: dict[str, float] = {}
        for token in query_tokens:
            query_counts[token] = query_counts.get(token, 0.0) + 1.0
        norm = math.sqrt(sum(value * value for value in query_counts.values())) or 1.0
        query_vec = {token: value / norm for token, value in query_counts.items()}

        results: list[DenseResult] = []
        for doc_id, vector in self._vectors.items():
            score = self._dot(vector, query_vec)
            if score <= 0:
                continue
            results.append(DenseResult(doc_id=doc_id, content=self._documents[doc_id], score=score))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:k]

    @staticmethod
    def _dot(left: dict[str, float], right: dict[str, float]) -> float:
        """Sparse dot-product that allows using the smaller input as the loop."""

        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(token, 0.0) for token, value in left.items())


__all__ = ["DenseRetriever", "DenseResult"]
