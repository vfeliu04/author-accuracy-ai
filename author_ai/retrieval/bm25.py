"""Small BM25-style retriever implemented in pure Python.

The goal is to offer a deterministic lexical scoring component that works in the
unit-test environment without depending on external libraries such as
`rank_bm25`.  It uses a straightforward token counter with inverse document
frequency weighting and produces the familiar BM25 score.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List


TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokeniser that keeps the score deterministic."""

    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class ScoredDocument:
    """Convenience container returned by `search`."""

    doc_id: str
    content: str
    score: float


class BM25Retriever:
    """Simple and fully in-memory BM25 implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._documents: dict[str, str] = {}
        self._term_freqs: dict[str, Counter[str]] = defaultdict(Counter)
        self._doc_lengths: dict[str, int] = {}
        self._index: dict[str, set[str]] = defaultdict(set)
        self._avg_doc_len: float = 0.0

    def index(self, docs: Iterable[tuple[str, str]]) -> None:
        """Index a collection of `(doc_id, content)` pairs."""

        docs_list = list(docs)
        if not docs_list:
            self._avg_doc_len = 0.0
            return

        total_len = 0
        for doc_id, content in docs_list:
            tokens = _tokenize(content)
            self._documents[doc_id] = content
            self._doc_lengths[doc_id] = len(tokens)
            total_len += len(tokens)

            counts = Counter(tokens)
            self._term_freqs[doc_id] = counts
            for token in counts:
                self._index[token].add(doc_id)

        self._avg_doc_len = total_len / max(len(docs_list), 1)

    def search(self, query: str, k: int = 5) -> List[ScoredDocument]:
        """Return the top-k documents ordered by BM25 score."""

        query_tokens = _tokenize(query)
        doc_scores: Counter[str] = Counter()
        for token in query_tokens:
            doc_ids = self._index.get(token)
            if not doc_ids:
                continue
            for doc_id in doc_ids:
                doc_scores[doc_id] += self._score_term(doc_id, token)

        scored = [
            ScoredDocument(doc_id=doc_id, content=self._documents[doc_id], score=score)
            for doc_id, score in doc_scores.items()
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]

    def _score_term(self, doc_id: str, term: str) -> float:
        """Classic BM25 term score with document length normalisation."""

        tf = self._term_freqs[doc_id][term]
        if tf == 0:
            return 0.0

        doc_len = self._doc_lengths[doc_id]
        idf = self._idf(term)
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1))
        return idf * (numerator / max(denominator, 1e-9))

    def _idf(self, term: str) -> float:
        """Compute inverse document frequency with the BM25+ smoothing trick."""

        doc_ids = self._index.get(term)
        if not doc_ids:
            return 0.0
        document_count = max(len(self._documents), 1)
        return math.log((document_count - len(doc_ids) + 0.5) / (len(doc_ids) + 0.5) + 1.0)


__all__ = ["BM25Retriever", "ScoredDocument"]
