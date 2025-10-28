"""Hybrid retrieval orchestrator used in Stage B.

The pipeline wires together three signals:
    1. BM25 lexical similarity
    2. Dense (cosine) similarity
    3. Numeric overlap score

Table lines that look like delimited rows are also surfaced as pseudo-sentences
so Stage C can consider them when making a decision.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from author_ai.config import StageBConfig
from author_ai.models import Claim, EvidenceSpan
from author_ai.retrieval.bm25 import BM25Retriever
from author_ai.retrieval.dense import DenseRetriever
from author_ai.retrieval.numbers import NumberIndex
from author_ai.retrieval.tables import TableExtractor, TableRow


SNIPPET_LENGTH = 400
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _truncate(text: str, max_length: int = SNIPPET_LENGTH) -> str:
    """Trim long spans so the judge prompt stays small."""

    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _claim_values(claim: Claim) -> list[float]:
    """Promote all numeric hints (values + canonical) to a flat list."""

    numbers: list[float] = []
    if claim.values:
        for entry in claim.values:
            value = entry.get("value")
            if isinstance(value, (int, float)):
                numbers.append(float(value))
    if claim.canonical and isinstance(claim.canonical.get("value_norm"), (int, float)):
        numbers.append(float(claim.canonical["value_norm"]))
    return numbers


@dataclass
class RetrievalBundle:
    claim_id: str
    evidence: list[EvidenceSpan]


class HybridRetrievalPipeline:
    """Combines lexical, dense, numeric and table heuristics."""

    def __init__(self, config: StageBConfig) -> None:
        self.config = config
        self._bm25 = BM25Retriever()
        self._dense = DenseRetriever()
        self._numbers = NumberIndex()
        self._tables = TableExtractor()
        self._documents: dict[str, str] = {}
        self._table_rows: dict[str, list[TableRow]] = {}

    def index_corpus(self, corpus: Dict[str, str]) -> None:
        """Load a map of `{doc_id: text}` and build all indices."""

        trimmed = {
            doc_id: content[: self.config.max_corpus_chars] for doc_id, content in corpus.items()
        }
        self._documents = trimmed
        self._bm25.index(trimmed.items())
        self._dense.index(trimmed.items())
        self._numbers.index(trimmed.items())
        self._table_rows = {
            doc_id: self._tables.extract(doc_id, content) for doc_id, content in trimmed.items()
        }

    def load_directory(self, sources_dir: Path) -> None:
        """Helper for CLI/tests: read a directory of txt/md/json files into memory."""

        corpus: dict[str, str] = {}
        for path in sorted(sources_dir.glob("**/*")):
            if path.is_dir():
                continue
            if path.suffix.lower() not in {".txt", ".md", ".json"}:
                continue
            content = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".json":
                # Flatten JSON structures; this keeps our tests deterministic.
                try:
                    content = json.dumps(json.loads(content), ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    pass
            corpus[path.stem] = content
        self.index_corpus(corpus)

    def retrieve(self, claim: Claim) -> List[EvidenceSpan]:
        """Return a ranked list of evidence spans for the given claim."""

        if not self._documents:
            return []

        bm25_hits = self._bm25.search(claim.text, k=self.config.top_k_text * 2)
        dense_hits = self._dense.search(claim.text, k=self.config.top_k_text * 2)
        numeric_targets = _claim_values(claim)
        number_hits = self._numbers.search(numeric_targets, top_k=5)

        combined_scores: dict[str, dict[str, float]] = {}
        for hit in bm25_hits:
            combined_scores.setdefault(hit.doc_id, {})["bm25"] = hit.score
        for hit in dense_hits:
            entry = combined_scores.setdefault(hit.doc_id, {})
            entry["dense"] = max(hit.score, entry.get("dense", 0.0))
        for match in number_hits:
            entry = combined_scores.setdefault(match.doc_id, {})
            entry["num_match"] = float(match.count)
            entry["num_distance"] = float(match.distance)

        scored_docs: list[tuple[str, float]] = []
        for doc_id, score_parts in combined_scores.items():
            bm25_score = score_parts.get("bm25", 0.0)
            dense_score = score_parts.get("dense", 0.0)
            number_bonus = 0.0
            if score_parts.get("num_distance") is not None:
                # Use a smooth version of "smaller distance is better".
                number_bonus = 1.0 / (1.0 + score_parts["num_distance"])
            score = (
                self.config.bm25_weight * bm25_score
                + self.config.dense_weight * dense_score
                + self.config.number_weight * number_bonus
            )
            scored_docs.append((doc_id, score))

        scored_docs.sort(key=lambda item: item[1], reverse=True)
        top_docs = [doc_id for doc_id, _ in scored_docs[: self.config.top_k_text]]

        evidence_spans: list[EvidenceSpan] = []
        for doc_id in top_docs:
            doc_text = self._documents[doc_id]
            snippet = self._best_snippet(doc_text, claim.text)
            scores = combined_scores.get(doc_id, {})
            evidence_spans.append(
                EvidenceSpan(
                    doc_id=doc_id,
                    content=snippet,
                    provenance={"section": None, "table": None},
                    scores={
                        "bm25": scores.get("bm25"),
                        "dense": scores.get("dense"),
                        "num_match": scores.get("num_match"),
                    },
                )
            )

        # Add table rows that include the target numbers.
        for row in self._table_rows_for_claim(claim, numeric_targets)[: self.config.top_k_tables]:
            evidence_spans.append(
                EvidenceSpan(
                    doc_id=row.doc_id,
                    content=row.content,
                    provenance={"section": None, "table": row.row_id},
                    scores={"bm25": None, "dense": None, "num_match": None},
                )
            )

        return evidence_spans

    def _table_rows_for_claim(self, claim: Claim, numeric_targets: Sequence[float]) -> list[TableRow]:
        """Surface table rows that contain one of the claim's numbers."""

        rows: list[TableRow] = []
        if not numeric_targets:
            return rows

        string_targets = {f"{value:.2f}".rstrip("0").rstrip(".") for value in numeric_targets}
        for doc_rows in self._table_rows.values():
            for row in doc_rows:
                numbers = set(re.findall(r"-?\d+(?:\.\d+)?", row.content))
                if string_targets & numbers:
                    rows.append(row)
        return rows

    def _best_snippet(self, document: str, query: str) -> str:
        """Pick a short snippet that maximises token overlap with the query."""

        sentences = SENTENCE_SPLIT.split(document)
        if not sentences:
            return _truncate(document)

        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not query_tokens:
            return _truncate(sentences[0])

        best_sentence = sentences[0]
        best_score = -math.inf
        for sentence in sentences:
            sentence_tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            overlap = len(query_tokens & sentence_tokens)
            if overlap > best_score:
                best_sentence = sentence
                best_score = overlap
        return _truncate(best_sentence)


__all__ = ["HybridRetrievalPipeline", "RetrievalBundle"]
