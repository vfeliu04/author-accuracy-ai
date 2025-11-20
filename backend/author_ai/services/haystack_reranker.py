"""
Haystack ranker to refine retrieval hits before verdicting.
"""

from __future__ import annotations

from typing import List, Dict, Any

try:
    from haystack.schema import Document
    from haystack.nodes import SentenceTransformersRanker
except ImportError:  # pragma: no cover
    Document = None  # type: ignore
    SentenceTransformersRanker = None  # type: ignore

from ..services.logger import setup_logger


logger = setup_logger(__name__)


class HaystackReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if Document and SentenceTransformersRanker:
            try:
                self.ranker = SentenceTransformersRanker(model_name_or_path=model_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Haystack ranker failed to initialize (%s); falling back to vector order.", exc)
                self.ranker = None
        else:
            self.ranker = None
            logger.info("Haystack not available; skipping cross-encoder reranking.")

    def rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.ranker or not hits:
            return hits
        documents = []
        for idx, hit in enumerate(hits):
            text = (hit.get("snippet") or hit.get("text") or "").strip()
            doc_meta = dict(hit)
            doc_meta["original_index"] = idx
            documents.append(Document(content=text or " ", meta=doc_meta))

        try:
            ranked = self.ranker.rerank(query=query, documents=documents)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Haystack rerank failed; reverting to vector order: %s", exc)
            return hits

        ordered: List[Dict[str, Any]] = []
        seen = set()
        for doc in ranked:
            meta = dict(doc.meta)
            idx = meta.get("original_index")
            if idx is None or idx in seen or idx >= len(hits):
                continue
            hit = dict(hits[idx])
            hit["haystack_score"] = doc.score
            ordered.append(hit)
            seen.add(idx)

        for idx, hit in enumerate(hits):
            if idx not in seen:
                hit_copy = dict(hit)
                hit_copy.setdefault("haystack_score", None)
                ordered.append(hit_copy)
        return ordered
