"""
LLM-based evidence reranker.
"""

from __future__ import annotations

from typing import List, Dict, Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from ..config import get_settings
from .logger import setup_logger


logger = setup_logger(__name__)


class EvidenceReranker:
    def __init__(self):
        self.settings = get_settings()
        self.model = self.settings.rerank_model
        if OpenAI and self.settings.openai_api_key and self.model:
            self.client = OpenAI(api_key=self.settings.openai_api_key)
        else:
            self.client = None
            logger.info("Reranker using vector similarity order (set OPENAI_API_KEY for LLM reranking).")

    def rerank(self, claim_text: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.client or not hits:
            return hits

        prompt = self._build_prompt(claim_text, hits)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You rank candidate evidence snippets for factual verification. "
                            "Return rankings as newline-separated entries in the format 'index|score', "
                            "where index is the snippet number and score is between 0 and 1."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            content = response.choices[0].message.content if response.choices else None
            if not content:
                return hits
            ranking = self._parse_ranking(content, len(hits))
            if not ranking:
                return hits
            ordered = []
            for order, score in ranking:
                idx = order - 1
                if 0 <= idx < len(hits):
                    hit = dict(hits[idx])
                    hit["rerank_score"] = score
                    ordered.append(hit)
            remaining = {i for i in range(len(hits))} - {r[0] - 1 for r in ranking}
            for idx in remaining:
                hit = dict(hits[idx])
                hit.setdefault("rerank_score", None)
                ordered.append(hit)
            return ordered
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranker failed, falling back to similarity order: %s", exc)
            return hits

    @staticmethod
    def _build_prompt(claim_text: str, hits: List[Dict[str, Any]]) -> str:
        lines = [f"Claim: {claim_text.strip()}"]
        lines.append("Snippets:")
        for idx, hit in enumerate(hits, start=1):
            snippet = (hit.get("snippet") or hit.get("text") or "").strip()
            parent = hit.get("parent") or {}
            context_bits = []
            if parent.get("title"):
                context_bits.append(parent["title"])
            if parent.get("page"):
                context_bits.append(f"page {parent['page']}")
            context = f" ({', '.join(context_bits)})" if context_bits else ""
            lines.append(f"{idx}. Source {hit.get('doc_id')}{context}: {snippet[:600]}")
        lines.append(
            "Rank the snippets by how well they support the claim. "
            "Output newline-separated `index|score` entries from strongest to weakest."
        )
        return "\n".join(lines)

    @staticmethod
    def _parse_ranking(content: str, total: int) -> List[tuple[int, float]]:
        ranking: List[tuple[int, float]] = []
        for line in content.strip().splitlines():
            if "|" not in line:
                continue
            left, right = line.split("|", 1)
            try:
                idx = int(left.strip())
                score = float(right.strip())
            except ValueError:
                continue
            if 1 <= idx <= total:
                ranking.append((idx, max(0.0, min(score, 1.0))))
        return ranking
