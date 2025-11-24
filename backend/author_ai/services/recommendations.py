"""
Generate external source recommendations using the OpenAlex API.
"""

from __future__ import annotations

from datetime import datetime
import math
import re
from typing import Iterable, List, Dict, Any, Optional

import requests

from ..config import get_settings
from .logger import setup_logger
from .summarizer import summarize_text
from .embedding import embed_texts


logger = setup_logger(__name__)


STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "into",
    "over",
    "under",
    "about",
    "their",
    "there",
    "these",
    "those",
    "which",
    "while",
    "where",
    "when",
    "what",
    "also",
    "have",
    "has",
    "had",
    "been",
    "being",
    "across",
    "through",
    "between",
    "within",
    "into",
    "among",
    "such",
    "more",
    "less",
    "many",
    "most",
    "very",
}


def _stem(token: str) -> str:
    # Minimal stemming to group variants without extra deps.
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _top_terms(text: str, limit: int = 16) -> List[str]:
    tokens = re.findall(r"[A-Za-z]{4,}", text.lower())
    freq: Dict[str, int] = {}
    for token in tokens:
        if token in STOPWORDS:
            continue
        stemmed = _stem(token)
        freq[stemmed] = freq.get(stemmed, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [term for term, _ in ranked[:limit]]


class RecommendationService:
    """Lightweight wrapper around OpenAlex for surfacing higher-quality sources."""

    def __init__(self):
        self.settings = get_settings()
        self.session = requests.Session()

    def recommend(
        self,
        *,
        claims: Iterable[Dict[str, Any]],
        existing_sources: Iterable[Dict[str, Any]],
        report_title: str | None = None,
        report_summary: str | None = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        claims = list(claims)
        existing_sources = list(existing_sources)
        keywords, topic_terms = self._build_keywords_and_topics(claims, report_title, report_summary, existing_sources)
        if not keywords and not topic_terms:
            return []
        results = self._query_openalex(keywords, topic_terms, limit=max(limit * 2, 15))
        if not results:
            return []

        existing_titles = {source.get("name", "").lower() for source in existing_sources}
        recommendations: List[Dict[str, Any]] = []
        query_vector = self._embed_query_context(report_title, report_summary, claims)

        for result in results:
            title = (result.get("display_name") or "").strip()
            if not title or title.lower() in existing_titles:
                continue
            if topic_terms and not self._is_on_topic(result, topic_terms):
                continue
            recommendation = self._map_openalex_result(result)
            if recommendation:
                if query_vector:
                    rec_vector = self._embed_candidate(recommendation)
                    if rec_vector:
                        similarity = self._cosine(query_vector, rec_vector)
                        recommendation["relevance_score"] = similarity
                recommendations.append(recommendation)
            if len(recommendations) >= limit:
                break
        if query_vector:
            recommendations = [rec for rec in recommendations if rec.get("relevance_score", 0) >= 0.18]
            recommendations.sort(
                key=lambda rec: (
                    rec.get("relevance_score", 0),
                    _recency_boost(rec.get("publication_year")),
                    (rec.get("credibility_score") or 0),
                ),
                reverse=True,
            )
        return recommendations

    def _build_keywords_and_topics(
        self,
        claims: Iterable[Dict[str, Any]],
        report_title: str | None,
        report_summary: str | None,
        existing_sources: Iterable[Dict[str, Any]],
    ) -> tuple[str, List[str]]:
        claim_texts = [claim.get("text", "") for claim in claims if claim.get("text")]
        source_summaries = [source.get("summary", "") for source in existing_sources if source.get("summary")]
        corpus = " ".join(
            filter(
                None,
                [
                    report_title or "",
                    report_summary or "",
                    " ".join(claim_texts),
                    " ".join(source_summaries),
                ],
            )
        )
        terms = _top_terms(corpus, limit=16)
        keywords = " ".join(terms[:5])
        return keywords, terms

    def _query_openalex(self, search: str, topic_terms: List[str], limit: int) -> List[Dict[str, Any]]:
        topic_filter = " OR ".join(topic_terms) if topic_terms else ""
        search_param = f"{search} ({topic_filter})" if topic_filter else search
        params = {
            "search": search_param.strip(),
            "sort": "relevance_score:desc",
            "per-page": limit,
            "filter": "from_publication_date:2018-01-01,has_doi:true",
        }
        mailto = self.settings.openalex_mailto
        if mailto:
            params["mailto"] = mailto
        url = f"{self.settings.openalex_base_url.rstrip('/')}/works"
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.warning("OpenAlex request failed: %s", exc)
            return []
        return data.get("results") or []

    def _is_on_topic(self, result: Dict[str, Any], topic_terms: List[str]) -> bool:
        text = " ".join(
            filter(
                None,
                [
                    result.get("display_name") or "",
                    _decode_abstract(result.get("abstract_inverted_index")) or "",
                ],
            )
        ).lower()
        return any(term in text for term in topic_terms)

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        import numpy as np

        va = np.array(a, dtype=float)
        vb = np.array(b, dtype=float)
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
        return float(np.dot(va, vb) / denom)

    def _embed_query_context(
        self,
        report_title: Optional[str],
        report_summary: Optional[str],
        claims: Iterable[Dict[str, Any]],
    ) -> Optional[List[float]]:
        texts: List[str] = []
        if report_title:
            texts.append(report_title)
        if report_summary:
            texts.append(report_summary)
        texts.extend(claim.get("text", "") for claim in claims if claim.get("text"))
        if not texts:
            return None
        vectors = embed_texts([" ".join(texts)])
        return vectors[0] if vectors else None

    def _embed_candidate(self, rec: Dict[str, Any]) -> Optional[List[float]]:
        text = " ".join(filter(None, [rec.get("title"), rec.get("abstract"), rec.get("summary")]))
        if not text:
            return None
        vectors = embed_texts([text])
        return vectors[0] if vectors else None

    def _map_openalex_result(self, result: Dict[str, Any]) -> Dict[str, Any] | None:
        title = (result.get("display_name") or "").strip()
        if not title:
            return None
        publication_year = result.get("publication_year")
        cited_by = result.get("cited_by_count")
        authors = [
            (auth.get("author") or {}).get("display_name")
            for auth in result.get("authorships", [])
            if (auth.get("author") or {}).get("display_name")
        ][:4]
        venue = (result.get("host_venue") or {}).get("display_name")
        location = result.get("primary_location") or {}
        landing_page = location.get("landing_page_url") or (result.get("best_oa_location") or {}).get("url")
        doi = result.get("doi")
        abstract = _decode_abstract(result.get("abstract_inverted_index"))
        summary = entry_summary(abstract, title)
        date_published = result.get("publication_date") or (str(publication_year) if publication_year else None)
        credibility_score = _credibility_score(publication_year, cited_by, doi, authors)
        validity_score = _validity_score(abstract, publication_year)

        return {
            "id": result.get("id"),
            "title": title,
            "date_published": date_published,
            "publication_year": publication_year,
            "cited_by_count": cited_by,
            "authors": authors,
            "doi": doi,
            "url": landing_page,
            "openalex_url": result.get("id"),
            "abstract": abstract,
            "summary": summary,
            "credibility_score": credibility_score,
            "validity_score": validity_score,
            "host_venue": venue,
        }


def _decode_abstract(index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    if not index:
        return None
    positions: Dict[int, str] = {}
    try:
        for token, indices in index.items():
            for position in indices:
                positions[position] = token
        if not positions:
            return None
        text = " ".join(word for _, word in sorted(positions.items()))
        return text or None
    except Exception as exc:  # pragma: no cover - defensive against unexpected data
        logger.debug("Failed to decode OpenAlex abstract: %s", exc)
        return None


def entry_summary(abstract: Optional[str], title: str) -> str:
    if abstract:
        summary = summarize_text(abstract, word_limit=120)
        if summary:
            return summary
    fallback_text = f"{title}. This source discusses relevant factors for food security."
    return summarize_text(fallback_text, max_sentences=2)


def _credibility_score(publication_year: Optional[int], cited_by: Optional[int], doi: Optional[str], authors: List[str]) -> float:
    score = 20.0
    current_year = datetime.utcnow().year
    if publication_year:
        age = max(0, current_year - publication_year)
        score += max(0, 35 - min(7 * age, 35))
    if isinstance(cited_by, int):
        score += min(30, math.log10(cited_by + 1) * 12)
    if doi:
        score += 10
    if authors:
        score += min(10, len(authors) * 2)
    return max(5.0, min(score, 100.0))


def _validity_score(abstract: Optional[str], publication_year: Optional[int]) -> float:
    if not abstract:
        base = 45.0
    else:
        length = len(abstract.split())
        base = min(70.0, 40 + length * 0.05)
    if publication_year:
        freshness = max(0, 10 - (datetime.utcnow().year - publication_year))
        base += freshness * 2
    return max(10.0, min(base, 100.0))


def _recency_boost(publication_year: Optional[int]) -> float:
    if not publication_year:
        return 0.0
    age = max(0, datetime.utcnow().year - publication_year)
    return max(0.0, 10 - age) / 10.0


RECOMMENDATIONS = RecommendationService()
