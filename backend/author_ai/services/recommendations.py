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


logger = setup_logger(__name__)


def _flatten_tokens(text: str, limit: int = 6) -> List[str]:
    tokens = re.findall(r"[A-Za-z]{4,}", text.lower())
    seen = set()
    ordered: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
        if len(ordered) >= limit:
            break
    return ordered


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
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        keywords = self._build_keywords(claims, report_title)
        if not keywords:
            return []
        results = self._query_openalex(keywords, limit=max(limit * 2, 10))
        if not results:
            return []

        existing_titles = {source.get("name", "").lower() for source in existing_sources}
        recommendations: List[Dict[str, Any]] = []
        for result in results:
            title = (result.get("display_name") or "").strip()
            if not title or title.lower() in existing_titles:
                continue
            recommendation = self._map_openalex_result(result)
            if recommendation:
                recommendations.append(recommendation)
            if len(recommendations) >= limit:
                break
        return recommendations

    def _build_keywords(self, claims: Iterable[Dict[str, Any]], report_title: str | None) -> str:
        top_claims = list(claims)[:5]
        text = " ".join(claim.get("text", "") for claim in top_claims if claim.get("text"))
        if not text and report_title:
            text = report_title
        tokens = _flatten_tokens(text, limit=8)
        return " ".join(tokens)

    def _query_openalex(self, search: str, limit: int) -> List[Dict[str, Any]]:
        params = {
            "search": search,
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


RECOMMENDATIONS = RecommendationService()
