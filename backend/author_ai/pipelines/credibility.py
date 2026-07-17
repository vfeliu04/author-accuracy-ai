"""
Credibility scoring implementation per credibility_metric.txt.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from ..models import CredibilityScore
from ..services.metadata_enrichment import METADATA
from ..storage.database import Repository
from ..config import get_settings
from ..services.logger import setup_logger


logger = setup_logger(__name__)


def _year_from_date(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value[:4])
    except ValueError:
        return None


class CredibilityPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.repo = Repository()

    def score_source(self, upload: dict) -> CredibilityScore:
        source_path = Path(upload["path"]).resolve()
        metadata = METADATA.collect_metadata(source_path)

        components: Dict[str, float] = {
            "metadata": self._metadata_completeness(metadata),
            "authority": self._authority_score(metadata, self.settings.authority_publishers_tier1, self.settings.authority_publishers_tier2),
            "recency": self._recency_score(metadata),
            "confidence": self._confidence_score(metadata),
            "user_adjustment": float(metadata.get("user_adjustment", 0.0)),  # manual tweak if present
        }

        total = min(100.0, sum(components.values()))

        score = CredibilityScore(
            source_id=upload.get("upload_id", source_path.stem),
            score=total,
            metadata_confidence=metadata.get("confidence", "LOW"),
            components=components,
        )
        self.repo.upsert_credibility(score.__dict__)
        component_text = ", ".join(f"{key}={value:.1f}" for key, value in components.items())
        credibility_summary = (
            f"Credibility score {total:.1f} composed of {component_text}. "
            f"Metadata fields present: {len(metadata.get('authors') or [])} author(s), "
            f"publisher={metadata.get('publisher') or 'unknown'}, "
            f"publication date={metadata.get('publication_date') or 'unknown'}."
        )
        self.repo.update_document_metadata(
            upload.get("upload_id", source_path.stem),
            {
                "authors": metadata.get("authors"),
                "publication_date": metadata.get("publication_date"),
                "publisher": metadata.get("publisher"),
                "doi": metadata.get("doi"),
                "metadata_confidence": metadata.get("confidence"),
                "credibility_summary": credibility_summary,
            },
        )
        return score

    # Multipliers applied to usage_count weight based on metadata confidence level.
    _CONFIDENCE_WEIGHT = {"HIGH": 1.0, "MEDIUM": 0.75, "LOW": 0.5}

    def aggregate_report(self, report_id: str) -> Dict[str, Any] | None:
        usage_stats = self.repo.source_usage(report_id)
        if not usage_stats:
            return None
        total_weight = 0.0
        weighted_sum = 0.0
        details = []
        for record in usage_stats:
            credibility = self.repo.get_credibility(record["source_id"])
            if not credibility:
                continue
            confidence_level = (credibility.get("metadata_confidence") or "LOW").upper()
            confidence_multiplier = self._CONFIDENCE_WEIGHT.get(confidence_level, 0.5)
            weight = record["usage_count"] * confidence_multiplier
            total_weight += weight
            weighted_sum += credibility["score"] * weight
            details.append(
                {
                    "source_id": record["source_id"],
                    "usage": record["usage_count"],
                    "metadata_confidence": confidence_level,
                    "score": credibility["score"],
                }
            )
        if not total_weight:
            return None
        return {
            "report_id": report_id,
            "overall": weighted_sum / total_weight,
            "details": details,
        }

    @staticmethod
    def _metadata_completeness(metadata: Dict[str, Any]) -> float:
        fields = ["title", "authors", "publication_date", "publisher", "doi"]
        present = sum(1 for field in fields if metadata.get(field))
        return 30.0 * present / len(fields)

    @staticmethod
    def _authority_score(
        metadata: Dict[str, Any],
        tier1_csv: str = "fao,un,world bank,imf,who,unicef,oecd",
        tier2_csv: str = "reuters,associated press,bbc,nature,science,lancet",
    ) -> float:
        publisher = (metadata.get("publisher") or "").lower()
        tier1 = [kw.strip() for kw in tier1_csv.split(",") if kw.strip()]
        tier2 = [kw.strip() for kw in tier2_csv.split(",") if kw.strip()]
        if any(kw in publisher for kw in tier1):
            return 30.0
        if any(kw in publisher for kw in tier2):
            return 22.5
        if publisher:
            return 15.0
        return 7.5

    @staticmethod
    def _recency_score(metadata: Dict[str, Any]) -> float:
        publication_year = _year_from_date(metadata.get("publication_date"))
        if not publication_year:
            return 10.0
        age = datetime.utcnow().year - publication_year
        if age <= 2:
            return 20.0
        if age <= 5:
            return 12.0
        if age <= 10:
            return 6.0
        return 3.0

    @staticmethod
    def _confidence_score(metadata: Dict[str, Any]) -> float:
        mapping = {"HIGH": 10.0, "MEDIUM": 7.0, "LOW": 3.0}
        return mapping.get((metadata.get("confidence") or "LOW").upper(), 3.0)
