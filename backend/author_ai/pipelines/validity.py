"""
Validity scoring implementation per validity_metric.txt.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import statistics
import re

from ..models import ValidityScores
from ..storage.database import Repository
from ..config import get_settings
from ..services.logger import setup_logger
from .ingestion import IngestionPipeline


logger = setup_logger(__name__)


class ValidityPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.ingestion = IngestionPipeline()
        self.repo = Repository()

    def score_report(self, report_path: Path, report_id: str | None = None) -> ValidityScores:
        doc_id = report_id or report_path.stem
        payload = self.ingestion.ingest(report_path, doc_id=doc_id, doc_type="REPORT")
        section_lengths = [len(section["text"]) for section in payload["document"]["sections"] if section["text"]]
        coverage_score, missing_topics = self._coverage_score(payload["body_text"])
        consistency_score = self._internal_consistency_score(payload["document"]["sections"])
        methodology_score, methodology_flags = self._methodology_score(payload["document"]["sections"])
        context_score = self._context_alignment_score(payload["body_text"])
        recency_score = self._recency_score(payload)

        weights = self._parse_weights()

        overall = (
            coverage_score * weights["coverage"]
            + consistency_score * weights["consistency"]
            + methodology_score * weights["methodology"]
            + context_score * weights["context"]
            + recency_score * weights["recency"]
        )

        diagnostics = {
            "sections": len(section_lengths),
            "average_section_length": statistics.mean(section_lengths) if section_lengths else 0,
            "missing_topics": missing_topics,
            "methodology_elements": methodology_flags,
        }

        scores = ValidityScores(
            report_id=doc_id,
            overall=overall,
            coverage=coverage_score,
            consistency=consistency_score,
            methodology=methodology_score,
            context=context_score,
            recency=recency_score,
            diagnostics=diagnostics,
        )
        self.repo.upsert_validity(asdict(scores))
        summary_parts = [
            f"Validity overall {overall:.1f}.",
            f"Coverage {coverage_score:.1f}{' (missing: ' + ', '.join(missing_topics) + ')' if missing_topics else ''}.",
            f"Consistency {consistency_score:.1f}. Context alignment {context_score:.1f}. Recency {recency_score:.1f}.",
        ]
        methodology_gaps = [k for k, v in methodology_flags.items() if not v]
        if methodology_gaps:
            summary_parts.append(f"Methodology gaps: {', '.join(methodology_gaps)}.")
        self.repo.update_document_metadata(
            doc_id,
            {
                "validity_summary": " ".join(summary_parts),
                "validity_missing_topics": missing_topics,
                "validity_methodology_gaps": methodology_gaps,
            },
        )
        return scores

    def _parse_weights(self) -> Dict[str, float]:
        """Parse validity_weights config string into a dict of floats."""
        defaults = {
            "coverage": 0.25,
            "consistency": 0.25,
            "methodology": 0.2,
            "context": 0.2,
            "recency": 0.1,
        }
        try:
            parsed = {}
            for part in self.settings.validity_weights.split(","):
                key, value = part.strip().split(":")
                parsed[key.strip()] = float(value.strip())
            return parsed
        except Exception:
            logger.warning("Failed to parse validity_weights config; using defaults.")
            return defaults

    def _recency_score(self, payload: dict) -> float:
        """Calculate recency score from source document publication dates in the payload."""
        sources = payload.get("sources") or []
        now = datetime.utcnow()
        ages = []
        for source in sources:
            pub_date = source.get("publication_date") or source.get("pub_date") or source.get("year")
            if not pub_date:
                continue
            try:
                if isinstance(pub_date, int):
                    age_years = now.year - pub_date
                else:
                    parsed = datetime.strptime(str(pub_date)[:10], "%Y-%m-%d")
                    age_years = (now - parsed).days / 365.25
                ages.append(age_years)
            except (ValueError, TypeError):
                continue
        if not ages:
            return 70.0
        avg_age = statistics.mean(ages)
        if avg_age <= 2:
            return 100.0
        elif avg_age <= 5:
            return 80.0
        elif avg_age <= 10:
            return 60.0
        else:
            return 40.0

    def _internal_consistency_score(self, sections: list[dict]) -> float:
        statements = []
        for section in sections:
            statements.extend(re.findall(r"([A-Za-z\s]+)(\d+[\d,]*)(\s*\S*)", section["text"], re.IGNORECASE))
        if len(statements) < 2:
            return 90.0
        contradictions = 0
        seen: Dict[tuple, str] = {}
        for entity, number, unit_context in statements:
            entity_clean = entity.strip().lower()
            number_clean = number.replace(",", "")
            unit_hint = unit_context.strip().split()[0] if unit_context.strip() else ""
            key = (entity_clean, number_clean, unit_hint.lower())
            # Conflict: same entity+number+unit seen with a different raw number
            simple_key = entity_clean
            if simple_key in seen and seen[simple_key] != number_clean + unit_hint.lower():
                contradictions += 1
            seen[simple_key] = number_clean + unit_hint.lower()
        penalty = min(60, contradictions * 10)
        return max(40.0, 100.0 - penalty)

    def _coverage_score(self, body_text: str) -> tuple[float, list[str]]:
        topics = [t.strip() for t in self.settings.validity_topics.split(",") if t.strip()]
        text = body_text.lower()
        covered = [topic for topic in topics if topic in text]
        missing = [topic for topic in topics if topic not in covered]
        score = 60 + (len(covered) / len(topics)) * 40
        return score, missing

    def _methodology_score(self, sections: list[dict]) -> tuple[float, Dict[str, bool]]:
        keywords = {
            "data": False,
            "sample": False,
            "method": False,
            "limitation": False,
        }
        for section in sections:
            text = section["text"].lower()
            for word in keywords:
                if word in text:
                    keywords[word] = True
        covered = sum(1 for v in keywords.values() if v)
        return 50 + covered * 12.5, keywords

    def _context_alignment_score(self, text: str) -> float:
        geography_terms = ["global", "africa", "asia", "europe", "america"]
        hits = sum(1 for term in geography_terms if term in text.lower())
        return 60 + min(40, hits * 8)
