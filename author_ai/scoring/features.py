"""Feature engineering helpers used by Stage D."""

from __future__ import annotations

from typing import Iterable

from author_ai.models import EvidenceSpan, VerificationResult


LABEL_PRIORS = {
    "supported": 1.0,
    "contradicted": 0.1,
    "insufficient": 0.4,
    "outdated": 0.5,
    "computed-mismatch": 0.3,
    "disputed": 0.2,
}


def compute_features(verification: VerificationResult, evidence: Iterable[EvidenceSpan]) -> dict[str, float]:
    """Collapse evidence + judge output into a deterministic feature vector."""

    evidence_list = list(evidence)
    features = {
        "evidence_strength": _evidence_strength(evidence_list),
        "unit_time_geo_checks": _bool_average(
            verification.unit_ok,
            verification.time_ok,
            verification.population_ok,
        ),
        "numeric_tolerance": _numeric_component(verification.numeric_distance),
        "source_quality": 0.8 if evidence_list else 0.2,
        "judge_label_weight": LABEL_PRIORS.get(verification.label, 0.2),
    }
    return features


def _evidence_strength(evidence: Iterable[EvidenceSpan]) -> float:
    """Combine retrieval scores into a bounded [0,1] indicator."""

    scores = []
    for span in evidence:
        if not span.scores:
            continue
        bm25 = span.scores.get("bm25") or 0.0
        dense = span.scores.get("dense") or 0.0
        num_match = span.scores.get("num_match") or 0.0
        scores.append(bm25 + dense + num_match)
    if not scores:
        return 0.0
    return min(sum(scores) / len(scores), 1.0)


def _bool_average(*values: bool | None) -> float:
    """Treat missing values as unknown instead of biasing toward 0 or 1."""

    filtered = [1.0 if value else 0.0 for value in values if value is not None]
    if not filtered:
        return 0.5
    return sum(filtered) / len(filtered)


def _numeric_component(distance: float | None) -> float:
    """Reward small numeric distance and fall back to neutral if absent."""

    if distance is None:
        return 0.5
    return min(1.0, 1.0 / (1.0 + distance))


__all__ = ["compute_features", "LABEL_PRIORS"]
