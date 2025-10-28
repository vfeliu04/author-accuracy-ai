"""Stage D scoring model that combines engineered features."""

from __future__ import annotations

from typing import Iterable, Tuple

from author_ai.config import StageDConfig
from author_ai.models import EvidenceSpan, Score, VerificationResult
from author_ai.scoring.calibrate import build_calibrator
from author_ai.scoring.features import compute_features


class ScoringModel:
    """Weighted linear model + calibration, kept intentionally simple."""

    def __init__(self, config: StageDConfig) -> None:
        self.config = config
        self._calibrator = build_calibrator(config.calibration, config.temperature)

    def score(
        self,
        verification: VerificationResult,
        evidence: Iterable[EvidenceSpan],
    ) -> Tuple[Score, dict[str, float]]:
        """Return both the score card and the raw feature map."""

        features = compute_features(verification, evidence)
        weights = self.config.weights
        logit = (
            features["evidence_strength"] * weights.evidence_strength
            + features["unit_time_geo_checks"] * weights.unit_time_geo_checks
            + features["numeric_tolerance"] * weights.numeric_tolerance
            + features["source_quality"] * weights.source_quality
            + features["judge_label_weight"] * weights.judge_label_weight
        )
        probability = float(self._calibrator.calibrate(logit))
        score_value = max(0, min(100, int(round(probability * 100))))
        score = Score(
            claim_id=verification.claim_id,
            prob_supported=probability,
            score_0_100=score_value,
            uncertainty={"std": round((1.0 - probability) * 0.25, 3)},
            factors=features,
        )
        return score, features


__all__ = ["ScoringModel"]
