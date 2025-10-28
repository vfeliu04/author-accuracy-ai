"""Stage D scoring model tests."""

from __future__ import annotations

from author_ai.config import StageDConfig
from author_ai.models import EvidenceSpan, VerificationResult
from author_ai.scoring import ScoringModel


def test_scoring_model_returns_probability_and_score() -> None:
    verification = VerificationResult(
        claim_id="claim-1",
        label="supported",
        unit_ok=True,
        time_ok=True,
        population_ok=True,
        matched_value=4.2,
        expected_value=4.2,
        numeric_distance=0.0,
    )
    evidence = [
        EvidenceSpan(
            doc_id="doc-1",
            content="Unemployment stood at 4.2%.",
            scores={"bm25": 2.0, "dense": 1.2, "num_match": 1},
        )
    ]
    model = ScoringModel(StageDConfig())
    score, features = model.score(verification, evidence)
    assert 0 <= score.prob_supported <= 1
    assert features["evidence_strength"] > 0
