"""Tests covering the new Pydantic data contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from author_ai.models import Claim, EvidenceSpan, Score, VerificationResult


def test_claim_contract_requires_expected_fields() -> None:
    claim = Claim(
        id="claim-1",
        sentence_id="sent-1",
        text="Inflation was 6% in 2023.",
        is_statistic=True,
        kind="statistic",
        values=[{"value": 6.0, "unit": "%"}],
        time="2023",
        span={"start": 0, "end": 24},
        verbatim="Inflation was 6% in 2023.",
        canonical={"unit": "%", "value_norm": 6.0, "time_norm": "2023", "geo_norm": None, "population_norm": None},
    )
    assert claim.values[0]["unit"] == "%"


def test_evidence_span_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        EvidenceSpan(
            doc_id="doc-1",
            content="Inflation reached 6%.",
            unexpected="boom",  # type: ignore[arg-type]
        )


def test_verification_default_chosen_evidence_is_isolated() -> None:
    first = VerificationResult(
        claim_id="claim-1",
        label="supported",
        chosen_evidence=[],
    )
    second = VerificationResult(
        claim_id="claim-2",
        label="insufficient",
    )
    second.chosen_evidence.append({"doc_id": "doc-1"})
    assert first.chosen_evidence == []


def test_score_rounding_bounds_probability() -> None:
    score = Score(
        claim_id="claim-1",
        prob_supported=1.2,
        score_0_100=120,
    )
    assert score.score_0_100 == 120
