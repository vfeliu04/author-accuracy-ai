"""Stage C tests for the deterministic judge."""

from __future__ import annotations

from author_ai.config import StageCConfig
from author_ai.judge import Judge
from author_ai.models import Claim, EvidenceSpan


def _claim_and_evidence() -> tuple[Claim, list[EvidenceSpan]]:
    claim = Claim(
        id="claim-1",
        sentence_id="sent-1",
        text="GDP grew 2.5% in 2023.",
        is_statistic=True,
        kind="statistic",
        values=[{"value": 2.5, "unit": "%"}],
        time="2023",
        span={"start": 0, "end": 21},
        verbatim="GDP grew 2.5% in 2023.",
        canonical={"unit": "%", "value_norm": 2.5, "time_norm": "2023", "geo_norm": None, "population_norm": None},
    )
    evidence = [
        EvidenceSpan(
            doc_id="doc-1",
            content="Official release: GDP grew by 2.5% in 2023 compared with 2022.",
        )
    ]
    return claim, evidence


def test_judge_supports_matching_numeric_values() -> None:
    claim, evidence = _claim_and_evidence()
    judge = Judge(StageCConfig(), numeric_tolerance=0.1)
    checks = {
        "unit_ok": True,
        "time_ok": True,
        "population_ok": True,
        "value_claim": 2.5,
        "value_evidence": 2.5,
        "distance": 0.0,
    }
    result = judge.evaluate(claim, evidence, checks)
    assert result.label == "supported"
