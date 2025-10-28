"""Deterministic judge that mimics the behaviour of an LLM.

The real system would call an LLM with the prompts defined in `prompt.py` and
then parse the JSON.  For offline development we simulate the reasoning using
simple numeric rules so the rest of the pipeline can run end-to-end.
"""

from __future__ import annotations

from typing import Iterable

from author_ai.config import StageCConfig
from author_ai.models import Claim, EvidenceSpan, VerificationResult
from author_ai.judge.schema import EvidencePointer, JudgeResponse


class Judge:
    """Rule-based surrogate for the Stage C judge."""

    def __init__(self, config: StageCConfig, numeric_tolerance: float = 0.5) -> None:
        self.config = config
        self.numeric_tolerance = numeric_tolerance

    def evaluate(
        self,
        claim: Claim,
        evidence: Iterable[EvidenceSpan],
        precomputed_checks: dict,
    ) -> VerificationResult:
        """Return a verification decision based on simple heuristics."""

        evidence_list = list(evidence)[: self.config.max_spans]
        if not evidence_list:
            response = JudgeResponse(
                label="insufficient",
                matched_value=None,
                expected_value=precomputed_checks.get("value_claim"),
                unit_ok=bool(precomputed_checks.get("unit_ok")),
                time_ok=bool(precomputed_checks.get("time_ok")),
                population_ok=bool(precomputed_checks.get("population_ok")),
                chosen_evidence=[],
                rationale_short="No evidence retrieved",
            )
            return response.to_verification(claim_id=claim.id, numeric_distance=None)

        distance = precomputed_checks.get("distance")
        value_claim = precomputed_checks.get("value_claim")
        value_evidence = precomputed_checks.get("value_evidence")

        if distance is None:
            label = "insufficient"
        elif distance <= self.numeric_tolerance:
            label = "supported"
        elif not precomputed_checks.get("unit_ok", True):
            label = "computed-mismatch"
        else:
            label = "contradicted"

        pointers = [
            EvidencePointer(doc_id=span.doc_id, page=span.page, start=span.start, end=span.end)
            for span in evidence_list
        ]

        response = JudgeResponse(
            label=label,
            matched_value=value_evidence,
            expected_value=value_claim,
            unit_ok=bool(precomputed_checks.get("unit_ok")),
            time_ok=bool(precomputed_checks.get("time_ok")),
            population_ok=bool(precomputed_checks.get("population_ok")),
            chosen_evidence=pointers,
            rationale_short=self._rationale(label, distance),
        )
        return response.to_verification(claim_id=claim.id, numeric_distance=distance)

    @staticmethod
    def _rationale(label: str, distance: float | None) -> str:
        """Craft a terse rationale that respects the 20-word limit."""

        if label == "supported":
            return "Evidence numbers align with claim"
        if label == "contradicted":
            return "Evidence contradicts claim value"
        if label == "computed-mismatch":
            return "Evidence present but units mismatch"
        if distance is None:
            return "Evidence lacks numeric detail"
        return "Insufficient overlap between evidence and claim"


__all__ = ["Judge"]
