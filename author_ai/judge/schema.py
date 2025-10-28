"""JSON schema and parser for Stage C outputs.

The judge (LLM or otherwise) must emit JSON that matches this schema exactly.
By parsing through Pydantic we guarantee the downstream stages receive valid,
well-typed data.  The helper also provides an adapter that turns the response
into the shared `VerificationResult`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from author_ai.models import VerificationResult


class EvidencePointer(BaseModel):
    """Lightweight pointer that describes where the evidence lives."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    page: int | None = None
    start: int | None = None
    end: int | None = None


class JudgeResponse(BaseModel):
    """Schema-equivalent representation of the judge JSON contract."""

    model_config = ConfigDict(extra="forbid")

    label: Literal["supported", "contradicted", "insufficient", "outdated", "computed-mismatch"]
    matched_value: float | None
    expected_value: float | None
    unit_ok: bool
    time_ok: bool
    population_ok: bool
    chosen_evidence: list[EvidencePointer] = Field(default_factory=list)
    rationale_short: str

    @field_validator("rationale_short")
    @classmethod
    def _max_twenty_words(cls, value: str) -> str:
        """Ensure the rationale stays within the requested 20-word limit."""

        if len(value.split()) > 20:
            raise ValueError("rationale must be 20 words or fewer")
        return value

    def to_verification(self, claim_id: str, numeric_distance: float | None) -> VerificationResult:
        """Convert the judge response into the shared VerificationResult contract."""

        return VerificationResult(
            claim_id=claim_id,
            label=self.label,
            unit_ok=self.unit_ok,
            time_ok=self.time_ok,
            population_ok=self.population_ok,
            matched_value=self.matched_value,
            expected_value=self.expected_value,
            numeric_distance=numeric_distance,
            chosen_evidence=[pointer.model_dump() for pointer in self.chosen_evidence],
            rationale_short=self.rationale_short,
        )


def parse_judge_output(payload: str) -> JudgeResponse:
    """Strictly parse the judge JSON and raise user-friendly errors."""

    try:
        return JudgeResponse.model_validate_json(payload)
    except ValidationError as exc:  # pragma: no cover (defensive guardrail)
        raise ValueError(f"Judge output invalid: {exc}") from exc


__all__ = ["JudgeResponse", "EvidencePointer", "parse_judge_output"]
