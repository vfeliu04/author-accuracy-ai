"""Top-level orchestration for the five-stage verification pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from author_ai.claims import extract as legacy_extract
from author_ai.claims.schema import Claim as LegacyClaim
from author_ai.config import PipelineConfig, load_default_config
from author_ai.judge import Judge
from author_ai.models import Claim, EvidenceSpan, Score, VerificationResult
from author_ai.retrieval import HybridRetrievalPipeline
from author_ai.scoring import ScoringModel
from author_ai.ui import ReportBuilder, RenderedReport
from author_ai.utils import numbers as number_utils


SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]?")
NUMBER_FIND = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class PipelineOutput:
    """Bundle returned by `VerificationPipeline.run`."""

    claims: list[Claim]
    evidence: dict[str, list[EvidenceSpan]]
    verifications: list[VerificationResult]
    scores: list[Score]
    report: RenderedReport | None


class VerificationPipeline:
    """Coordinates Stage A–E with deterministic, test-friendly components."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or load_default_config()
        self._retriever = HybridRetrievalPipeline(self.config.stage_b)
        self._judge = Judge(
            config=self.config.stage_c,
            numeric_tolerance=self.config.stage_d.tolerances.percent_abs,
        )
        self._scoring = ScoringModel(self.config.stage_d)
        self._reporter = ReportBuilder(self.config.stage_e)

    # ------------------------------------------------------------------ Stage A
    def _stage_a(self, text: str) -> list[Claim]:
        """Run candidate extraction and map legacy claims to the new contract."""

        legacy_claims = legacy_extract.extract_claims(text)
        sentence_spans = _enumerate_sentences(text)
        converted: list[Claim] = []
        for idx, legacy_claim in enumerate(legacy_claims, start=1):
            sentence_id = _sentence_for_span(sentence_spans, legacy_claim.span.start)
            claim = _convert_legacy_claim(
                legacy_claim=legacy_claim,
                claim_index=idx,
                sentence_id=sentence_id,
            )
            converted.append(claim)
            if self.config.stage_a.max_claims and len(converted) >= self.config.stage_a.max_claims:
                break
        return converted

    # ------------------------------------------------------------------ Stage B
    def _stage_b(
        self,
        claims: Sequence[Claim],
        sources: Path | Dict[str, str],
    ) -> dict[str, list[EvidenceSpan]]:
        """Index the sources and retrieve spans for each claim."""

        if isinstance(sources, (str, Path)):
            sources_path = Path(sources)
            self._retriever.load_directory(sources_path)
        else:
            self._retriever.index_corpus(dict(sources))

        evidence_map: dict[str, list[EvidenceSpan]] = {}
        for claim in claims:
            evidence_map[claim.id] = self._retriever.retrieve(claim)
        return evidence_map

    # ------------------------------------------------------------------ Stage C
    def _stage_c(
        self,
        claims: Sequence[Claim],
        evidence_map: dict[str, list[EvidenceSpan]],
    ) -> list[VerificationResult]:
        """Evaluate each claim with the deterministic judge surrogate."""

        results: list[VerificationResult] = []
        for claim in claims:
            evidence = evidence_map.get(claim.id, [])
            checks = _precompute_checks(claim, evidence)
            result = self._judge.evaluate(claim, evidence, checks)
            results.append(result)
        return results

    # ------------------------------------------------------------------ Stage D
    def _stage_d(
        self,
        verifications: Sequence[VerificationResult],
        evidence_map: dict[str, list[EvidenceSpan]],
    ) -> list[Score]:
        """Compute calibrated scores using the deterministic feature model."""

        scores: list[Score] = []
        for verification in verifications:
            evidence = evidence_map.get(verification.claim_id, [])
            score, _ = self._scoring.score(verification, evidence)
            scores.append(score)
        return scores

    # ------------------------------------------------------------------ Stage E
    def _stage_e(
        self,
        text: str,
        claims: Sequence[Claim],
        verifications: Sequence[VerificationResult],
        scores: Sequence[Score],
        evidence_map: dict[str, list[EvidenceSpan]],
    ) -> RenderedReport | None:
        """Render the final report artefacts."""

        if not claims:
            return None
        return self._reporter.build(text, claims, verifications, scores, evidence_map)

    # ------------------------------------------------------------------ Public
    def run(self, text: str, sources: Path | Dict[str, str]) -> PipelineOutput:
        """Run the full Stage A→E pipeline."""

        claims = self._stage_a(text)
        evidence_map = self._stage_b(claims, sources)
        verifications = self._stage_c(claims, evidence_map)
        scores = self._stage_d(verifications, evidence_map)
        report = self._stage_e(text, claims, verifications, scores, evidence_map)
        return PipelineOutput(
            claims=claims,
            evidence=evidence_map,
            verifications=verifications,
            scores=scores,
            report=report,
        )


# ------------------------------------------------------------------ helpers


def _enumerate_sentences(text: str) -> list[tuple[str, int, int]]:
    """Return a list of `(sentence_id, start, end)` tuples."""

    spans: list[tuple[str, int, int]] = []
    for index, match in enumerate(SENTENCE_PATTERN.finditer(text), start=1):
        start, end = match.span()
        spans.append((f"sent-{index}", start, end))
    if not spans:
        spans.append(("sent-1", 0, len(text)))
    return spans


def _sentence_for_span(sentences: Sequence[tuple[str, int, int]], start: int) -> str:
    """Return the sentence id that contains the provided character offset."""

    for sentence_id, s_start, s_end in sentences:
        if s_start <= start < s_end:
            return sentence_id
    return sentences[-1][0]


def _convert_legacy_claim(legacy_claim: LegacyClaim, claim_index: int, sentence_id: str) -> Claim:
    """Translate the legacy claim schema into the new pipeline contract."""

    values = _claim_values_from_legacy(legacy_claim)
    canonical = _canonical_from_legacy(legacy_claim, values)
    kind = _kind_from_legacy(legacy_claim, canonical)
    return Claim(
        id=f"claim-{claim_index}",
        sentence_id=sentence_id,
        text=legacy_claim.text,
        is_statistic=kind in {"statistic", "range", "rate_per"},
        kind=kind,
        values=values if values else None,
        time=legacy_claim.time,
        comparison_time=legacy_claim.baseline_time,
        geo=legacy_claim.location,
        population=legacy_claim.population,
        span={"start": legacy_claim.span.start, "end": legacy_claim.span.end},
        verbatim=legacy_claim.text,
        canonical=canonical,
    )


def _claim_values_from_legacy(legacy_claim: LegacyClaim) -> list[dict]:
    """Build the `values` list based on the legacy claim fields."""

    collected: list[dict] = []
    if legacy_claim.quantity is not None:
        collected.append({"value": float(legacy_claim.quantity), "unit": legacy_claim.unit})
    if legacy_claim.range:
        collected.extend(
            [
                {"value": float(legacy_claim.range[0]), "unit": legacy_claim.unit},
                {"value": float(legacy_claim.range[1]), "unit": legacy_claim.unit},
            ]
        )
    if legacy_claim.delta is not None:
        collected.append({"value": float(legacy_claim.delta), "unit": legacy_claim.unit})
    if legacy_claim.ratio:
        numerator, denominator = legacy_claim.ratio
        if denominator:
            collected.append({"value": float(numerator / denominator), "unit": "ratio"})
    return [entry for entry in collected if entry.get("value") is not None]


def _canonical_from_legacy(legacy_claim: LegacyClaim, values: Sequence[dict]) -> dict | None:
    """Create the canonical dict used throughout the rest of the system."""

    value_norm = None
    if values:
        value_norm = float(values[0]["value"])
    elif legacy_claim.delta is not None:
        value_norm = float(legacy_claim.delta)
    elif legacy_claim.ratio:
        numerator, denominator = legacy_claim.ratio
        value_norm = float(numerator / denominator) if denominator else None

    if value_norm is None and legacy_claim.range:
        value_norm = float(sum(legacy_claim.range) / 2.0)

    if value_norm is None:
        return None

    return {
        "unit": legacy_claim.unit,
        "value_norm": value_norm,
        "time_norm": legacy_claim.time,
        "geo_norm": legacy_claim.location,
        "population_norm": legacy_claim.population,
    }


def _kind_from_legacy(legacy_claim: LegacyClaim, canonical: dict | None) -> str:
    """Infer the new `kind` literal from the legacy claim payload."""

    if legacy_claim.type == "ratio":
        return "ratio"
    if legacy_claim.type == "range":
        return "range"
    if legacy_claim.type == "delta":
        return "delta"
    if legacy_claim.unit and "per" in legacy_claim.unit.lower():
        return "rate_per"
    if canonical and canonical.get("value_norm") is not None and legacy_claim.qualifier:
        return "relative_change"
    return "statistic"


def _precompute_checks(claim: Claim, evidence: Sequence[EvidenceSpan]) -> dict:
    """Derive lightweight heuristics to prime the judge."""

    value_claim = _primary_claim_value(claim)
    value_evidence = _primary_evidence_value(evidence)
    distance = None
    if value_claim is not None and value_evidence is not None:
        distance = abs(value_claim - value_evidence)

    return {
        "unit_ok": bool(claim.canonical and claim.canonical.get("unit")),
        "time_ok": claim.time is not None,
        "population_ok": claim.population is not None,
        "value_claim": value_claim,
        "value_evidence": value_evidence,
        "distance": distance,
    }


def _primary_claim_value(claim: Claim) -> float | None:
    """Return the best numeric value for the claim."""

    if claim.canonical and isinstance(claim.canonical.get("value_norm"), (int, float)):
        return float(claim.canonical["value_norm"])
    if claim.values:
        for entry in claim.values:
            if isinstance(entry.get("value"), (int, float)):
                return float(entry["value"])
    return None


def _primary_evidence_value(evidence: Sequence[EvidenceSpan]) -> float | None:
    """Extract the first numeric value mentioned across the evidence spans."""

    for span in evidence:
        for match in NUMBER_FIND.findall(span.content):
            value = number_utils.to_float(match)
            if value is not None:
                return value
    return None


__all__ = ["VerificationPipeline", "PipelineOutput"]
