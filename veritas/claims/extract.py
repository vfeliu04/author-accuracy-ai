"""Entry point for claim extraction."""

from __future__ import annotations

from typing import Dict, List, Tuple

from rapidfuzz import fuzz

from veritas.claims import delta, normalize, range_, ratio, patterns
from veritas.claims.schema import Claim, Span
from veritas.utils import dates, text as text_utils


def extract_claims(text: str) -> List[Claim]:
    """Extract quantitative claims from free-form text."""
    if not text:
        return []

    claims: List[Claim] = []
    occupied: List[Tuple[int, int]] = []

    def register(candidate: Dict[str, object]) -> None:
        span = (candidate["start"], candidate["end"])  # type: ignore[assignment]
        if _overlaps(span, occupied):
            return
        surface = text_utils.slice_text(text, span)
        for existing in claims:
            if existing.type == candidate["type"] and fuzz.ratio(existing.text, surface) > 95:
                return
        claim = _candidate_to_claim(text, candidate, span)
        claims.append(claim)
        occupied.append(span)

    # order matters: prefer more specific constructs first
    for candidate in range_.find_ranges(text):
        register(candidate)

    for candidate in ratio.find_ratios(text):
        register(candidate)

    for candidate in delta.find_deltas(text):
        register(candidate)

    for candidate in _find_statistics(text, occupied):
        register(candidate)

    claims.sort(key=lambda c: (c.span.start, c.span.end))
    return claims


def _find_statistics(text: str, occupied: List[Tuple[int, int]]) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for match in patterns.STATISTIC_PATTERN.finditer(text):
        start, end = match.span()
        span = (start, end)
        if _overlaps(span, occupied):
            continue
        number_text = match.group("number")
        quantity = normalize.normalize_quantity(number_text)
        if quantity is None:
            continue
        unit = normalize.normalize_unit(match.group("unit"))
        qualifier = normalize.normalize_qualifier(match.group("qualifier"))
        candidates.append(
            {
                "type": "statistic",
                "start": start,
                "end": end,
                "text": text[start:end],
                "quantity": float(quantity),
                "unit": unit,
                "qualifier": qualifier,
            }
        )
    return candidates


def _candidate_to_claim(
    text: str, candidate: Dict[str, object], span: Tuple[int, int]
) -> Claim:
    qualifier = normalize.normalize_qualifier(candidate.get("qualifier"))  # type: ignore[arg-type]

    subject = text_utils.extract_subject(text, span)
    location = text_utils.extract_location_from_subject(subject)
    time = dates.find_nearest_time(text, span) if candidate["type"] != "delta" else None

    base_fields: Dict[str, object] = {
        "type": candidate["type"],
        "text": text_utils.slice_text(text, span),
        "span": Span(start=span[0], end=span[1]),
        "unit": candidate.get("unit"),
        "qualifier": qualifier,
        "subject": subject,
        "location": location,
        "time": time,
    }

    if candidate["type"] == "statistic":
        base_fields["quantity"] = candidate.get("quantity")
    elif candidate["type"] == "ratio":
        base_fields["ratio"] = candidate.get("ratio")
    elif candidate["type"] == "range":
        base_fields["range"] = candidate.get("range")
    elif candidate["type"] == "delta":
        base_fields["delta"] = candidate.get("delta")
        base_fields["delta_direction"] = candidate.get("delta_direction")
        base_fields["baseline_time"] = candidate.get("baseline_time")

    return Claim(**base_fields)


def _overlaps(span: Tuple[int, int], occupied: List[Tuple[int, int]]) -> bool:
    return any(text_utils.spans_overlap(span, other) for other in occupied)
