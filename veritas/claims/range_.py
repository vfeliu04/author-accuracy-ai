"""Detection of numeric ranges such as '20–25%' or 'between 10 and 12 million'."""

from __future__ import annotations

from typing import Iterable

from veritas.claims import normalize, patterns
from veritas.utils import numbers


def find_candidates(text: str, reserved: Iterable[tuple[int, int]]) -> list[dict]:
    candidates: list[dict] = []
    occupied = list(reserved)

    for match in patterns.RANGE_PATTERN.finditer(text):
        start, end = match.span()
        if not normalize.span_is_available(occupied, (start, end)):
            continue

        low_raw = match.group("start")
        high_raw = match.group("end")
        if not low_raw or not high_raw:
            continue

        low = numbers.to_float(low_raw)
        high = numbers.to_float(high_raw)
        if low is None or high is None:
            continue

        unit_raw = match.group("unit")
        unit = normalize.normalize_unit(unit_raw.strip().rstrip(",.;")) if unit_raw else None

        qualifier = None
        prefix = match.group("prefix") or ""
        if prefix:
            qualifier_candidate = prefix.strip().split()[0].lower().strip("~")
            if qualifier_candidate in patterns.QUALIFIER_WORDS:
                qualifier = qualifier_candidate

        qualifier = qualifier or normalize.extract_qualifier(text, start)

        candidates.append(
            {
                "type": "range",
                "start": start,
                "end": end,
                "range": (float(low), float(high)),
                "unit": unit,
                "qualifier": qualifier,
            }
        )
        occupied.append((start, end))
    return candidates
