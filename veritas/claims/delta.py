"""Delta claim parsing (e.g. 'up 12% vs 2023')."""

from __future__ import annotations

from typing import Iterable

from veritas.claims import normalize, patterns
from veritas.utils import dates, numbers

_DIRECTION_MAP = {
    "up": "up",
    "increase": "up",
    "increased": "up",
    "rise": "up",
    "rose": "up",
    "down": "down",
    "decrease": "down",
    "decreased": "down",
    "fell": "down",
    "fall": "down",
    "dropped": "down",
}


def find_candidates(text: str, reserved: Iterable[tuple[int, int]]) -> list[dict]:
    candidates: list[dict] = []
    occupied = list(reserved)

    for match in patterns.DELTA_PATTERN.finditer(text):
        start, end = match.span()
        if not normalize.span_is_available(occupied, (start, end)):
            continue

        raw_direction = match.group("direction")
        if not raw_direction:
            continue
        direction = _DIRECTION_MAP.get(raw_direction.lower())
        if not direction:
            continue

        raw_value = match.group("value")
        value = numbers.to_float(raw_value) if raw_value else None
        if value is None:
            continue

        unit = normalize.normalize_unit(match.group("unit"))
        baseline_raw = match.group("baseline")
        baseline = dates.extract_baseline(baseline_raw) if baseline_raw else None

        candidates.append(
            {
                "type": "delta",
                "start": start,
                "end": end,
                "delta": float(value),
                "unit": unit,
                "delta_direction": direction,
                "baseline_time": baseline,
            }
        )
        occupied.append((start, end))
    return candidates
