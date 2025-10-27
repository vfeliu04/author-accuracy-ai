"""Parsing helpers for ratio-style claims (e.g. 'one in five')."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from veritas.claims import normalize, patterns
from veritas.utils import numbers

WORD_NUMBERS: Dict[str, float] = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "thirteen": 13.0,
    "fourteen": 14.0,
    "fifteen": 15.0,
    "sixteen": 16.0,
    "seventeen": 17.0,
    "eighteen": 18.0,
    "nineteen": 19.0,
    "twenty": 20.0,
    "thirty": 30.0,
    "forty": 40.0,
    "fifty": 50.0,
    "sixty": 60.0,
    "seventy": 70.0,
    "eighty": 80.0,
    "ninety": 90.0,
    "hundred": 100.0,
}


def _word_to_number(token: str) -> Optional[float]:
    lowered = token.lower()
    if lowered in WORD_NUMBERS:
        return WORD_NUMBERS[lowered]
    if "-" in lowered:
        total = 0.0
        for part in lowered.split("-"):
            value = WORD_NUMBERS.get(part)
            if value is None:
                return None
            total += value
        return total
    return None


def _token_to_float(token: str) -> Optional[float]:
    numeric = numbers.to_float(token)
    if numeric is not None:
        return numeric
    return _word_to_number(token)


def find_candidates(text: str, reserved: Iterable[tuple[int, int]]) -> list[dict]:
    candidates: list[dict] = []
    occupied = list(reserved)

    for match in patterns.RATIO_PATTERN.finditer(text):
        start, end = match.span()
        if not normalize.span_is_available(occupied, (start, end)):
            continue

        num_token = match.group("num")
        den_token = match.group("den")
        num = _token_to_float(num_token)
        den = _token_to_float(den_token)
        if num is None or den is None or den == 0:
            continue

        qualifier = normalize.extract_qualifier(text, start)
        candidates.append(
            {
                "type": "ratio",
                "start": start,
                "end": end,
                "ratio": (float(num), float(den)),
                "qualifier": qualifier,
            }
        )
        occupied.append((start, end))
    return candidates
