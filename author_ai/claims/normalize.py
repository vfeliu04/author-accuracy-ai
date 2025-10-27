"""Normalization helpers for claim extraction outputs."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import regex as re

from author_ai.claims import patterns
from author_ai.utils import numbers, text as text_utils

UNIT_MAP = {
    "percent": "%",
    "percentage": "%",
    "%": "%",
    "percentage points": "pp",
    "pp": "pp",
    "ppt": "pp",
    "ppoints": "pp",
}

WORD_TO_NUMBER: Dict[str, float] = {
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
    "thousand": 1_000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
}


def _word_phrase_to_float(raw: str) -> Optional[float]:
    total = 0.0
    current = 0.0
    tokens = raw.lower().replace("-", " ").split()
    if not tokens:
        return None

    for token in tokens:
        if token in {"and", "of"}:
            continue
        value = WORD_TO_NUMBER.get(token)
        if value is None:
            return None
        if value == 100.0:
            if current == 0.0:
                current = 1.0
            current *= value
        elif value in {1_000.0, 1_000_000.0, 1_000_000_000.0}:
            if current == 0.0:
                current = 1.0
            current *= value
            total += current
            current = 0.0
        else:
            current += value
    total += current
    if total == 0.0:
        return 0.0 if raw.lower().strip() == "zero" else None
    return total


def normalize_quantity(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    numeric = numbers.to_float(cleaned)
    if numeric is not None:
        return float(numeric)
    word_value = _word_phrase_to_float(cleaned)
    if word_value is not None and word_value != 0.0:
        return word_value
    if cleaned.lower() == "zero":
        return 0.0
    return None


def normalize_unit(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in UNIT_MAP:
        return UNIT_MAP[lowered]

    if lowered.startswith("per "):
        return lowered

    if lowered.startswith(("$", "€", "£")):
        return cleaned[0]

    return cleaned


def normalize_qualifier(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    cleaned = raw.strip().lower()
    if not cleaned:
        return None
    first_match = patterns.QUALIFIER_PATTERN.search(cleaned)
    if first_match:
        return first_match.group(0).lower()
    return cleaned if cleaned in patterns.QUALIFIER_WORDS else None


def extract_qualifier(text: str, start: int) -> Optional[str]:
    """Return a nearby qualifier token preceding the given start index."""

    window_start = max(start - 24, 0)
    window = text[window_start:start]
    match = None
    for match in patterns.QUALIFIER_PATTERN.finditer(window):
        pass
    if match:
        return match.group(0).lower()
    return None


STOPWORDS = {"in", "of", "at", "for", "vs", "versus", "since", "by", "to", "from"}


def extract_subject(text: str, start: int, end: int) -> Optional[str]:
    """Attempt to extract a short subject phrase near the claim."""

    tail = text[end : end + 64]
    tokens = re.split(r"[,.();]", tail, maxsplit=1)
    candidate = tokens[0].strip()
    if not candidate:
        return None
    words = candidate.split()
    kept = []
    for word in words:
        lowered = word.lower()
        if lowered in STOPWORDS and kept:
            break
        kept.append(word)
        if len(kept) >= 5:
            break
    subject = " ".join(kept).strip()
    return subject or None


def span_is_available(existing: Iterable[tuple[int, int]], candidate: tuple[int, int]) -> bool:
    return text_utils.is_span_free(existing, candidate)


__all__ = [
    "normalize_quantity",
    "normalize_unit",
    "normalize_qualifier",
    "extract_qualifier",
    "extract_subject",
    "span_is_available",
]
