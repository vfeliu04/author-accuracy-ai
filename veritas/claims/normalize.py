"""Normalization helpers for claim extraction outputs."""

from __future__ import annotations

from typing import Optional

import regex as re

from veritas.claims import patterns
from veritas.utils import text as text_utils

UNIT_MAP = {
    "percent": "%",
    "percentage": "%",
    "%": "%",
    "percentage points": "pp",
    "pp": "pp",
    "ppt": "pp",
    "ppoints": "pp",
}


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


def span_is_available(existing: list[tuple[int, int]], candidate: tuple[int, int]) -> bool:
    return text_utils.is_span_free(existing, candidate)
