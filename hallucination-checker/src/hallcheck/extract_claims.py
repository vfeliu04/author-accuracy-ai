"""Heuristic claim extraction from report text."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from .chunking import sent_tokenize


NUMBER_PATTERN = re.compile(r"-?\d[\d,\.]*")
YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def _parse_number(token: str) -> Optional[float]:
    token = token.strip()
    if not token:
        return None
    normalized = token.replace(" ", "")
    if normalized.count(",") > 1 and "." not in normalized:
        normalized = normalized.replace(",", "")
    else:
        normalized = normalized.replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def _detect_units(sentence: str) -> Optional[str]:
    lowered = sentence.lower()
    if "%" in sentence or "percent" in lowered:
        return "percent"
    if "per 100k" in lowered or "per 100,000" in lowered or "per 100000" in lowered:
        return "per 100,000"
    return None


def find_claims(text: str) -> Iterable[Dict[str, object]]:
    """Yield claim dictionaries for sentences containing numeric information."""
    for sentence in sent_tokenize(text):
        numbers = NUMBER_PATTERN.findall(sentence)
        if not numbers:
            continue

        value = _parse_number(numbers[0])
        year_match = YEAR_PATTERN.search(sentence)
        year = int(year_match.group()) if year_match else None
        units = _detect_units(sentence)

        meta: Dict[str, object] = {"numbers": numbers}
        if year:
            meta["year_match"] = year

        yield {
            "sentence": sentence.strip(),
            "value": value,
            "units": units,
            "year": year,
            "meta": meta,
        }

