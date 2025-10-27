"""Date and temporal parsing utilities."""

from __future__ import annotations

from typing import Optional

import dateparser

from author_ai.claims import patterns


def _normalize_year(year_str: str) -> Optional[str]:
    if len(year_str) == 2:
        year = int(year_str)
        year += 2000 if year < 50 else 1900
        return f"{year:04d}"
    if len(year_str) == 4:
        return year_str
    return None


def _match_quarter(text: str) -> Optional[str]:
    match = patterns.QUARTER_PATTERN.search(text)
    if not match:
        return None

    if match.group("qprefix"):
        year = _normalize_year(match.group("year"))
        quarter = match.group("qnum")
    elif match.group("year_leading"):
        year = _normalize_year(match.group("year_leading"))
        quarter = match.group("qnum_leading")
    else:
        year = _normalize_year(match.group("compact_year"))
        quarter = match.group("compact")

    if year and quarter:
        return f"{year}-Q{int(quarter)}"
    return None


def _match_year(text: str) -> Optional[str]:
    match = patterns.YEAR_PATTERN.search(text)
    if match:
        return match.group(0)
    return None


def extract_time_context(text: str, start: int, end: int, window: int = 48) -> Optional[str]:
    """Return a normalized temporal reference surrounding a span if available."""

    left = max(start - window, 0)
    right = min(end + window, len(text))
    snippet = text[left:right]

    quarter = _match_quarter(snippet)
    if quarter:
        return quarter

    year = _match_year(snippet)
    if year:
        return year

    parsed = dateparser.parse(snippet, settings={"PREFER_DAY_OF_MONTH": "first"})
    if parsed:
        return f"{parsed.year:04d}"
    return None


def extract_baseline(text: str) -> Optional[str]:
    """Normalize baseline temporal references (used for deltas)."""

    quarter = _match_quarter(text)
    if quarter:
        return quarter

    year = _match_year(text)
    if year:
        return year

    parsed = dateparser.parse(text, settings={"PREFER_DAY_OF_MONTH": "first"})
    if parsed:
        return f"{parsed.year:04d}"
    return None
