"""Detection of numeric ranges such as '20–25%' or 'between 10 and 12 million'."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import regex as re

NUM = r"~?\d+(?:[.,]\d+)?|~?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?"
UNIT = r"(?:%|percent|per\s*100k|per\s*100\,?000|pp|million|bn|billion|thousand|k|m)?"

# Examples matched:
#  - "10–12 million"  (en dash)
#  - "10-12%"         (hyphen)
#  - "Between 20 and 25%" / "between 20 and 25 percent"
#  - Optional tilde "~" for approx
RANGE_DASH = re.compile(
    rf"""
    (?P<a>{NUM})\s* [\-–—] \s* (?P<b>{NUM})
    (?:\s*(?P<unit>{UNIT}))?
    """,
    re.VERBOSE | re.IGNORECASE,
)

RANGE_BETWEEN = re.compile(
    rf"""
    \bbetween\s+(?P<a>{NUM})\s+(?:and|to)\s+(?P<b>{NUM})
    (?:\s*(?P<unit>{UNIT}))?
    """,
    re.VERBOSE | re.IGNORECASE,
)


# Utility: normalize number with decimal comma (e.g., "1,2" -> 1.2) while
# tolerating thousands separators like "10,000" and "1 000".
def _to_float(token: str) -> Optional[float]:
    t = token.strip().lstrip("~")
    # remove spaces
    t = re.sub(r"\s+", "", t)
    # if both '.' and ',' appear, assume ',' is thousands and drop them
    if "," in t and "." in t:
        t = t.replace(",", "")
    else:
        # single kind of separator: if comma used as decimal (e.g., "1,2")
        # make a best effort: if exactly one comma and no dot, treat as decimal
        if "," in t and "." not in t:
            # beware "10,000": check 3-digit group after comma
            if re.match(r"^\d{1,3}(?:,\d{3})+$", t):
                t = t.replace(",", "")
            else:
                t = t.replace(",", ".")
        else:
            # plain thousands like "10,000" → remove commas
            t = t.replace(",", "")
    try:
        return float(t)
    except Exception:
        return None


def _unit_canon(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    u = u.strip().lower()
    if u in {"percent"}:
        return "%"
    if u in {"billion", "bn"}:
        return "billion"
    if u in {"million", "m"}:
        return "million"
    if u in {"thousand", "k"}:
        return "thousand"
    if re.fullmatch(r"per\s*100k|per\s*100\,?000", u):
        return "per 100k"
    if u in {"pp"}:
        return "pp"
    if u == "%":
        return "%"
    return u


def _iter_matches(pattern: re.Pattern, text: str) -> Iterable[Tuple[int, int, Dict[str, object]]]:
    for m in pattern.finditer(text):
        a_raw = m.group("a")
        b_raw = m.group("b")
        unit = _unit_canon(m.group("unit"))
        a = _to_float(a_raw)
        b = _to_float(b_raw)
        if a is None or b is None:
            continue
        start, end = m.span()
        surface = text[start:end]
        # naive phone-number guard (e.g., 555-1234)
        if unit is None and re.fullmatch(r"\s*\d{3}\s*[-–—]\s*\d{4}\s*", surface):
            continue
        # ensure low <= high
        low, high = (a, b) if a <= b else (b, a)
        yield start, end, {
            "type": "range",
            "text": surface,
            "start": start,
            "end": end,
            "range": (low, high),
            "unit": unit,
        }


def find_ranges(text: str) -> List[Dict[str, object]]:
    """
    Return list of candidate dicts for numeric ranges with optional unit.
    Dict keys: type, text, start, end, range=(low,high), unit
    """
    results: List[Dict[str, object]] = []
    seen: List[Tuple[int, int]] = []
    for pattern in (RANGE_BETWEEN, RANGE_DASH):
        for start, end, cand in _iter_matches(pattern, text):
            # simple overlap suppression (keep first)
            if any(not (end <= s or e <= start) for s, e in seen):
                continue
            results.append(cand)
            seen.append((start, end))
    return results
