"""Helpers for parsing numbers robustly across locales."""

from __future__ import annotations

from typing import Optional

import regex as re

DECIMAL_COMMA = re.compile(r"^\d{1,3}(?:\.\d{3})*,\d+$")
DECIMAL_DOT = re.compile(r"^\d{1,3}(?:,\d{3})*\.\d+$")
ONLY_COMMA = re.compile(r"^\d+,\d+$")


def canonicalize_number(raw: str) -> str:
    """Return a canonical float-friendly representation from a numeric string."""

    cleaned = raw.strip()
    cleaned = cleaned.replace("\u2212", "-")
    cleaned = cleaned.replace("−", "-")

    if DECIMAL_COMMA.match(cleaned):
        return cleaned.replace(".", "").replace(",", ".")

    if ONLY_COMMA.match(cleaned):
        return cleaned.replace(",", ".")

    if DECIMAL_DOT.match(cleaned):
        return cleaned.replace(",", "")

    parts = cleaned.split(",")
    if len(parts) > 1:
        if all(len(part) == 3 for part in parts[1:]):
            return "".join(parts)
        return cleaned.replace(",", ".")
    return cleaned


def to_float(raw: str) -> Optional[float]:
    """Parse a float value or return None if conversion fails."""

    try:
        canonical = canonicalize_number(raw)
        return float(canonical)
    except (ValueError, TypeError):
        return None
