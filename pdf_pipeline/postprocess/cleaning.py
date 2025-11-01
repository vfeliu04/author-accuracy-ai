"""Text normalisation helpers for PDF extraction."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

COPYRIGHT_PATTERNS = [
    re.compile(r"copyright\s+\d{4}", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"doi:\s*\S+", re.IGNORECASE),
]


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00ad", "")
    normalized = re.sub(r"-\s*\n", "", normalized)
    normalized = re.sub(r"\s+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def remove_boilerplate(lines: Iterable[str]) -> List[str]:
    result: List[str] = []
    for line in lines:
        lowered = line.lower()
        if any(pattern.search(lowered) for pattern in COPYRIGHT_PATTERNS):
            continue
        result.append(line)
    return result


__all__ = ["normalize_text", "remove_boilerplate"]
