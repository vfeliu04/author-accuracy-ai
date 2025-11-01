"""Heading detection helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from ..config import pipeline_config


HEADING_SANITIZE_PATTERN = re.compile(r"\s+")


@dataclass
class HeadingCandidate:
    text: str
    level: int
    score: float


def normalize_heading(text: str) -> str:
    collapsed = text.replace("·", " ").replace("•", " ")
    collapsed = collapsed.replace("-", " ").replace("_", " ")
    collapsed = collapsed.strip()
    spaced = collapsed.replace(" ", "")
    if len(spaced) > 8 and spaced.isupper():
        collapsed = collapsed.replace(" ", "")
    return " ".join(collapsed.split())


def map_heading_type(title: str | None) -> str:
    if not title:
        return "generic"
    stripped = title.lower().strip()
    if stripped.startswith("executive summary"):
        return "executive_summary"
    if stripped == "abstract":
        return "abstract"
    return "generic"


def estimate_heading_level(font_size: float, base_font: float, hierarchy: Optional[List[float]] = None) -> int:
    hierarchy = hierarchy or []
    ratios = sorted(set(hierarchy + [font_size / base_font if base_font else 1.0]), reverse=True)
    for idx, ratio in enumerate(ratios):
        if abs(font_size / base_font - ratio) <= 0.05:
            return idx + 1
    return 1


def infer_level_from_numbering(text: str) -> int:
    match = re.match(r"^(\d+(\.\d+)*)[.)\s]+", text.strip())
    if not match:
        return 0
    numbers = match.group(1).split(".")
    return min(len(numbers), 6)


__all__ = [
    "HeadingCandidate",
    "normalize_heading",
    "map_heading_type",
    "estimate_heading_level",
    "infer_level_from_numbering",
]
