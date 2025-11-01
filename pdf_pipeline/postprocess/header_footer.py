"""Repeated header/footer detection and stripping."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Tuple

from ..config import pipeline_config


def identify_repeated_lines(pages: Iterable[List[str]], *, threshold: float | None = None) -> Tuple[List[str], List[str]]:
    threshold = pipeline_config.header_footer_repeat_threshold if threshold is None else threshold
    header_counter: Counter[str] = Counter()
    footer_counter: Counter[str] = Counter()
    page_total = 0

    for page_lines in pages:
        if not page_lines:
            continue
        page_total += 1
        normalized = [normalize_line(line) for line in page_lines if line.strip()]
        if not normalized:
            continue
        header_counter.update([normalized[0]])
        footer_counter.update([normalized[-1]])

    required = max(1, int(page_total * threshold))
    headers = [line for line, count in header_counter.items() if count >= required]
    footers = [line for line, count in footer_counter.items() if count >= required]
    return headers, footers


def normalize_line(line: str) -> str:
    return " ".join(line.strip().split()).lower()


def strip_repeated_lines(page_text: List[str], headers: List[str], footers: List[str]) -> List[str]:
    if not page_text:
        return page_text
    result: List[str] = []
    for idx, line in enumerate(page_text):
        norm = normalize_line(line)
        if idx == 0 and norm in headers:
            continue
        if idx == len(page_text) - 1 and norm in footers:
            continue
        result.append(line)
    return result


__all__ = ["identify_repeated_lines", "strip_repeated_lines", "normalize_line"]
