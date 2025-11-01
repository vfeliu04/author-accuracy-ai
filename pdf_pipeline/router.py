"""Document router deciding extraction strategy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any

import fitz  # PyMuPDF

from .config import pipeline_config
from .postprocess.cleaning import normalize_text

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    label: str
    features: Dict[str, Any]


def route(pdf_path: str) -> RouteDecision:
    doc = fitz.open(pdf_path)
    try:
        page_count = doc.page_count
        total_chars = 0
        total_area = 0.0
        text_pages = 0
        table_like_pages = 0

        for page_index, page in enumerate(doc):
            text = page.get_text()
            chars = len(text.strip())
            total_chars += chars
            if chars > 20:
                text_pages += 1
            blocks = page.get_text("blocks")
            area = sum(abs(block[2] - block[0]) * abs(block[3] - block[1]) for block in blocks if len(block) >= 4)
            total_area += area
            if _looks_table_like(blocks):
                table_like_pages += 1

        text_density = total_chars / max(total_area, 1.0)
        is_scanned = text_pages == 0 or text_density < pipeline_config.text_density_threshold
        label = _classify(doc, is_scanned, table_like_pages, text_pages)
        features = {
            "page_count": page_count,
            "text_pages": text_pages,
            "table_like_pages": table_like_pages,
            "text_density": text_density,
            "is_scanned": is_scanned,
        }
        logger.debug("Router decision for %s: %s", pdf_path, features)
        return RouteDecision(label=label, features=features)
    finally:
        doc.close()


def _classify(doc: fitz.Document, is_scanned: bool, table_like_pages: int, text_pages: int) -> str:
    if is_scanned:
        return "scanned"
    first_page_text = normalize_text(doc[0].get_text() or "")
    last_page_text = normalize_text(doc[-1].get_text() or "")
    if "abstract" in first_page_text.lower():
        return "paper"
    if "executive summary" in first_page_text.lower():
        return "report"
    if table_like_pages >= max(1, doc.page_count // 2):
        return "table-heavy"
    if "references" in last_page_text.lower():
        return "paper"
    if text_pages <= 2:
        return "form"
    return "report"


def _looks_table_like(blocks) -> bool:
    numeric_blocks = 0
    for block in blocks or []:
        if len(block) < 5:
            continue
        text = block[4].strip()
        if not text:
            continue
        tokens = text.split()
        if not tokens:
            continue
        digit_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
        if digit_tokens / len(tokens) > 0.5:
            numeric_blocks += 1
    return numeric_blocks >= 3


__all__ = ["route", "RouteDecision"]
