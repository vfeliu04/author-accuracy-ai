"""Layout-aware extractor built on pdfplumber with optional layoutparser support."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

try:
    import pdfplumber
except ImportError as exc:  # pragma: no cover - handled in runtime
    pdfplumber = None

try:  # pragma: no cover - optional dependency
    import layoutparser as lp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    lp = None

from ..config import PDFPipelineConfig
from ..postprocess.cleaning import normalize_text, remove_boilerplate
from ..postprocess.header_footer import identify_repeated_lines, strip_repeated_lines
from ..postprocess.headings import infer_level_from_numbering, map_heading_type, normalize_heading
from ..schema import Document, Section, new_document
from .base import Extractor

logger = logging.getLogger(__name__)


@dataclass
class Line:
    text: str
    font_size: float
    page_index: int  # 1-based
    bbox: Tuple[float, float, float, float]
    confidence: float


class GeneralLayoutExtractor(Extractor):
    name = "general_layout"
    version = "0.2"

    def extract(self, pdf_path: str, *, ocr_pdf_path: str | None = None) -> Tuple[Document, Dict[str, Any]]:
        cfg = self.config
        if pdfplumber is None:
            raise RuntimeError("pdfplumber is required for GeneralLayoutExtractor.")

        working_path = ocr_pdf_path or pdf_path
        document = new_document(pdf_path)
        metrics: Dict[str, Any] = {
            "heading_candidates": 0,
            "sections": 0,
            "header_footer_removed": 0,
            "layoutparser_available": bool(lp),
        }

        with pdfplumber.open(working_path) as pdf:
            document.metadata["page_count"] = len(pdf.pages)
            all_lines: List[List[Line]] = []
            font_sizes: List[float] = []

            for page_number, page in enumerate(pdf.pages, start=1):
                lines = _extract_lines(page, page_number)
                all_lines.append(lines)
                font_sizes.extend(line.font_size for line in lines if line.font_size)

            base_font = median(font_sizes) if font_sizes else 10.0
            logger.debug("Base font size detected as %.2f for %s", base_font, pdf_path)

            header_candidates = [[line.text for line in lines] for lines in all_lines]
            headers, footers = identify_repeated_lines(header_candidates, threshold=cfg.header_footer_repeat_threshold)
            metrics["headers_removed"] = headers
            metrics["footers_removed"] = footers

            cleaned_lines: List[List[Line]] = []
            header_footer_removed = 0
            for page_lines in all_lines:
                filtered = []
                page_text = [line.text for line in page_lines]
                stripped_lines = strip_repeated_lines(page_text, headers, footers)
                removed_count = len(page_text) - len(stripped_lines)
                header_footer_removed += removed_count
                stripped_lookup = set(stripped_lines)
                for line in page_lines:
                    if line.text in stripped_lookup:
                        filtered.append(line)
                cleaned_lines.append(filtered)

            metrics["header_footer_removed"] = header_footer_removed

            sections = _assemble_sections(cleaned_lines, base_font, cfg)
            document.sections = sections
            metrics["heading_candidates"] = sum(1 for section in sections if section.title)
            metrics["sections"] = len(sections)

            if sections:
                top_section = sections[0]
                if top_section.title:
                    document.title = top_section.title
                    document.metadata["title_conf"] = top_section.confidence

        chain = document.metadata.setdefault("extractor_chain", [])
        entry = {"name": self.name, "version": self.version}
        if entry not in chain:
            chain.append(entry)
        return document, metrics


def _extract_lines(page, page_number: int) -> List[Line]:
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
    buckets: Dict[float, List[dict]] = {}
    for word in words:
        top = round(word.get("top", 0.0), 1)
        buckets.setdefault(top, []).append(word)

    lines: List[Line] = []
    for top in sorted(buckets.keys()):
        group = sorted(buckets[top], key=lambda w: w.get("x0", 0.0))
        text = " ".join(word.get("text", "") for word in group).strip()
        if not text:
            continue
        font_sizes = [word.get("size", 0.0) or 0.0 for word in group]
        avg_size = sum(font_sizes) / max(len(font_sizes), 1)
        x0 = min(word.get("x0", 0.0) for word in group)
        y0 = min(word.get("top", 0.0) for word in group)
        x1 = max(word.get("x1", 0.0) for word in group)
        y1 = max(word.get("bottom", 0.0) for word in group)
        lines.append(
            Line(
                text=normalize_text(text),
                font_size=avg_size,
                page_index=page_number,
                bbox=(x0, y0, x1, y1),
                confidence=0.0,
            )
        )
    return lines


def _assemble_sections(pages: List[List[Line]], base_font: float, cfg: PDFPipelineConfig) -> List[Section]:
    sections: List[Section] = []
    current_title: Optional[str] = None
    current_text: List[str] = []
    current_level = 1
    current_type = "generic"
    current_confidence = 0.3
    section_start_page = 1
    section_end_page = 1

    def flush() -> None:
        nonlocal current_title, current_text, current_level, current_type, current_confidence, section_start_page, section_end_page
        if not current_text and not current_title:
            return
        text_block = "\n".join(remove_boilerplate(current_text)).strip()
        sections.append(
            Section(
                title=current_title,
                level=current_level,
                type=current_type,
                text=text_block,
                page_range=(section_start_page, section_end_page),
                confidence=current_confidence,
            )
        )
        current_title = None
        current_text = []
        current_level = 1
        current_type = "generic"
        current_confidence = 0.3
        section_start_page = section_end_page

    for page_lines in pages:
        if not page_lines:
            continue
        page_number = page_lines[0].page_index
        if not current_text and current_title is None:
            section_start_page = page_number
        for line in page_lines:
            if not line.text.strip():
                continue
            normalized = normalize_heading(line.text)
            ratio = (line.font_size / base_font) if base_font else 1.0
            numbering_level = infer_level_from_numbering(normalized)
            is_heading = False
            level = 1
            confidence = 0.4

            if ratio >= cfg.heading_font_ratio_primary:
                is_heading = True
                level = 1
                confidence = min(0.95, 0.55 + max(ratio - cfg.heading_font_ratio_primary, 0.0) * 0.3)
            elif ratio >= cfg.heading_font_ratio_secondary:
                is_heading = True
                level = 2
                confidence = 0.5
            elif numbering_level:
                is_heading = True
                level = numbering_level
                confidence = 0.45

            if is_heading and confidence >= cfg.min_heading_confidence:
                flush()
                current_title = normalized
                current_level = max(1, level)
                current_type = map_heading_type(normalized)
                current_confidence = confidence
                section_start_page = page_number
                section_end_page = page_number
                continue

            current_text.append(line.text)
            section_end_page = page_number

    flush()
    return [section for section in sections if section.text or section.title]


__all__ = ["GeneralLayoutExtractor"]
