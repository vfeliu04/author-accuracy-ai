"""Configuration helpers for the PDF pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Any


def _get_bool_env(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _get_float_env(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_int_env(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class PDFPipelineConfig:
    """Runtime switches and thresholds for the PDF pipeline."""

    ocr_enabled: bool = True
    unstructured_enabled: bool = True
    layout_enabled: bool = True
    tables_enabled: bool = True
    scholarly_enabled: bool = False
    header_footer_strip: bool = True
    tables_engine: str = "camelot"  # camelot | tabula | auto
    text_density_threshold: float = 0.18
    header_footer_repeat_threshold: float = 0.6
    heading_font_ratio_primary: float = 1.25
    heading_font_ratio_secondary: float = 1.1
    min_heading_confidence: float = 0.45
    router_min_pages_for_header_detection: int = 3
    metrics_enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ocr_enabled": self.ocr_enabled,
            "unstructured_enabled": self.unstructured_enabled,
            "layout_enabled": self.layout_enabled,
            "tables_enabled": self.tables_enabled,
            "scholarly_enabled": self.scholarly_enabled,
            "header_footer_strip": self.header_footer_strip,
            "tables_engine": self.tables_engine,
            "text_density_threshold": self.text_density_threshold,
            "header_footer_repeat_threshold": self.header_footer_repeat_threshold,
            "heading_font_ratio_primary": self.heading_font_ratio_primary,
            "heading_font_ratio_secondary": self.heading_font_ratio_secondary,
            "min_heading_confidence": self.min_heading_confidence,
            "router_min_pages_for_header_detection": self.router_min_pages_for_header_detection,
            "metrics_enabled": self.metrics_enabled,
            "extra": dict(self.extra),
        }


def load_config_from_env() -> PDFPipelineConfig:
    """Load pipeline config from environment variables."""
    return PDFPipelineConfig(
        ocr_enabled=_get_bool_env("PDF_OCR_ENABLED", True),
        unstructured_enabled=_get_bool_env("PDF_UNSTRUCTURED_ENABLED", True),
        layout_enabled=_get_bool_env("PDF_LAYOUT_ENABLED", True),
        tables_enabled=_get_bool_env("PDF_TABLES_ENABLED", True),
        scholarly_enabled=_get_bool_env("PDF_SCHOLARLY_ENABLED", False),
        header_footer_strip=_get_bool_env("PDF_HEADER_FOOTER_STRIP", True),
        tables_engine=os.getenv("PDF_TABLES_ENGINE", "camelot").lower() or "camelot",
        text_density_threshold=_get_float_env("PDF_TEXT_DENSITY_THRESHOLD", 0.18),
        header_footer_repeat_threshold=_get_float_env("PDF_HEADER_FOOTER_REPEAT_THRESHOLD", 0.6),
        heading_font_ratio_primary=_get_float_env("PDF_HEADING_FONT_RATIO_PRIMARY", 1.25),
        heading_font_ratio_secondary=_get_float_env("PDF_HEADING_FONT_RATIO_SECONDARY", 1.1),
        min_heading_confidence=_get_float_env("PDF_MIN_HEADING_CONFIDENCE", 0.45),
        router_min_pages_for_header_detection=_get_int_env("PDF_ROUTER_MIN_PAGES_FOR_HEADER_DETECTION", 3),
        metrics_enabled=_get_bool_env("PDF_PIPELINE_METRICS_ENABLED", True),
        extra={},
    )


pipeline_config = load_config_from_env()
