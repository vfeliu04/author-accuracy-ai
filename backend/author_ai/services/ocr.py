"""
Heuristics for detecting whether a PDF requires OCR and a dispatcher to run OCRmyPDF.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple

from PyPDF2 import PdfReader  # type: ignore

from ..config import get_settings
from .logger import setup_logger


logger = setup_logger(__name__)


def _pdf_text_stats(path: Path) -> Tuple[int, float]:
    reader = PdfReader(str(path))
    total_chars = 0
    ascii_chars = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        total_chars += len(text)
        ascii_chars += sum(1 for ch in text if ch.isascii())

    ratio = (ascii_chars / total_chars) if total_chars else 0.0
    return total_chars, ratio


def should_run_ocr(path: Path) -> bool:
    settings = get_settings()
    if not settings.ocr_enabled:
        return False

    total_chars, ascii_ratio = _pdf_text_stats(path)
    logger.debug("PDF text stats for %s: chars=%d, ascii_ratio=%.2f", path, total_chars, ascii_ratio)
    trigger = total_chars < settings.ocr_min_text_threshold or ascii_ratio < settings.ocr_low_quality_ratio
    return trigger


def run_ocr(path: Path, output_path: Path) -> None:
    settings = get_settings()
    if not settings.ocr_enabled:
        raise RuntimeError("OCR requested but PDF_OCR_ENABLED=false")

    cmd = [
        settings.ocr_binary_path,
        "--skip-text",
        "--rotate-pages",
        "--deskew",
        str(path),
        str(output_path),
    ]
    logger.info("Running OCR: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=False)
