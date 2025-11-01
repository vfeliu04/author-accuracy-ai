"""Optional scholarly extractor integrating external services."""

from __future__ import annotations

from typing import Dict, Tuple

from ..schema import Document, new_document
from .base import Extractor


class ScholarlyExtractor(Extractor):
    name = "scholarly_optional"
    version = "0.1"

    def extract(self, pdf_path: str, *, ocr_pdf_path: str | None = None) -> Tuple[Document, Dict]:
        # Placeholder extractor that returns an empty document. Actual integration (e.g. GROBID)
        # can override this behaviour when enabled and configured.
        metrics = {"enabled": False}
        document = new_document(pdf_path)
        return document, metrics


__all__ = ["ScholarlyExtractor"]
