"""Base classes for PDF extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from ..schema import Document
from ..config import PDFPipelineConfig, pipeline_config


class Extractor(ABC):
    name: str = "base"
    version: str = "0.0"

    def __init__(self, config: PDFPipelineConfig | None = None):
        self.config = config or pipeline_config

    @abstractmethod
    def extract(self, pdf_path: str, *, ocr_pdf_path: str | None = None) -> Tuple[Document, dict]:
        """Return a partial Document and metrics contextual data."""


def extractor_info(extractor: Extractor) -> dict:
    return {"name": extractor.name, "version": extractor.version}


__all__ = ["Extractor", "extractor_info"]
