"""Second-generation PDF ingestion pipeline."""

from .schema import Document, Section, Table, Figure, ExtractorInfo
from .config import pipeline_config, PDFPipelineConfig
from .router import route
from .ocr import ocr_if_needed
from .ingest import ingest_pdf_v2

__all__ = [
    "Document",
    "Section",
    "Table",
    "Figure",
    "ExtractorInfo",
    "PDFPipelineConfig",
    "pipeline_config",
    "route",
    "ocr_if_needed",
    "ingest_pdf_v2",
]
