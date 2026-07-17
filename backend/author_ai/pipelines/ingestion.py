"""
PDF ingestion pipeline closely mirroring the spec in accuracy_algorithm.txt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import uuid

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover
    RecursiveCharacterTextSplitter = None  # type: ignore

from ..config import get_settings
from ..services import ocr, table_extraction
from ..services.charts import extract_charts_from_pdf, chart_to_chunks
from ..services.logger import setup_logger
from ..storage.database import Repository
from ..models import _now_iso
from ..services.summarizer import summarize_text
from ..services.section_indexer import SECTION_INDEXER


logger = setup_logger(__name__)


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    # Chunk first using structural splitter, then optionally apply semantic merge on the chunks.
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=overlap,
            separators=DEFAULT_SEPARATORS,
        )
        logger.info("Chunking mode: recursive splitter (max_chars=%d, overlap=%d)", max_chars, overlap)
        return [chunk for chunk in splitter.split_text(text) if chunk.strip()]

    # Fallback to simple sliding window chunking when LangChain is unavailable.
    logger.info("Chunking mode: sliding window (max_chars=%d, overlap=%d)", max_chars, overlap)
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(start + max_chars - overlap, end)
    return chunks


class IngestionPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.repo = Repository()

    def ingest(self, pdf_path: Path, doc_id: str, doc_type: str = "SOURCE") -> Dict[str, Any]:
        logger.info("Ingesting PDF %s", pdf_path)
        processed_path = pdf_path

        # Run OCR only when heuristics say the PDF text is sparse/low ASCII.
        try:
            if ocr.should_run_ocr(pdf_path):
                processed_path = pdf_path.parent / f"{pdf_path.stem}_ocr.pdf"
                ocr.run_ocr(pdf_path, processed_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR failed for %s: %s — continuing ingestion with original PDF", pdf_path, exc)
            processed_path = pdf_path

        tables = table_extraction.extract_tables(processed_path)
        table_preview = tables[:3] if tables else []

        from PyPDF2 import PdfReader

        reader = PdfReader(str(processed_path))
        full_text_parts: List[str] = []
        sections: List[Dict[str, Any]] = []

        charts = []
        try:
            charts = extract_charts_from_pdf(processed_path, doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chart extraction failed for %s: %s", processed_path, exc)

        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            full_text_parts.append(text)
            sections.append(
                {
                    "id": f"page-{index}",
                    "title": f"Page {index}",
                    "page": index,
                    "text": text,
                }
            )

        body_text = "\n".join(full_text_parts)
        summary_text = summarize_text(body_text)
        section_summaries: Dict[str, str] = {}
        if self.settings.section_summary_mode.lower() == "eager":
            section_summaries = SECTION_INDEXER.summarize_sections(sections)
        chunk_payload: List[Dict[str, Any]] = []
        for section in sections:
            section_chunks = chunk_text(section["text"])
            for chunk in section_chunks:
                chunk_payload.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "text": chunk,
                        "doc_id": doc_id,
                        "chunk_type": None,
                        "chart_id": None,
                        "metadata": {
                            "parent_id": section["id"],
                            "parent_title": section["title"],
                            "parent_page": section["page"],
                        },
                    }
                )
        for chart in charts:
            try:
                chart_chunks = chart_to_chunks(chart)
                chunk_payload.extend(chart_chunks)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to convert chart %s to chunks: %s", chart.id, exc)

        chunk_records = []
        for chunk in chunk_payload:
            chunk_records.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": doc_id,
                    "text": chunk["text"],
                    "page_start": None,
                    "page_end": None,
                    "chunk_type": chunk.get("chunk_type"),
                    "chart_id": chunk.get("chart_id"),
                    "x_value": chunk.get("x_value"),
                    "y_value": chunk.get("y_value"),
                    "series_name": chunk.get("series_name"),
                    "metadata": chunk.get("metadata") or {},
                }
            )

        detailed_sections = []
        for section in sections:
            detailed_sections.append(
                {
                    **section,
                    "summary": section_summaries.get(section["id"]),
                }
            )

        self.repo.upsert_document(
            doc_id=doc_id,
            doc_type=doc_type,
            path=str(pdf_path),
            metadata={
                "sections": len(sections),
                "tables": len(tables),
                "table_preview": table_preview,
                "charts": len(charts),
                "summary": summary_text,
                "sections_detail": detailed_sections,
            },
            body_text=body_text,
            created_at=_now_iso(),
        )
        self.repo.insert_chunks(chunk_records)
        if charts:
            chart_rows = [
                {
                    "id": chart.id,
                    "doc_id": chart.doc_id,
                    "page": chart.page,
                    "figure_label": chart.figure_label,
                    "bbox": chart.bbox,
                    "chart_type": chart.chart_type,
                    "raw_json": chart.raw_json,
                }
                for chart in charts
            ]
            self.repo.insert_charts(chart_rows)

        return {
            "document": {
                "sections": sections,
                "tables": tables,
            },
            "body_text": body_text,
            "chunks": chunk_payload,
            "doc_id": doc_id,
        }
