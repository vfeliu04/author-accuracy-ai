"""
PDF ingestion pipeline closely mirroring the spec in accuracy_algorithm.txt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import uuid

from ..config import get_settings
from ..services import ocr, table_extraction
from ..services.logger import setup_logger
from ..storage.database import Repository
from ..models import _now_iso
from ..services.summarizer import summarize_text


logger = setup_logger(__name__)


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
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

        if ocr.should_run_ocr(pdf_path):
            processed_path = pdf_path.parent / f"{pdf_path.stem}_ocr.pdf"
            ocr.run_ocr(pdf_path, processed_path)

        tables = table_extraction.extract_tables(processed_path)
        table_preview = tables[:3] if tables else []

        from PyPDF2 import PdfReader

        reader = PdfReader(str(processed_path))
        full_text_parts: List[str] = []
        sections: List[Dict[str, Any]] = []

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
        raw_chunks = chunk_text(body_text)

        chunk_payload = []
        for chunk in raw_chunks:
            chunk_payload.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk,
                    "doc_id": doc_id,
                }
            )

        chunk_records = []
        for chunk in chunk_payload:
            chunk_records.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": doc_id,
                    "text": chunk["text"],
                    "page_start": None,
                    "page_end": None,
                    "metadata": {},
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
                "summary": summary_text,
            },
            body_text=body_text,
            created_at=_now_iso(),
        )
        self.repo.insert_chunks(chunk_records)

        return {
            "document": {
                "sections": sections,
                "tables": tables,
            },
            "body_text": body_text,
            "chunks": chunk_payload,
            "doc_id": doc_id,
        }
