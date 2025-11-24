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
from ..services.logger import setup_logger
from ..storage.database import Repository
from ..models import _now_iso
from ..services.summarizer import summarize_text
from ..services.section_indexer import SECTION_INDEXER
from ..services.embedding import embed_texts
import numpy as np
import re


logger = setup_logger(__name__)


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    # Semantic-ish merge: group adjacent sentences that are similar, up to max_chars.
    sentences = _split_sentences(text)
    if sentences:
        try:
            sent_vectors = embed_texts(sentences)
            chunks: List[str] = []
            current: list[str] = []
            current_vec: list[np.ndarray] = []
            def _cosine(a: np.ndarray, b: np.ndarray) -> float:
                denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
                return float(np.dot(a, b) / denom)

            SIM_THRESHOLD = 0.95
            for sent, vec in zip(sentences, sent_vectors):
                if not current:
                    current.append(sent)
                    current_vec.append(np.array(vec, dtype=float))
                    continue
                tentative = " ".join(current + [sent])
                if len(tentative) > max_chars:
                    chunks.append(" ".join(current))
                    current = [sent]
                    current_vec = [np.array(vec, dtype=float)]
                    continue
                avg_vec = sum(current_vec) / len(current_vec)
                similarity = _cosine(avg_vec, np.array(vec, dtype=float))
                if similarity >= SIM_THRESHOLD:
                    current.append(sent)
                    current_vec.append(np.array(vec, dtype=float))
                else:
                    chunks.append(" ".join(current))
                    current = [sent]
                    current_vec = [np.array(vec, dtype=float)]
            if current:
                chunks.append(" ".join(current))
            logger.info("Chunking mode: semantic merge (threshold=%.2f)", SIM_THRESHOLD)
            return chunks
        except Exception as exc:  # noqa: BLE001
            logger.debug("Semantic merge failed, falling back to recursive: %s", exc)

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
                        "metadata": {
                            "parent_id": section["id"],
                            "parent_title": section["title"],
                            "parent_page": section["page"],
                        },
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
                "summary": summary_text,
                "sections_detail": detailed_sections,
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
