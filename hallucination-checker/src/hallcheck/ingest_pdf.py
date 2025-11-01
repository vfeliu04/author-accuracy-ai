"""PDF ingestion helpers leveraging Unstructured with PyMuPDF fallback."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF

from pdf_pipeline.ingest import ingest_pdf_v2
from pdf_pipeline.schema import Document

from .config import settings
try:
    from unstructured.partition.pdf import partition_pdf
except ImportError:  # pragma: no cover - optional dependency guard
    partition_pdf = None  # type: ignore[assignment]


@dataclass
class PDFContent:
    """Normalized PDF content returned by extract_pdf_content."""

    text: str
    title: Optional[str] = None
    authors: Sequence[str] = ()
    year: Optional[int] = None
    tables: Dict[str, dict] = field(default_factory=dict)
    document: Optional[Document] = None
    diagnostics: Dict[str, object] = field(default_factory=dict)


def legacy_extract_pdf_content(pdf_path: str) -> PDFContent:
    """Extract the main body text and metadata from a PDF."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if partition_pdf is None:
        return PDFContent(text=_extract_text_pymupdf(path), title=None, authors=(), year=None, tables={})

    try:
        elements = partition_pdf(  # type: ignore[misc]
            filename=str(path),
            strategy="hi_res",
            infer_table_structure=True,
            include_page_breaks=False,
        )
    except Exception:
        try:
            elements = partition_pdf(  # type: ignore[misc]
                filename=str(path),
                strategy="fast",
                infer_table_structure=False,
                include_page_breaks=False,
            )
        except Exception:
            return PDFContent(text=_extract_text_pymupdf(path), title=None, authors=(), year=None, tables={})

    body_parts: List[str] = []
    titles: List[str] = []
    authors: List[str] = []
    year: Optional[int] = None
    tables: Dict[str, dict] = {}
    placeholder_counter = 1

    for element in elements:
        metadata = getattr(element, "metadata", None)
        text = (element.text or "").strip()
        category = getattr(element, "category", "")

        if metadata:
            meta_title = getattr(metadata, "title", None)
            if meta_title and meta_title not in titles:
                titles.append(str(meta_title).strip())

            meta_authors = getattr(metadata, "authors", None)
            if meta_authors and not authors:
                authors.extend(str(author).strip() for author in meta_authors if str(author).strip())

            meta_date = getattr(metadata, "published", None) or getattr(metadata, "date", None)
            if meta_date and not year:
                year = _extract_year(meta_date)

        if category == "Title":
            if text:
                titles.append(text)
            continue

        if _looks_like_front_matter(text or "", metadata=metadata):
            continue

        if category == "Table":
            placeholder = f"[[TABLE-{placeholder_counter}]]"
            placeholder_counter += 1
            html_repr = None
            if metadata is not None:
                html_repr = getattr(metadata, "text_as_html", None)
            tables[placeholder] = {
                "html": html_repr,
                "text": text,
                "page_number": getattr(metadata, "page_number", None) if metadata else None,
                "caption": getattr(metadata, "text", None) if metadata else None,
            }
            if text:
                body_parts.append(f"{text}\n{placeholder}")
            else:
                body_parts.append(placeholder)
            continue

        if category in {"NarrativeText", "Paragraph", "ListItem"}:
            if text:
                body_parts.append(text)
            continue

        if not text:
            continue

        body_parts.append(text)

    if not body_parts:
        body_text = _clean_text(_extract_text_pymupdf(path))
    else:
        body_text = _clean_text("\n\n".join(body_parts).strip())

    title = titles[0].strip() if titles else None
    return PDFContent(text=body_text, title=title, authors=tuple(authors), year=year, tables=tables)


def extract_pdf_content(pdf_path: str) -> PDFContent:
    """Primary entrypoint that uses the new pipeline unless legacy mode is configured."""
    if settings.pdf_pipeline_legacy:
        return legacy_extract_pdf_content(pdf_path)
    document, body_text, table_map, metrics = ingest_pdf_v2(pdf_path)
    return _document_to_pdfcontent(document, body_text, table_map, diagnostics=metrics)


def ingest_pdf_document(pdf_path: str) -> Tuple[Document, str, Dict[str, dict], Dict[str, object]]:
    """Expose the v2 ingestion pipeline for callers that need structured output."""
    return ingest_pdf_v2(pdf_path)


def extract_text(pdf_path: str) -> str:
    """Compatibility helper returning just the body text."""
    content = extract_pdf_content(pdf_path)
    return strip_table_placeholders(content.text, content.tables)


def _extract_text_pymupdf(path: Path) -> str:
    doc = fitz.open(path)
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "\n".join(pages).strip()


def _extract_year(value) -> Optional[int]:
    """Try to normalize arbitrary date values to a year integer."""
    if isinstance(value, _dt.date):
        return value.year
    if isinstance(value, _dt.datetime):
        return value.year
    try:
        text = str(value).strip()
        if not text:
            return None
        parsed = _dt.datetime.fromisoformat(text)
        return parsed.year
    except Exception:
        pass
    for token in str(value).split():
        if token.isdigit() and len(token) == 4:
            try:
                year = int(token)
                if 1800 <= year <= 2100:
                    return year
            except ValueError:
                continue
    return None


def _looks_like_front_matter(text: str, metadata=None) -> bool:
    lowered = text.lower()
    if len(text) <= 40 and any(token in lowered for token in ("abstract", "keywords", "article info")):
        return True
    if "http://creativecommons.org" in lowered or "copyright" in lowered:
        return True
    if lowered.startswith("doi:") or lowered.startswith("helion"):
        return True
    if metadata is not None:
        page_number = getattr(metadata, "page_number", None)
        if page_number is not None and page_number <= 1:
            if lowered.startswith("article history") or lowered.startswith("article info"):
                return True
    return False


def _clean_text(text: str) -> str:
    stripped_lines: List[str] = []
    for line in text.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue
        if _looks_like_front_matter(clean_line):
            continue
        stripped_lines.append(clean_line)
    return "\n".join(stripped_lines).strip()


def strip_table_placeholders(text: str, tables: Optional[Dict[str, dict]]) -> str:
    if not tables:
        return text
    cleaned = text
    for placeholder in tables.keys():
        cleaned = cleaned.replace(placeholder, " ")
    return " ".join(cleaned.split())


def _document_to_pdfcontent(
    document: Document,
    body_text: str,
    table_map: Dict[str, dict],
    *,
    diagnostics: Optional[Dict[str, object]] = None,
) -> PDFContent:
    tables: Dict[str, dict] = {}
    for table_id, meta in table_map.items():
        placeholder = f"[[{table_id}]]"
        tables[placeholder] = {
            "id": table_id,
            "html": meta.get("html"),
            "text": meta.get("text"),
            "page_number": meta.get("page_number"),
            "caption": meta.get("caption"),
            "bbox": meta.get("bbox"),
        }
    year = _extract_year(document.date) if document.date else None
    return PDFContent(
        text=body_text,
        title=document.title,
        authors=tuple(document.authors),
        year=year,
        tables=tables,
        document=document,
        diagnostics=diagnostics or {},
    )
