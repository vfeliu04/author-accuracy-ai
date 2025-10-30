"""PDF ingestion helpers leveraging Unstructured with PyMuPDF fallback."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import fitz  # PyMuPDF

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


def extract_pdf_content(pdf_path: str) -> PDFContent:
    """Extract the main body text and metadata from a PDF."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if partition_pdf is None:
        return PDFContent(text=_extract_text_pymupdf(path), title=None, authors=(), year=None)

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
            return PDFContent(text=_extract_text_pymupdf(path), title=None, authors=(), year=None)

    body_parts: List[str] = []
    titles: List[str] = []
    authors: List[str] = []
    year: Optional[int] = None

    for element in elements:
        text = (element.text or "").strip()
        if not text:
            continue

        category = getattr(element, "category", "")
        if category == "Title":
            titles.append(text)
            continue
        if _looks_like_front_matter(text, metadata=getattr(element, "metadata", None)):
            continue
        if category in {"NarrativeText", "Paragraph", "ListItem"}:
            body_parts.append(text)
            continue
        if category in {"Table", "FigureCaption", "Equation"}:
            continue

        metadata = getattr(element, "metadata", None)
        if not metadata:
            continue

        meta_title = getattr(metadata, "title", None)
        if meta_title and meta_title not in titles:
            titles.append(str(meta_title).strip())

        meta_authors = getattr(metadata, "authors", None)
        if meta_authors and not authors:
            authors.extend(str(author).strip() for author in meta_authors if str(author).strip())

        # Published date or general date
        meta_date = getattr(metadata, "published", None) or getattr(metadata, "date", None)
        if meta_date and not year:
            year = _extract_year(meta_date)

    if not body_parts:
        body_text = _clean_text(_extract_text_pymupdf(path))
    else:
        body_text = _clean_text("\n\n".join(body_parts).strip())

    title = titles[0].strip() if titles else None
    return PDFContent(text=body_text, title=title, authors=tuple(authors), year=year)


def extract_text(pdf_path: str) -> str:
    """Compatibility helper returning just the body text."""
    return extract_pdf_content(pdf_path).text


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
        # Try ISO-like formats
        parsed = _dt.datetime.fromisoformat(text)
        return parsed.year
    except Exception:
        pass
    # Fallback: look for 4-digit year
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
    stripped_lines = []
    for line in text.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue
        if _looks_like_front_matter(clean_line):
            continue
        stripped_lines.append(clean_line)
    return "\n".join(stripped_lines).strip()
