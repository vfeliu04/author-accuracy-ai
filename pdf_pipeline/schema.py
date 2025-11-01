"""Unified data structures for PDF extraction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class ExtractorInfo:
    name: str
    version: str


@dataclass
class Section:
    title: Optional[str]
    level: int
    type: str
    text: str
    page_range: Tuple[int, int]
    confidence: float


@dataclass
class Table:
    id: str
    page: int
    caption: Optional[str]
    html: Optional[str]
    text: Optional[str]
    bbox: Optional[Tuple[float, float, float, float]]


@dataclass
class Figure:
    id: str
    page: int
    caption: Optional[str]
    bbox: Optional[Tuple[float, float, float, float]]


@dataclass
class Document:
    metadata: Dict[str, Any]
    title: Optional[str]
    authors: List[str]
    date: Optional[str]
    sections: List[Section]
    tables: List[Table]
    figures: List[Figure]
    references: List[str]

    def clone(self, **updates: Any) -> "Document":
        data = {
            "metadata": dict(self.metadata),
            "title": self.title,
            "authors": list(self.authors),
            "date": self.date,
            "sections": [replace(section) for section in self.sections],
            "tables": [replace(table) for table in self.tables],
            "figures": [replace(fig) for fig in self.figures],
            "references": list(self.references),
        }
        data.update(updates)
        return Document(**data)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


def asdict(document: Document) -> Dict[str, Any]:
    """Convert a Document to a plain dictionary."""
    return {
        "metadata": dict(document.metadata),
        "title": document.title,
        "authors": list(document.authors),
        "date": document.date,
        "sections": [
            {
                "title": section.title,
                "level": int(section.level),
                "type": section.type,
                "text": section.text,
                "page_range": list(section.page_range),
                "confidence": float(section.confidence),
            }
            for section in document.sections
        ],
        "tables": [
            {
                "id": table.id,
                "page": int(table.page),
                "caption": table.caption,
                "html": table.html,
                "text": table.text,
                "bbox": list(table.bbox) if table.bbox else None,
            }
            for table in document.tables
        ],
        "figures": [
            {
                "id": figure.id,
                "page": int(figure.page),
                "caption": figure.caption,
                "bbox": list(figure.bbox) if figure.bbox else None,
            }
            for figure in document.figures
        ],
        "references": list(document.references),
    }


def new_document(source_path: str, extractor_chain: Optional[List[ExtractorInfo]] = None) -> Document:
    absolute = Path(source_path).resolve()
    extractor_meta = [
        {"name": info.name, "version": info.version}
        for info in (extractor_chain or [])
    ]
    metadata = {
        "doc_id": "",
        "source_path": str(absolute),
        "created_at": _now_iso(),
        "extractor_chain": extractor_meta,
        "is_scanned": False,
        "page_count": 0,
        "title_conf": 0.0,
        "router_label": "unknown",
    }
    return Document(
        metadata=metadata,
        title=None,
        authors=[],
        date=None,
        sections=[],
        tables=[],
        figures=[],
        references=[],
    )


def compute_content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def merge_documents(base: Document, supplements: Iterable[Document]) -> Document:
    merged = base.clone()
    existing_section_texts = {section.text for section in merged.sections}
    for supplement in supplements:
        merged.metadata["extractor_chain"].extend(supplement.metadata.get("extractor_chain", []))
        if supplement.title and not merged.title:
            merged.title = supplement.title
            merged.metadata["title_conf"] = max(
                merged.metadata.get("title_conf", 0.0),
                supplement.metadata.get("title_conf", 0.0),
            )
        if supplement.authors and not merged.authors:
            merged.authors = list(supplement.authors)
        if supplement.date and not merged.date:
            merged.date = supplement.date
        for section in supplement.sections:
            if not section.text.strip() and not section.title:
                continue
            if section.text in existing_section_texts:
                continue
            merged.sections.append(section)
            existing_section_texts.add(section.text)
        merged.tables.extend(supplement.tables)
        merged.figures.extend(supplement.figures)
        if supplement.references:
            merged.references.extend(ref for ref in supplement.references if ref.strip())
    return merged


def validate_document(document: Document) -> None:
    if not isinstance(document.metadata, dict):
        raise ValueError("Document.metadata must be a dict")
    if "source_path" not in document.metadata:
        raise ValueError("Document.metadata.source_path is required")
    for section in document.sections:
        if section.level < 0:
            raise ValueError("Section.level must be non-negative")
        if not isinstance(section.page_range, tuple):
            raise ValueError("Section.page_range must be a tuple")
        if len(section.page_range) != 2:
            raise ValueError("Section.page_range must have two integers")
    for table in document.tables:
        if not table.id:
            raise ValueError("Tables require stable IDs")


__all__ = [
    "Document",
    "Section",
    "Table",
    "Figure",
    "ExtractorInfo",
    "new_document",
    "compute_content_hash",
    "merge_documents",
    "validate_document",
    "asdict",
]
