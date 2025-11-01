"""Wrapper around the legacy Unstructured pipeline."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

from ..schema import Document, ExtractorInfo, Section, Table, new_document
from .base import Extractor


class UnstructuredExtractor(Extractor):
    name = "unstructured"
    version = "0.2"

    def extract(self, pdf_path: str, *, ocr_pdf_path: str | None = None) -> Tuple[Document, Dict[str, Any]]:
        module = importlib.import_module("src.hallcheck.ingest_pdf")
        legacy_func = getattr(module, "legacy_extract_pdf_content", None)
        if legacy_func is None:
            legacy_func = getattr(module, "extract_pdf_content")

        legacy_content = legacy_func(pdf_path)  # type: ignore[call-arg]
        document = new_document(pdf_path)
        document.metadata["extractor_chain"].append({"name": self.name, "version": self.version})
        document.title = legacy_content.title
        if legacy_content.authors:
            document.authors = list(legacy_content.authors)
        if legacy_content.year:
            document.date = str(legacy_content.year)

        text = legacy_content.text.strip()
        sections = [
            Section(
                title=None,
                level=1,
                type="legacy_fulltext",
                text=text,
                page_range=(1, 1),
                confidence=0.5 if text else 0.0,
            )
        ] if text else []
        document.sections = sections

        tables: Dict[str, dict] = legacy_content.tables or {}
        table_items = []
        counter = 1
        for placeholder, meta in tables.items():
            table_id = placeholder.strip("[]")
            if not table_id:
                table_id = f"TABLE-{counter}"
            counter += 1
            table_items.append(
                Table(
                    id=table_id,
                    page=int(meta.get("page_number") or 1),
                    caption=meta.get("caption"),
                    html=meta.get("html"),
                    text=meta.get("text"),
                    bbox=None,
                )
            )
        document.tables = table_items

        metrics = {
            "tables": len(table_items),
            "sections": len(sections),
        }
        return document, metrics


__all__ = ["UnstructuredExtractor"]
