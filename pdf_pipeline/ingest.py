"""High level orchestration for the mixed PDF extraction pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import PDFPipelineConfig, pipeline_config
from .ocr import ocr_if_needed
from .router import route
from .schema import Document, Table, compute_content_hash, merge_documents, new_document, validate_document
from .extractors.general_layout import GeneralLayoutExtractor
from .extractors.tabular import TableExtractor
from .extractors.unstructured_wrap import UnstructuredExtractor
from .extractors.scholarly_optional import ScholarlyExtractor

logger = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    pass


def ingest_pdf_v2(pdf_path: str, config: Optional[PDFPipelineConfig] = None) -> Tuple[Document, str, Dict[str, dict], Dict[str, object]]:
    cfg = config or pipeline_config
    pdf_path = str(Path(pdf_path).resolve())
    decision = route(pdf_path)
    ocr_pdf, ocr_used = ocr_if_needed(pdf_path, decision, config=cfg)
    working_path = ocr_pdf or pdf_path

    base_document = new_document(pdf_path)
    base_document.metadata.update(decision.features)
    base_document.metadata["router_label"] = decision.label
    base_document.metadata["is_scanned"] = bool(decision.features.get("is_scanned"))
    base_document.metadata["ocr_used"] = ocr_used

    extractors: list[Tuple[str, Document, Dict[str, object]]] = []

    if cfg.layout_enabled:
        layout_extractor = GeneralLayoutExtractor(cfg)
        try:
            layout_doc, layout_metrics = layout_extractor.extract(working_path, ocr_pdf_path=ocr_pdf)
            extractors.append(("layout", layout_doc, layout_metrics))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Layout extraction failed: %s", exc)

    if cfg.unstructured_enabled:
        unstructured_extractor = UnstructuredExtractor(cfg)
        try:
            unstructured_doc, unstructured_metrics = unstructured_extractor.extract(pdf_path)
            extractors.append(("unstructured", unstructured_doc, unstructured_metrics))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unstructured extraction failed: %s", exc)

    if cfg.tables_enabled:
        table_extractor = TableExtractor(cfg)
        try:
            table_doc, table_metrics = table_extractor.extract(working_path, ocr_pdf_path=ocr_pdf)
            extractors.append(("tabular", table_doc, table_metrics))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Table extraction failed: %s", exc)

    if cfg.scholarly_enabled:
        scholarly_extractor = ScholarlyExtractor(cfg)
        try:
            scholarly_doc, scholarly_metrics = scholarly_extractor.extract(working_path, ocr_pdf_path=ocr_pdf)
            extractors.append(("scholarly", scholarly_doc, scholarly_metrics))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Scholarly extraction failed: %s", exc)

    documents = [doc for _, doc, _ in extractors]
    merged_document = merge_documents(base_document, documents)
    seen_chain = set()
    chain = []
    for supplement in documents:
        for info in supplement.metadata.get("extractor_chain", []):
            key = (info.get("name"), info.get("version"))
            if key not in seen_chain:
                seen_chain.add(key)
                chain.append(info)
    merged_document.metadata["extractor_chain"] = chain
    merged_document.metadata["content_hash"] = compute_content_hash(Path(pdf_path).read_bytes())

    validate_document(merged_document)

    body_text, table_map = build_body_text_with_tables(merged_document)

    metrics = {
        "router": decision.features,
        "ocr_used": ocr_used,
        "extractors": {name: metrics for name, _, metrics in extractors},
    }

    logger.info(
        "pdf_ingest_v2 completed",
        extra={
            "pdf_path": pdf_path,
            "router_label": decision.label,
            "ocr_used": ocr_used,
            "section_count": len(merged_document.sections),
            "table_count": len(merged_document.tables),
        },
    )

    if ocr_pdf:
        Path(ocr_pdf).unlink(missing_ok=True)

    return merged_document, body_text, table_map, metrics


def build_body_text_with_tables(document: Document) -> Tuple[str, Dict[str, dict]]:
    lines: list[str] = []
    table_map: Dict[str, dict] = {}

    tables_by_page = {}
    for table in document.tables:
        tables_by_page.setdefault(table.page, []).append(table)
        table_map[table.id] = {
            "id": table.id,
            "html": table.html,
            "text": table.text,
            "caption": table.caption,
            "page_number": table.page,
            "bbox": table.bbox,
        }

    primary_sections = [section for section in document.sections if section.type != "legacy_fulltext"]
    sections_iterable = primary_sections if primary_sections else document.sections

    for section in sections_iterable:
        if section.title:
            lines.append(section.title.strip())
        if section.text:
            lines.append(section.text.strip())
        start_page, end_page = section.page_range
        relevant_tables = []
        for page in range(start_page, end_page + 1):
            relevant_tables.extend(tables_by_page.get(page, []))
        for table in relevant_tables:
            lines.append(f"[[{table.id}]]")

    body_text = "\n\n".join(line for line in lines if line)
    return body_text, table_map


__all__ = ["ingest_pdf_v2", "build_body_text_with_tables", "IngestionError"]
