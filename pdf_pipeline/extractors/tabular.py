"""Table extraction helper integrating Camelot/Tabula."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

try:  # pragma: no cover - optional dependency
    import camelot
except ImportError:  # pragma: no cover - optional dependency
    camelot = None

try:  # pragma: no cover - optional dependency
    import tabula  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tabula = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from ..config import PDFPipelineConfig, pipeline_config
from ..schema import Document, ExtractorInfo, Table, new_document
from .base import Extractor

logger = logging.getLogger(__name__)


class TableExtractor(Extractor):
    name = "tabular"
    version = "0.2"

    def extract(self, pdf_path: str, *, ocr_pdf_path: str | None = None) -> Tuple[Document, Dict[str, Any]]:
        cfg = self.config
        working_path = ocr_pdf_path or pdf_path
        metrics: Dict[str, Any] = {"engine": None, "tables": 0}

        if not cfg.tables_enabled:
            return new_document(pdf_path), metrics

        document = new_document(pdf_path)
        table_items: List[Table] = []

        engine_preference = self._engine_preference(cfg)
        for engine in engine_preference:
            try:
                if engine == "camelot":
                    tables = self._extract_with_camelot(working_path)
                elif engine == "tabula":
                    tables = self._extract_with_tabula(working_path)
                else:
                    tables = self._extract_with_pdfplumber(working_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Table extraction via %s failed: %s", engine, exc)
                continue

            if tables:
                table_items = tables
                metrics["engine"] = engine
                break

        document.tables = table_items
        metrics["tables"] = len(table_items)
        document.metadata["extractor_chain"].append({"name": self.name, "version": self.version})
        return document, metrics

    def _engine_preference(self, cfg: PDFPipelineConfig) -> List[str]:
        if cfg.tables_engine == "camelot":
            return ["camelot", "pdfplumber"]
        if cfg.tables_engine == "tabula":
            return ["tabula", "pdfplumber"]
        return ["camelot", "tabula", "pdfplumber"]

    def _extract_with_camelot(self, pdf_path: str) -> List[Table]:
        if camelot is None:
            return []
        tables: List[Table] = []
        try:
            result = camelot.read_pdf(pdf_path, pages="all", flavor="lattice", suppress_console=True)
        except Exception:
            result = []
        for idx, table in enumerate(result, start=1):
            try:
                html = table.to_html()
            except Exception:
                html = None
            text_rows = ["\t".join(row) for row in table.data]
            table_id = f"TABLE-{idx}"
            tables.append(
                Table(
                    id=table_id,
                    page=int(getattr(table, "page", 1) or 1),
                    caption=None,
                    html=html,
                    text="\n".join(text_rows),
                    bbox=None,
                )
            )
        return tables

    def _extract_with_tabula(self, pdf_path: str) -> List[Table]:
        if tabula is None:
            return []
        try:
            dataframes = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)  # type: ignore
        except Exception:
            return []
        tables: List[Table] = []
        for idx, df in enumerate(dataframes, start=1):
            html = df.to_html(index=False)
            text = df.to_csv(index=False)
            tables.append(
                Table(
                    id=f"TABLE-{idx}",
                    page=idx,
                    caption=None,
                    html=html,
                    text=text,
                    bbox=None,
                )
            )
        return tables

    def _extract_with_pdfplumber(self, pdf_path: str) -> List[Table]:
        if pdfplumber is None:
            return []
        tables: List[Table] = []
        with pdfplumber.open(pdf_path) as pdf:
            counter = 1
            for page_number, page in enumerate(pdf.pages, start=1):
                extracted = page.extract_tables()
                for table_data in extracted or []:
                    rows = ["\t".join(cell or "" for cell in row) for row in table_data]
                    html_rows = "".join(
                        "<tr>" + "".join(f"<td>{(cell or '').strip()}</td>" for cell in row) + "</tr>"
                        for row in table_data
                    )
                    html = (
                        "<table class=\"evidence-table table table-striped table-sm\"><tbody>"
                        f"{html_rows}</tbody></table>"
                    )
                    tables.append(
                        Table(
                            id=f"TABLE-{counter}",
                            page=page_number,
                            caption=None,
                            html=html,
                            text="\n".join(rows),
                            bbox=None,
                        )
                    )
                    counter += 1
        return tables


__all__ = ["TableExtractor"]
