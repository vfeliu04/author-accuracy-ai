# PDF Pipeline Overview

This package provides the second generation PDF ingestion pipeline used by the hallucination checker application. It is organised into three layers:

### Routing & OCR
- `router.route` inspects the original PDF to determine high-level characteristics (e.g., report vs. scholarly paper, scanned-only pages, table-heavy content).
- `ocr.ocr_if_needed` uses `ocrmypdf` (optional via `pdf.ocr.enabled`) to add a text layer when the router flags low text density or image-only pages.

### Extraction
- `extractors.general_layout` relies on local tooling (PyMuPDF/pdfplumber + layoutparser) to recover text blocks, headings, sections, figures and captions.
- `extractors.unstructured_wrap` preserves the previous Unstructured-based pipeline and normalises its output into the unified schema.
- `extractors.tabular` runs Camelot/Tabula to capture structured tables for pages identified as table-heavy (GPT formatting is used later for display when enabled).
- `extractors.scholarly_optional` can enrich scholarly articles with data from optional services such as GROBID (guarded by configuration).

All extractors emit a `Document` object defined in `schema.py`. The general extractor produces the main body; specialised extractors merge additional detail where they offer higher confidence.

### Post-processing & Integration
- `postprocess.header_footer`, `postprocess.headings`, and `postprocess.cleaning` remove repeating headers/footers, build heading hierarchies, and normalise text.
- `ingest_pdf_v2` orchestrates the router, OCR, and extractors. It returns the structured `Document`, the plain body text with `[[TABLE-n]]` placeholders, and a table map compatible with the legacy chunking/verification stack (table text can be reformatted to HTML via GPT when `TABLES_FORMAT_WITH_GPT` is true).
- Configuration flags live in `config.py`. They are loaded from environment variables and surface toggles for OCR, extractor enablement, table engines, header/footer stripping, and thresholds.

See `tests/pdf_pipeline` for synthetic fixtures and unit tests covering the router, OCR integration, layout extraction heuristics, and header/footer removal.
