"""PDF ingestion helpers."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF using PyMuPDF."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(path)
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "\n".join(pages).strip()

