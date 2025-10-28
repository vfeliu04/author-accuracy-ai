from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader


# This function is responsible for reading a PDF file and extracting the text from each page.
def extract_text(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract (page_number, text) tuples from a PDF file."""
    # The `pypdf` library is used to open and read the PDF.
    reader = PdfReader(str(pdf_path))
    pages: List[Tuple[int, str]] = []
    # We loop through each page in the PDF.
    # `enumerate` with `start=1` gives us a page number starting from 1.
    for index, page in enumerate(reader.pages, start=1):
        # `extract_text()` gets the text content from the page.
        # We use `or ""` as a fallback in case a page has no extractable text.
        text = page.extract_text() or ""
        # We store the result as a tuple: (page number, page text).
        pages.append((index, text.strip()))
    return pages
