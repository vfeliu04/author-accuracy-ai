"""
Metadata enrichment helpers for credibility scoring.
"""

from __future__ import annotations

import requests
from pathlib import Path
from typing import Dict, Any, Optional

from ..config import get_settings
from .logger import setup_logger

import re
from PyPDF2 import PdfReader  # type: ignore


logger = setup_logger(__name__)


class MetadataService:
    def __init__(self):
        self.settings = get_settings()

    def fetch_embedded_metadata(self, path: Path) -> Dict[str, Any]:
        reader = PdfReader(str(path))
        info = reader.metadata
        data = {
            "title": (info.title or path.stem.replace("_", " ").title()) if info else path.stem,
            "authors": [info.author] if info and info.author else [],
        }
        return data

    def extract_header_metadata(self, path: Path) -> Dict[str, Any]:
        reader = PdfReader(str(path))
        pages = reader.pages[: min(2, len(reader.pages))]
        text = "\n".join([page.extract_text() or "" for page in pages])
        doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.IGNORECASE)
        author_match = re.search(r"Author(?:s)?[:\-]\s*(.+)", text, re.IGNORECASE)
        published_match = re.search(r"Published[:\-]\s*([A-Za-z0-9, ]+)", text, re.IGNORECASE)
        metadata = {}
        if doi_match:
            metadata["doi"] = doi_match.group(0)
        if author_match:
            metadata.setdefault("authors", [author_match.group(1).strip()])
        if published_match:
            metadata["publication_date"] = published_match.group(1).strip()
        return metadata

    def fetch_crossref(self, doi: str) -> Optional[Dict[str, Any]]:
        headers = {"User-Agent": "AuthorAI/0.1 (mailto:dev@example.com)"}
        response = requests.get(f"https://api.crossref.org/works/{doi}", headers=headers, timeout=10)
        if response.status_code != 200:
            logger.warning("Crossref lookup failed for DOI %s: %s", doi, response.status_code)
            return None
        return response.json().get("message")

    def collect_metadata(self, path: Path) -> Dict[str, Any]:
        embedded = self.fetch_embedded_metadata(path)
        header = self.extract_header_metadata(path)
        merged = {**embedded}
        merged.update({k: v for k, v in header.items() if v})

        doi = merged.get("doi")
        if doi:
            crossref = self.fetch_crossref(doi)
            if crossref:
                merged["title"] = crossref.get("title", [merged.get("title")])[0]
                merged["publisher"] = crossref.get("publisher")
                merged["publication_date"] = "-".join(map(str, crossref.get("published", {}).get("date-parts", [[None]])[0]))
                merged["confidence"] = "HIGH"
        if "confidence" not in merged:
            merged["confidence"] = "MEDIUM" if header else "LOW"
        return merged


METADATA = MetadataService()
