"""Helpers bridging the CLI core logic with the web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from sqlalchemy import select

from .config import settings
from .db import get_session, init_db
from .models import Document
from .verify import get_verdicts, index_sources, verify_report


def _inputs_dir() -> Path:
    root = settings.project_root or Path(__file__).resolve().parents[2]
    inputs = root / "data" / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    return inputs


def _normalize_paths(paths: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for path_str in paths:
        if not path_str:
            continue
        candidate = Path(path_str)
        if not candidate.is_absolute():
            candidate = _inputs_dir() / candidate
        if candidate.exists() and candidate.suffix.lower() == ".pdf":
            normalized.append(str(candidate.resolve()))
    # Deduplicate while preserving order
    seen: Dict[str, bool] = {}
    ordered: List[str] = []
    for item in normalized:
        if item not in seen:
            seen[item] = True
            ordered.append(item)
    return ordered


def run_index(source_paths: List[str], index_name: str = "sources") -> dict:
    """Validate and index the provided source PDFs."""
    normalized = _normalize_paths(source_paths)
    if not normalized:
        raise ValueError("No valid PDF source files provided.")
    return index_sources(normalized, index_name=index_name)


def run_verify(report_path: str, index_name: str = "sources", topk: int = 5) -> int:
    """Run the verification workflow for a report PDF."""
    normalized = _normalize_paths([report_path])
    if not normalized:
        raise ValueError("Report PDF not found or unsupported format.")
    report_doc_id = verify_report(normalized[0], index_name=index_name, topk=topk)
    if report_doc_id is None:
        raise RuntimeError("Verification could not be completed.")
    return report_doc_id


def fetch_results(report_doc_id: int) -> List[dict]:
    """Return verdict data for a report document."""
    return get_verdicts(report_doc_id)


def get_report_details(report_doc_id: int) -> dict | None:
    """Fetch metadata about the report document for display purposes."""
    init_db()
    with get_session() as session:
        report = session.scalar(select(Document).where(Document.id == report_doc_id))
        if not report:
            return None
        return {
            "id": report.id,
            "title": report.title or Path(report.path).stem,
            "path": report.path,
            "filename": Path(report.path).name,
        }


def list_existing_pdfs() -> List[dict]:
    """Enumerate PDFs available under the inputs directory."""
    inputs = _inputs_dir()
    items = []
    for pdf_path in sorted(inputs.glob("*.pdf")):
        items.append(
            {
                "label": pdf_path.name,
                "path": pdf_path.name,
                "full_path": str(pdf_path.resolve()),
            }
        )
    return items
