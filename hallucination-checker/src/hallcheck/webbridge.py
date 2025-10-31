"""Helpers bridging the CLI core logic with the web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from sqlalchemy import select

from .config import settings
from .db import get_session, init_db
from .gpt import OpenAIUnavailable, generate_explanation
from .models import Chunk, Claim, Document, Verdict
from .tables import extract_tables_from_text
from .verify import get_verdicts, index_sources, verify_report


def _inputs_dir() -> Path:
    root = settings.project_root or Path(__file__).resolve().parents[2]
    inputs = root / "data" / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    return inputs


def _sources_dir() -> Path:
    sources = _inputs_dir() / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    return sources


def _reports_dir() -> Path:
    reports = _inputs_dir() / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return reports


def _normalize_paths(paths: Iterable[str], *, kind: str | None = None) -> List[str]:
    normalized: List[str] = []
    for path_str in paths:
        if not path_str:
            continue
        candidate = Path(path_str)
        if not candidate.is_absolute():
            parts = candidate.parts
            if parts and parts[0] in {"sources", "reports"}:
                first, *rest = parts
                base = _sources_dir() if first == "sources" else _reports_dir()
                candidate = base / Path(*rest) if rest else base
            else:
                base = _inputs_dir()
                if kind == "source":
                    base = _sources_dir()
                elif kind == "report":
                    base = _reports_dir()
                candidate = base / candidate
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
    normalized = _normalize_paths(source_paths, kind="source")
    if not normalized:
        raise ValueError("No valid PDF source files provided.")
    return index_sources(normalized, index_name=index_name)


def run_verify(report_path: str, index_name: str = "sources", topk: int = 5) -> int:
    """Run the verification workflow for a report PDF."""
    normalized = _normalize_paths([report_path], kind="report")
    if not normalized:
        raise ValueError("Report PDF not found or unsupported format.")
    report_doc_id = verify_report(normalized[0], index_name=index_name, topk=topk)
    if report_doc_id is None:
        raise RuntimeError("Verification could not be completed.")
    return report_doc_id


def fetch_results(report_doc_id: int) -> List[dict]:
    """Return verdict data for a report document."""
    return get_verdicts(report_doc_id)


def fetch_claim_detail(claim_id: int) -> dict:
    """Return detailed information about a single claim, its verdict, and evidence."""
    init_db()
    with get_session() as session:
        claim = session.get(Claim, claim_id)
        if not claim:
            raise ValueError(f"Claim {claim_id} not found.")

        verdict = claim.verdicts[0] if claim.verdicts else session.scalar(
            select(Verdict).where(Verdict.claim_id == claim_id)
        )

        candidate = _top_candidate(verdict)
        (
            snippet,
            full_text,
            source_title,
            source_doc_id,
            source_author,
            structured_tables,
            rerank_score,
            rerank_label,
        ) = _resolve_candidate_details(session, candidate)

        status = verdict.status.value if verdict and verdict.status else "UNKNOWN"
        confidence = float(verdict.confidence) if verdict and verdict.confidence is not None else None

        is_table = bool(candidate.get("is_table")) if isinstance(candidate, dict) else False
        detail = {
            "claim_id": claim.id,
            "sentence": claim.sentence,
            "value": claim.value,
            "units": claim.units,
            "year": claim.year,
            "report_doc_id": claim.doc_id,
            "status": status,
            "confidence": confidence,
            "evidence_text": snippet,
            "evidence_full_text": None if is_table else full_text,
            "source_title": source_title,
            "source_doc_id": source_doc_id,
            "source_author": source_author,
            "rerank_score": rerank_score,
            "rerank_label": rerank_label,
            "explanation": verdict.explanation if verdict else None,
            "is_table": is_table,
            "evidence_is_table": is_table,
        }

        table_html: List[str] = []
        seen_html: set[str] = set()
        if structured_tables:
            for table_meta in structured_tables:
                if not isinstance(table_meta, dict):
                    continue
                html_repr = table_meta.get("html")
                text_repr = table_meta.get("text")
                if html_repr and html_repr not in seen_html:
                    table_html.append(html_repr)
                    seen_html.add(html_repr)
                elif text_repr:
                    for html_candidate in extract_tables_from_text(text_repr):
                        if html_candidate not in seen_html:
                            table_html.append(html_candidate)
                            seen_html.add(html_candidate)

        source_text = full_text or snippet or ""
        for html_candidate in extract_tables_from_text(source_text):
            if html_candidate not in seen_html:
                table_html.append(html_candidate)
                seen_html.add(html_candidate)

        detail["evidence_tables"] = table_html
        return detail


def get_or_make_explanation(claim_id: int) -> str:
    """Return (or generate) an explanation for a claim's verdict."""
    if not settings.openai_api_key:
        raise RuntimeError("OpenAI API key is not configured.")

    init_db()
    with get_session() as session:
        claim = session.get(Claim, claim_id)
        if not claim:
            raise ValueError(f"Claim {claim_id} not found.")

        verdict = claim.verdicts[0] if claim.verdicts else session.scalar(
            select(Verdict).where(Verdict.claim_id == claim_id)
        )
        if not verdict:
            raise ValueError(f"No verdict found for claim {claim_id}.")

        if verdict.explanation:
            return verdict.explanation

        candidate = _top_candidate(verdict)
        snippet, full_text, _, _, _, _, _, _ = _resolve_candidate_details(session, candidate)
        evidence_for_prompt = (full_text or snippet or "")[:2000]

        payload = {
            "claim_sentence": claim.sentence,
            "verdict_status": verdict.status.value if verdict and verdict.status else None,
            "confidence": float(verdict.confidence) if verdict and verdict.confidence is not None else None,
            "evidence_text": evidence_for_prompt,
            "full_context": evidence_for_prompt,
            "value": claim.value,
            "units": claim.units,
            "year": claim.year,
        }

        try:
            explanation = generate_explanation(payload)
        except OpenAIUnavailable as exc:
            raise RuntimeError(str(exc)) from exc
        verdict.explanation = explanation
        session.add(verdict)
        return explanation


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
            "author": report.author,
            "year": report.year,
        }


def _list_pdfs_in_dir(directory: Path, *, prefix: str) -> List[dict]:
    items: List[dict] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        items.append(
            {
                "label": pdf_path.name,
                "path": f"{prefix}/{pdf_path.name}",
                "full_path": str(pdf_path.resolve()),
            }
        )
    return items


def list_source_pdfs() -> List[dict]:
    return _list_pdfs_in_dir(_sources_dir(), prefix="sources")


def list_report_pdfs() -> List[dict]:
    return _list_pdfs_in_dir(_reports_dir(), prefix="reports")


def _top_candidate(verdict: Optional[Verdict]) -> Optional[dict]:
    if not verdict or not isinstance(verdict.evidence, dict):
        return None
    candidates = verdict.evidence.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    return first if isinstance(first, dict) else None


def _resolve_candidate_details(
    session, candidate: Optional[dict]
) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[int],
    Optional[str],
    Optional[List[dict]],
    Optional[float],
    Optional[str],
]:
    if not candidate:
        return None, None, None, None, None, None, None, None

    snippet: Optional[str] = candidate.get("snippet")
    evidence_text: Optional[str] = candidate.get("text")
    source_title: Optional[str] = None
    source_doc_id: Optional[int] = None
    source_author: Optional[str] = candidate.get("doc_author")
    structured_tables: Optional[List[dict]] = candidate.get("tables")

    source_doc_id_raw = candidate.get("doc_id")
    if isinstance(source_doc_id_raw, int):
        source_doc_id = source_doc_id_raw
    else:
        try:
            source_doc_id = int(source_doc_id_raw)
        except (TypeError, ValueError):
            source_doc_id = None

    source_title = candidate.get("doc_title") or candidate.get("doc_path")

    chunk_id_raw = candidate.get("chunk_id")
    chunk_id: Optional[int]
    if isinstance(chunk_id_raw, int):
        chunk_id = chunk_id_raw
    else:
        try:
            chunk_id = int(chunk_id_raw)
        except (TypeError, ValueError):
            chunk_id = None

    if chunk_id is not None:
        chunk = session.get(Chunk, chunk_id)
        if chunk:
            if not evidence_text:
                evidence_text = chunk.text
            if structured_tables is None:
                structured_tables = chunk.tables

    if evidence_text is None:
        text_value = candidate.get("text")
        evidence_text = text_value if isinstance(text_value, str) else None

    if source_doc_id is not None:
        document = session.get(Document, source_doc_id)
        if document:
            if source_title is None:
                source_title = document.title or document.path
            if not source_author:
                source_author = document.author

    effective_snippet = snippet or evidence_text

    return (
        effective_snippet,
        evidence_text,
        source_title,
        source_doc_id,
        source_author,
        structured_tables,
        candidate.get("rerank_score"),
        candidate.get("rerank_label"),
    )


def clear_explanation(claim_id: int) -> None:
    """Remove any cached explanation for the given claim."""
    init_db()
    with get_session() as session:
        verdict = session.scalar(select(Verdict).where(Verdict.claim_id == claim_id))
        if not verdict:
            raise ValueError(f"No verdict found for claim {claim_id}.")
        verdict.explanation = None
        session.add(verdict)
