"""Helpers bridging the CLI core logic with the web UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from sqlalchemy import select

from .config import settings
from .db import get_session, init_db
from .models import Chunk, Claim, Document, Verdict
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
        evidence_text, source_title, source_doc_id = _resolve_candidate_details(session, candidate)

        status = verdict.status.value if verdict and verdict.status else "UNKNOWN"
        confidence = float(verdict.confidence) if verdict and verdict.confidence is not None else None

        return {
            "claim_id": claim.id,
            "sentence": claim.sentence,
            "value": claim.value,
            "units": claim.units,
            "year": claim.year,
            "report_doc_id": claim.doc_id,
            "status": status,
            "confidence": confidence,
            "evidence_text": evidence_text,
            "source_title": source_title,
            "source_doc_id": source_doc_id,
            "explanation": verdict.explanation if verdict else None,
        }


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
        evidence_text, _, _ = _resolve_candidate_details(session, candidate)
        evidence_for_prompt = (evidence_text or "")[:2000]

        payload = {
            "claim_sentence": claim.sentence,
            "verdict_status": verdict.status.value if verdict and verdict.status else None,
            "confidence": float(verdict.confidence) if verdict and verdict.confidence is not None else None,
            "evidence_text": evidence_for_prompt,
            "value": claim.value,
            "units": claim.units,
            "year": claim.year,
        }

        client = OpenAI(api_key=settings.openai_api_key or None)
        explanation = _call_openai_for_explanation(client, payload)
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
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    if not candidate:
        return None, None, None

    evidence_text: Optional[str] = None
    source_title: Optional[str] = None
    source_doc_id: Optional[int] = None

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
            evidence_text = chunk.text

    if evidence_text is None:
        text_value = candidate.get("text")
        evidence_text = text_value if isinstance(text_value, str) else None

    if source_title is None and source_doc_id is not None:
        document = session.get(Document, source_doc_id)
        if document:
            source_title = document.title or document.path

    return evidence_text, source_title, source_doc_id


def clear_explanation(claim_id: int) -> None:
    """Remove any cached explanation for the given claim."""
    init_db()
    with get_session() as session:
        verdict = session.scalar(select(Verdict).where(Verdict.claim_id == claim_id))
        if not verdict:
            raise ValueError(f"No verdict found for claim {claim_id}.")
        verdict.explanation = None
        session.add(verdict)


def _call_openai_for_explanation(client: OpenAI, payload: dict) -> str:
    """Generate an explanation using the best available OpenAI endpoint."""
    system_prompt = (
        "You are a precise scientific verifier. Explain in ≤150 words why the verdict was assigned, "
        "focusing on numeric values, units, and years. If numbers mismatch, show both values and explain relative difference."
    )

    if hasattr(client, "responses"):
        response = client.responses.create(
            model=settings.openai_chat_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0,
            max_output_tokens=350,
        )
        explanation = _extract_responses_text(response)
        if explanation:
            return explanation

    chat_model = settings.openai_chat_model
    if chat_model.startswith("gpt-4.1"):
        chat_model = "gpt-4o-mini"

    if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
        raise RuntimeError(
            "Installed openai SDK does not support Responses API or chat completions. Upgrade to openai>=1.2.0."
        )

    completion = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        max_tokens=350,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    choice = (completion.choices or [None])[0]
    if not choice or not getattr(choice, "message", None):
        raise RuntimeError("OpenAI chat completion returned no choices.")
    content = getattr(choice.message, "content", None)
    if not content:
        raise RuntimeError("OpenAI chat completion returned an empty message.")
    explanation = content.strip()
    if not explanation:
        raise RuntimeError("OpenAI chat completion produced blank output.")
    return explanation


def _extract_responses_text(response) -> Optional[str]:
    text = getattr(response, "output_text", None)
    if text:
        stripped = text.strip()
        if stripped:
            return stripped

    output = getattr(response, "output", None)
    if not output:
        return None
    fragments: List[str] = []
    try:
        for item in output:
            for part in getattr(item, "content", []):
                fragment = getattr(part, "text", None)
                if fragment:
                    fragments.append(fragment)
    except Exception:
        return None

    joined = "".join(fragments).strip()
    return joined or None
