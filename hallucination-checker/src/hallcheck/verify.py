"""Indexing and verification routines."""

from __future__ import annotations

import re
from pathlib import Path
from textwrap import shorten
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_session, init_db, reset_db
from .embeddings import embed_texts
from .extract_claims import find_claims
from .chunking import chunk_text
from .ingest_pdf import extract_text
from .models import (
    Claim,
    Document,
    DocumentKind,
    Verdict,
    VerdictStatus,
    Chunk,
)
from .retrieval import (
    INDEX_DIR,
    build_faiss,
    load_faiss,
    load_metadata,
    save_metadata,
    search_faiss,
)


NUMBER_PATTERN = re.compile(r"-?\d[\d,\.]*")


def index_sources(pdf_paths: Sequence[str], index_name: str = "sources") -> dict:
    """Index a collection of source PDFs and return a summary."""
    summary = {"number_of_docs": 0, "number_of_chunks": 0, "index_name": index_name}
    _reset_environment(index_name)
    init_db()
    if not pdf_paths:
        print("No source PDFs provided.")
        return summary

    chunk_records: List[Tuple[Chunk, str]] = []
    documents_ingested = 0

    with get_session() as session:
        for path_str in pdf_paths:
            pdf_path = Path(path_str)
            try:
                text = extract_text(str(pdf_path))
            except Exception as exc:
                print(f"[index] Skipping {pdf_path}: {exc}")
                continue

            if not text.strip():
                print(f"[index] No text extracted from {pdf_path}, skipping.")
                continue

            document = Document(
                kind=DocumentKind.SOURCE,
                path=str(pdf_path.resolve()),
                title=pdf_path.stem,
                year=None,
            )
            session.add(document)
            session.flush()

            segments = chunk_text(text, settings.chunk_tokens, settings.chunk_overlap)
            if not segments:
                print(f"[index] No chunks produced for {pdf_path}.")
                continue

            documents_ingested += 1

            offset = 0
            for segment in segments:
                start = offset
                end = start + len(segment)
                offset = end
                chunk = Chunk(
                    doc_id=document.id,
                    start=start,
                    end=end,
                    text=segment,
                    embedding_dim=None,
                )
                session.add(chunk)
                session.flush()
                chunk_records.append((chunk, segment))

        if not chunk_records:
            print("[index] No chunks persisted; aborting index build.")
            summary["number_of_docs"] = documents_ingested
            return summary

        print(f"[index] Embedding {len(chunk_records)} chunks...")
        embeddings = embed_texts([text for _, text in chunk_records])
        if embeddings.size == 0:
            raise RuntimeError("Embedding creation failed; ensure OpenAI credentials are configured.")
        dim = embeddings.shape[1]

        for chunk, _ in chunk_records:
            chunk.embedding_dim = dim
        session.flush()

        build_faiss(embeddings, dim, index_name)
        doc_ids = [chunk.doc_id for chunk, _ in chunk_records]
        chunk_payloads = [
            {"chunk_id": chunk.id, "doc_id": chunk.doc_id, "text": text}
            for chunk, text in chunk_records
        ]
        save_metadata(index_name, doc_ids, chunk_payloads)
        summary["number_of_docs"] = documents_ingested
        summary["number_of_chunks"] = len(chunk_records)
        print(f"[index] Stored index '{index_name}' with {len(chunk_records)} vectors across {documents_ingested} documents.")

    return summary


def verify_report(report_pdf: str, index_name: str = "sources", topk: int = 5) -> int | None:
    """Verify claims found in a report PDF against the indexed sources."""
    init_db()

    index = load_faiss(index_name)
    doc_ids, chunk_payloads = load_metadata(index_name)
    if len(doc_ids) != len(chunk_payloads):
        raise RuntimeError("Metadata alignment mismatch.")

    report_path = Path(report_pdf)
    report_text = extract_text(str(report_path))
    if not report_text.strip():
        print("[verify] No text extracted from report; aborting.")
        return None

    summaries: List[dict] = []
    report_doc_id: int | None = None

    with get_session() as session:
        _clear_previous_reports(session)

        report_doc = Document(
            kind=DocumentKind.REPORT,
            path=str(report_path.resolve()),
            title=report_path.stem,
            year=None,
        )
        session.add(report_doc)
        session.flush()
        report_doc_id = report_doc.id

        claim_records: List[Tuple[Claim, Dict[str, object]]] = []
        for claim_data in find_claims(report_text):
            claim = Claim(
                doc_id=report_doc.id,
                sentence=str(claim_data.get("sentence", "")),
                value=_safe_float(claim_data.get("value")),
                units=str(claim_data.get("units")) if claim_data.get("units") else None,
                year=int(claim_data["year"]) if claim_data.get("year") else None,
                meta=claim_data.get("meta"),
            )
            if not claim.sentence:
                continue
            session.add(claim)
            session.flush()
            claim_records.append((claim, claim_data))

        if not claim_records:
            print("[verify] No numeric claims detected in report.")
            return report_doc_id

        print(f"[verify] Embedding {len(claim_records)} claims...")
        claim_embeddings = embed_texts([claim.sentence for claim, _ in claim_records])
        if claim_embeddings.size == 0:
            raise RuntimeError("Failed to compute claim embeddings.")

        distances, indices = search_faiss(index, claim_embeddings, k=topk)
        doc_cache: Dict[int, Document] = {}

        for claim_idx, (claim, _) in enumerate(claim_records):
            sims = distances[claim_idx]
            neighbors = indices[claim_idx]
            evidence = _collect_evidence(neighbors, sims, chunk_payloads, doc_cache, session)
            verdict_status, confidence = _decide_verdict(claim, sims, evidence)
            verdict = Verdict(
                claim_id=claim.id,
                status=verdict_status,
                confidence=confidence,
                evidence=evidence,
            )
            session.add(verdict)
            top_snippet = _top_evidence_snippet(evidence)
            summaries.append(
                {
                    "status": verdict_status,
                    "confidence": confidence,
                    "claim_text": claim.sentence,
                    "evidence_text": top_snippet.get("text") if top_snippet else None,
                    "source_label": (top_snippet.get("doc_title") or top_snippet.get("doc_path")) if top_snippet else None,
                }
            )

        print(f"[verify] Generated verdicts for {len(claim_records)} claims.")

    _print_summary(summaries)
    return report_doc_id


def get_verdicts(report_doc_id: int) -> List[dict]:
    """Return verdict data for claims associated with a report document."""
    init_db()
    results: List[dict] = []

    with get_session() as session:
        claims = list(
            session.scalars(
                select(Claim).where(Claim.doc_id == report_doc_id).order_by(Claim.id)
            )
        )

        for claim in claims:
            verdict = claim.verdicts[0] if claim.verdicts else None
            evidence_snippet = None
            source_title = None
            source_doc_id = None

            if verdict and isinstance(verdict.evidence, dict):
                candidates = verdict.evidence.get("candidates")
                if isinstance(candidates, list) and candidates:
                    top_candidate = candidates[0]
                    if isinstance(top_candidate, dict):
                        evidence_snippet = top_candidate.get("text")
                        source_title = top_candidate.get("doc_title") or top_candidate.get("doc_path")
                        source_doc_id = top_candidate.get("doc_id")

            results.append(
                {
                    "claim_id": claim.id,
                    "sentence": claim.sentence,
                    "value": claim.value,
                    "units": claim.units,
                    "year": claim.year,
                    "verdict_status": verdict.status.value if verdict else None,
                    "verdict_confidence": verdict.confidence if verdict else None,
                    "evidence_snippet": evidence_snippet,
                    "source_title": source_title,
                    "source_doc_id": source_doc_id,
                }
            )

    return results


def _collect_evidence(
    neighbors: np.ndarray,
    sims: np.ndarray,
    chunk_payloads: Sequence[dict],
    doc_cache: Dict[int, Document],
    session,
    max_items: int = 5,
) -> Dict[str, object]:
    evidence_list: List[dict] = []
    for rank, idx in enumerate(neighbors[:max_items]):
        if idx < 0 or idx >= len(chunk_payloads):
            continue
        payload = chunk_payloads[int(idx)]
        doc_id = int(payload["doc_id"])
        if doc_id not in doc_cache:
            doc_cache[doc_id] = session.get(Document, doc_id)
        doc = doc_cache[doc_id]
        evidence_list.append(
            {
                "rank": rank,
                "score": float(sims[rank]) if rank < len(sims) else None,
                "doc_id": doc_id,
                "doc_title": doc.title if doc else None,
                "doc_path": doc.path if doc else None,
                "chunk_id": payload.get("chunk_id"),
                "text": payload.get("text"),
            }
        )
    return {"candidates": evidence_list}


def _decide_verdict(
    claim: Claim,
    sims: np.ndarray,
    evidence: Dict[str, object],
) -> Tuple[VerdictStatus, float]:
    best_score = float(sims[0]) if sims.size else 0.0
    candidates = evidence.get("candidates", []) if isinstance(evidence, dict) else []
    top_text = candidates[0]["text"] if candidates else ""

    if best_score < 0.25 or not top_text:
        return VerdictStatus.NOT_FOUND, 0.2

    year_match = True
    if claim.year:
        year_match = str(claim.year) in top_text

    numeric_status = _evaluate_numeric_alignment(claim, top_text)
    if numeric_status == "match":
        confidence = _clamp(0.6 + 0.4 * best_score, 0.0, 0.95)
        if not year_match:
            confidence = _clamp(confidence - 0.2, 0.1, 0.9)
        return VerdictStatus.SUPPORTED, confidence
    if numeric_status == "conflict":
        confidence = _clamp(0.5 + 0.3 * (1 - abs(best_score - 0.5)), 0.2, 0.9)
        if not year_match:
            confidence = _clamp(confidence - 0.1, 0.1, 0.8)
        return VerdictStatus.CONTRADICTED, confidence

    if best_score >= 0.45:
        confidence = _clamp(0.4 + 0.5 * best_score, 0.2, 0.8)
        if not year_match:
            confidence = _clamp(confidence - 0.2, 0.1, 0.7)
        return VerdictStatus.SUPPORTED, confidence

    confidence = _clamp(0.2 + 0.4 * best_score, 0.1, 0.6)
    if not year_match:
        confidence = _clamp(confidence - 0.1, 0.1, 0.5)
    return VerdictStatus.NOT_FOUND, confidence


def _evaluate_numeric_alignment(claim: Claim, evidence_text: str) -> str | None:
    if claim.value is None:
        return None
    numbers = _extract_numbers(evidence_text)
    if not numbers:
        return None

    tolerance_absolute = 0.05
    tolerance_relative = 0.1

    for number in numbers:
        if not _units_match(claim.units, evidence_text):
            continue
        diff = abs(number - claim.value)
        rel = diff / max(abs(claim.value), 1e-6)
        if diff <= tolerance_absolute or rel <= tolerance_relative:
            return "match"

    if _units_match(claim.units, evidence_text):
        return "conflict"
    return None


def _extract_numbers(text: str) -> List[float]:
    numbers: List[float] = []
    for token in NUMBER_PATTERN.findall(text):
        parsed = _parse_number(token)
        if parsed is not None:
            numbers.append(parsed)
    return numbers


def _parse_number(token: str) -> float | None:
    cleaned = token.replace(" ", "")
    if cleaned.count(",") > 1 and "." not in cleaned:
        cleaned = cleaned.replace(",", "")
    else:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _units_match(units: str | None, text: str) -> bool:
    if not units:
        return True
    lowered = text.lower()
    if units == "percent":
        return "%" in text or "percent" in lowered
    if units == "per 100,000":
        return "per 100000" in lowered or "per 100,000" in lowered or "per 100k" in lowered
    return units in lowered


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_evidence_snippet(evidence: Dict[str, object]) -> dict | None:
    if not isinstance(evidence, dict):
        return None
    candidates = evidence.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    return first


def _print_summary(summaries: Sequence[dict]) -> None:
    if not summaries:
        print("[verify] No verdicts to summarize.")
        return

    print("[verify] Verdict summary:")
    for idx, item in enumerate(summaries, start=1):
        status = item.get("status")
        status_label = status.value if isinstance(status, VerdictStatus) else str(status)
        confidence = item.get("confidence") or 0.0
        claim_text = item.get("claim_text") or ""
        evidence_text = item.get("evidence_text")
        source_label = item.get("source_label")

        print(f"  [{idx}] {status_label} (confidence {confidence:.2f})")
        print(f"      report: {shorten(claim_text, width=120, placeholder='...')}")
        if evidence_text:
            snippet = shorten(evidence_text, width=120, placeholder="...")
            if source_label:
                print(f"      source: {snippet}  ← {source_label}")
            else:
                print(f"      source: {snippet}")
        else:
            print("      source: (no evidence snippet available)")


def _reset_environment(index_name: str) -> None:
    """Drop existing database tables and remove prior index artifacts."""
    reset_db()
    _delete_index_artifacts(index_name)
    print(f"[index] Reset previous data for index '{index_name}'.")


def _delete_index_artifacts(index_name: str) -> None:
    targets = [
        INDEX_DIR / f"{index_name}.faiss",
        INDEX_DIR / f"{index_name}_docids.npy",
        INDEX_DIR / f"{index_name}_chunks.jsonl",
    ]
    for path in targets:
        path.unlink(missing_ok=True)


def _clear_previous_reports(session: Session) -> None:
    """Remove previously stored report documents (cascades clean up dependent rows)."""
    reports = list(session.scalars(select(Document).where(Document.kind == DocumentKind.REPORT)))
    for report in reports:
        session.delete(report)
    if reports:
        session.flush()
