"""Indexing and verification routines."""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path
from textwrap import shorten
from typing import Dict, List, Sequence, Tuple, Set, Optional

import numpy as np
from rapidfuzz import fuzz
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_session, init_db, reset_db
from .embeddings import embed_texts
from .extract_claims import find_claims
from .chunking import best_sentence_snippet, chunk_text
from .ingest_pdf import extract_pdf_content
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
from .gpt import OpenAIUnavailable, score_relevance, confirm_entity_alignment
from pdf_pipeline.schema import asdict as document_asdict


NUMBER_PATTERN = re.compile(r"-?\d[\d,\.]*")

_STOPWORDS: Set[str] = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "from",
    "that",
    "this",
    "these",
    "those",
    "as",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "its",
    "their",
    "there",
    "which",
    "who",
    "whom",
    "into",
    "over",
    "under",
    "more",
    "less",
    "than",
    "since",
    "per",
    "each",
    "any",
    "most",
    "many",
    "much",
    "also",
    "because",
    "while",
    "where",
    "when",
    "if",
    "so",
    "but",
    "about",
    "across",
    "such",
}

_GENERIC_ENTITY_TERMS: Set[str] = {
    "country",
    "countries",
    "people",
    "person",
    "population",
    "populations",
    "level",
    "levels",
    "rate",
    "rates",
    "number",
    "numbers",
    "percent",
    "percentage",
    "share",
    "shares",
    "score",
    "scores",
    "index",
    "indexes",
    "indices",
    "value",
    "values",
    "figure",
    "figures",
    "total",
    "overall",
    "global",
    "world",
    "report",
    "table",
    "tables",
    "year",
    "years",
    "region",
    "regions",
    "area",
    "areas",
    "state",
    "states",
    "province",
    "provinces",
    "city",
    "cities",
    "group",
    "groups",
    "category",
    "categories",
}


ALIGNMENT_DEFAULT_CONFIDENCE = 0.55
ALIGNMENT_OVERRIDE_MARGIN = 0.15
ALIGNMENT_PENALTY = 0.12


def _looks_like_table_text(text: str) -> bool:
    lowered = text.lower()
    if "rank" in lowered and "country" in lowered:
        return True
    tokens = text.split()
    if not tokens:
        return False
    numeric_tokens = sum(1 for token in tokens if re.search(r"\d", token))
    return numeric_tokens >= 12 and numeric_tokens / len(tokens) > 0.6


def _looks_like_table_text(text: str) -> bool:
    lowered = text.lower()
    if "rank" in lowered and "country" in lowered:
        return True
    numeric_tokens = sum(1 for token in text.split() if re.search(r"\d", token))
    total_tokens = max(len(text.split()), 1)
    return numeric_tokens / total_tokens > 0.6 and numeric_tokens >= 12


def _normalize_entity_term(term: str) -> str | None:
    cleaned = term.strip("-'\"")
    if len(cleaned) < 3:
        return None
    if cleaned.endswith("ed") or cleaned.endswith("ing"):
        return None
    if cleaned in _STOPWORDS or cleaned in _GENERIC_ENTITY_TERMS:
        return None
    if cleaned.isdigit():
        return None
    if cleaned.endswith("ies") and len(cleaned) > 4:
        cleaned = cleaned[:-3] + "y"
    elif cleaned.endswith("ses") and len(cleaned) > 4:
        cleaned = cleaned[:-2]
    elif cleaned.endswith("s") and not cleaned.endswith("ss") and len(cleaned) > 3:
        cleaned = cleaned[:-1]
    return cleaned


def _extract_entity_terms(text: str) -> Set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    normalized_terms: Set[str] = set()
    filtered_tokens: List[str] = []
    for token in tokens:
        normalized = _normalize_entity_term(token)
        if normalized:
            normalized_terms.add(normalized)
            filtered_tokens.append(normalized)
    # add simple bigrams of filtered tokens for additional context
    for idx in range(len(filtered_tokens) - 1):
        first, second = filtered_tokens[idx], filtered_tokens[idx + 1]
        if first == second:
            continue
        bigram = f"{first} {second}"
        if first not in _GENERIC_ENTITY_TERMS or second not in _GENERIC_ENTITY_TERMS:
            normalized_terms.add(bigram)
    return normalized_terms


def _entity_guard_allows_support(claim_text: str, evidence_text: str) -> bool:
    claim_terms = _extract_entity_terms(claim_text)
    evidence_terms = _extract_entity_terms(evidence_text)
    if not claim_terms or not evidence_terms:
        return True
    if claim_terms & evidence_terms:
        return True
    for claim_term in claim_terms:
        for evidence_term in evidence_terms:
            if fuzz.token_sort_ratio(claim_term, evidence_term) >= 85:
                return True
    return False


def _confirm_entity_alignment(claim_text: str, evidence_text: str) -> Tuple[Optional[bool], Optional[float], Optional[str]]:
    if not claim_text or not evidence_text or not settings.openai_api_key:
        return None, None, None
    try:
        return confirm_entity_alignment(claim_text, evidence_text)
    except OpenAIUnavailable:
        return None, None, None
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[verify] Entity alignment check failed: {exc}")
        return None, None, None


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
                content = extract_pdf_content(str(pdf_path))
                text = content.text
            except Exception as exc:
                print(f"[index] Skipping {pdf_path}: {exc}")
                continue

            if not text.strip():
                print(f"[index] No text extracted from {pdf_path}, skipping.")
                continue

            doc_struct = content.document
            doc_metadata = doc_struct.metadata if doc_struct else {}
            document = Document(
                kind=DocumentKind.SOURCE,
                path=str(pdf_path.resolve()),
                title=content.title or pdf_path.stem,
                author=", ".join(content.authors) if content.authors else None,
                year=content.year,
                router_label=doc_metadata.get("router_label"),
                is_scanned=bool(doc_metadata.get("is_scanned")) if doc_metadata else None,
                content_hash=doc_metadata.get("content_hash"),
                extractor_chain=doc_metadata.get("extractor_chain"),
                document_json=document_asdict(doc_struct) if doc_struct else None,
            )
            session.add(document)
            session.flush()

            segments = chunk_text(
                text,
                settings.chunk_tokens,
                settings.chunk_overlap,
                min_overlap=settings.chunk_overlap_min,
                max_overlap=settings.chunk_overlap_max,
                topic_threshold=settings.chunk_topic_similarity,
                stable_threshold=settings.chunk_stable_similarity,
            )
            if not segments:
                print(f"[index] No chunks produced for {pdf_path}.")
                continue

            documents_ingested += 1

            offset = 0
            for segment in segments:
                segment_tables: List[dict] = []
                clean_segment = segment
                for marker, table_meta in (content.tables or {}).items():
                    if marker in clean_segment:
                        segment_tables.append(dict(table_meta))
                        clean_segment = clean_segment.replace(marker, " ")
                clean_segment = " ".join(clean_segment.split())
                start = offset
                end = start + len(clean_segment)
                offset = end
                chunk = Chunk(
                    doc_id=document.id,
                    start=start,
                    end=end,
                    text=clean_segment,
                    embedding_dim=None,
                    tables=segment_tables or None,
                )
                session.add(chunk)
                session.flush()
                chunk_records.append((chunk, clean_segment))

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
            {
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "text": text,
                "tables": chunk.tables,
            }
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
    report_content = extract_pdf_content(str(report_path))
    report_text = report_content.text
    if not report_text.strip():
        print("[verify] No text extracted from report; aborting.")
        return None

    summaries: List[dict] = []
    report_doc_id: int | None = None

    with get_session() as session:
        _clear_previous_reports(session)

        report_doc_struct = report_content.document
        report_doc_metadata = report_doc_struct.metadata if report_doc_struct else {}
        report_doc = Document(
            kind=DocumentKind.REPORT,
            path=str(report_path.resolve()),
            title=report_content.title or report_path.stem,
            author=", ".join(report_content.authors) if report_content.authors else None,
            year=report_content.year,
            router_label=report_doc_metadata.get("router_label"),
            is_scanned=bool(report_doc_metadata.get("is_scanned")) if report_doc_metadata else None,
            content_hash=report_doc_metadata.get("content_hash"),
            extractor_chain=report_doc_metadata.get("extractor_chain"),
            document_json=document_asdict(report_doc_struct) if report_doc_struct else None,
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
            evidence = _refine_evidence(claim, evidence)
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
                    "evidence_text": (
                        top_snippet.get("snippet") or top_snippet.get("text") if top_snippet else None
                    ),
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
            evidence_full = None
            source_title = None
            source_doc_id = None
            source_author = None
            rerank_score = None
            rerank_label = None
            is_table = False

            if verdict and isinstance(verdict.evidence, dict):
                candidates = verdict.evidence.get("candidates")
                if isinstance(candidates, list) and candidates:
                    top_candidate = candidates[0]
                    if isinstance(top_candidate, dict):
                        is_table = bool(top_candidate.get("is_table"))
                        evidence_snippet = (top_candidate.get("snippet") or top_candidate.get("text")) if not is_table else None
                        evidence_full = None if is_table else top_candidate.get("text")
                        source_title = top_candidate.get("doc_title") or top_candidate.get("doc_path")
                        source_doc_id = top_candidate.get("doc_id")
                        source_author = top_candidate.get("doc_author")
                        rerank_score = top_candidate.get("rerank_score")
                        rerank_label = top_candidate.get("rerank_label")

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
                    "evidence_full_text": evidence_full,
                    "source_title": source_title,
                    "source_doc_id": source_doc_id,
                    "source_author": source_author,
                    "rerank_score": rerank_score,
                    "rerank_label": rerank_label,
                    "is_table": is_table,
                    "evidence_is_table": is_table,
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
        tables = payload.get("tables")
        if payload.get("chunk_id") is not None and tables is None:
            chunk = session.get(Chunk, payload.get("chunk_id"))
            if chunk:
                tables = chunk.tables
        evidence_list.append(
            {
                "rank": rank,
                "score": float(sims[rank]) if rank < len(sims) else None,
                "doc_id": doc_id,
                "doc_title": doc.title if doc else None,
                "doc_path": doc.path if doc else None,
                "doc_author": doc.author if doc else None,
                "chunk_id": payload.get("chunk_id"),
                "text": payload.get("text"),
                "tables": tables,
                "method": payload.get("method") or "faiss",
            }
        )
    return {"candidates": evidence_list}


def _refine_evidence(claim: Claim, evidence: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(evidence, dict):
        return evidence
    candidates = evidence.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return evidence

    for candidate in candidates:
        full_text = candidate.get("text") or ""
        tables = candidate.get("tables")

        if isinstance(tables, list) and tables:
            candidate["snippet"] = full_text.strip()
            candidate["snippet_score"] = 1.0
            candidate["is_table"] = True
            continue

        if _looks_like_table_text(full_text):
            candidate["snippet"] = full_text.strip()
            candidate["snippet_score"] = 1.0
            candidate["is_table"] = True
        else:
            snippet, snippet_score = best_sentence_snippet(claim.sentence, full_text)
            if snippet:
                candidate["snippet"] = snippet
                candidate["snippet_score"] = snippet_score

    candidates = _apply_rerank(claim, candidates)
    evidence["candidates"] = candidates
    return evidence


def _apply_rerank(claim: Claim, candidates: List[dict]) -> List[dict]:
    if not candidates:
        return candidates

    should_rerank = settings.rerank_with_gpt and bool(settings.openai_api_key)
    top_limit = max(0, settings.rerank_max_candidates)
    if should_rerank and top_limit:
        for candidate in candidates[:top_limit]:
            snippet = candidate.get("snippet") or candidate.get("text")
            if not snippet:
                continue
            try:
                score, label = score_relevance(
                    claim_sentence=claim.sentence,
                    evidence_snippet=snippet,
                    retrieval_score=candidate.get("score"),
                )
            except OpenAIUnavailable:
                break
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[verify] Rerank error for chunk {candidate.get('chunk_id')}: {exc}")
                continue
            candidate["rerank_score"] = score
            if label:
                candidate["rerank_label"] = label
            candidate["method"] = "gpt_rerank"

    sorted_candidates = sorted(
        candidates,
        key=lambda c: (
            1 if c.get("rerank_score") is not None else 0,
            c.get("rerank_score") if c.get("rerank_score") is not None else 0.0,
            c.get("snippet_score", 0.0),
            c.get("score", 0.0),
        ),
        reverse=True,
    )
    for idx, candidate in enumerate(sorted_candidates):
        candidate["rank"] = idx
    return sorted_candidates


def _decide_verdict(
    claim: Claim,
    sims: np.ndarray,
    evidence: Dict[str, object],
) -> Tuple[VerdictStatus, float]:
    best_score = float(sims[0]) if sims.size else 0.0
    candidates = evidence.get("candidates", []) if isinstance(evidence, dict) else []
    top_candidate: Optional[dict] = None
    top_text = ""
    top_rerank_score = 0.0
    top_rerank_label: Optional[str] = None
    evaluation_text = ""
    if candidates:
        possible_candidate = candidates[0]
        if isinstance(possible_candidate, dict):
            top_candidate = possible_candidate
            top_text = top_candidate.get("snippet") or top_candidate.get("text") or ""
            evaluation_text = top_candidate.get("text") or top_text
            try:
                if top_candidate.get("rerank_score") is not None:
                    top_rerank_score = float(top_candidate.get("rerank_score"))
            except (TypeError, ValueError):
                top_rerank_score = 0.0
            label = top_candidate.get("rerank_label")
            if isinstance(label, str):
                top_rerank_label = label.lower().strip()

    def _set_candidate_meta(key: str, value) -> None:
        if top_candidate is not None and isinstance(top_candidate, dict):
            top_candidate[key] = value

    if best_score < 0.25 or not top_text:
        return VerdictStatus.NOT_FOUND, 0.2

    evaluation_text = evaluation_text or top_text

    year_match = True
    if claim.year:
        year_match = str(claim.year) in evaluation_text

    numbers_in_evidence: List[float] = []
    if claim.value is not None and evaluation_text:
        numbers_in_evidence = _extract_numbers(evaluation_text)

    numeric_status = _evaluate_numeric_alignment(claim, evaluation_text, numbers_in_evidence or None)
    if numeric_status == "match":
        _set_candidate_meta("numeric_match", "exact")
    elif numeric_status == "match_sum":
        _set_candidate_meta("numeric_match", "sum")
    elif numeric_status == "conflict":
        _set_candidate_meta("numeric_match", "conflict")
    else:
        _set_candidate_meta("numeric_match", None)

    candidate_status: Optional[VerdictStatus] = None
    candidate_confidence: float = 0.0

    if numeric_status in {"match", "match_sum"}:
        candidate_status = VerdictStatus.SUPPORTED
        candidate_confidence = _clamp(0.6 + 0.4 * best_score, 0.0, 0.95)
        if numeric_status == "match_sum":
            candidate_confidence = _clamp(candidate_confidence - 0.15, 0.1, 0.85)
    elif numeric_status == "conflict":
        candidate_status = VerdictStatus.CONTRADICTED
        candidate_confidence = _clamp(0.5 + 0.3 * (1 - abs(best_score - 0.5)), 0.2, 0.9)
    else:
        if claim.value is not None:
            if not numbers_in_evidence:
                gpt_supports = bool(top_rerank_label == "supported" and top_rerank_score >= 0.7)
                gpt_contradicts = bool(top_rerank_label == "contradicted" and top_rerank_score >= 0.6)
                if gpt_supports:
                    candidate_status = VerdictStatus.SUPPORTED
                    candidate_confidence = _clamp(0.35 + 0.35 * best_score, 0.15, 0.65)
                    _set_candidate_meta("verdict_source", "gpt_support")
                elif gpt_contradicts:
                    candidate_status = VerdictStatus.CONTRADICTED
                    candidate_confidence = _clamp(0.3 + 0.3 * best_score, 0.15, 0.6)
                    _set_candidate_meta("verdict_source", "gpt_contradict")
                else:
                    confidence = _clamp(0.2 + 0.3 * best_score, 0.1, 0.5)
                    if not year_match:
                        confidence = _clamp(confidence - 0.1, 0.1, 0.5)
                    return VerdictStatus.NOT_FOUND, confidence
            else:
                gpt_supports = bool(top_rerank_label == "supported" and top_rerank_score >= 0.7)
                gpt_contradicts = bool(top_rerank_label == "contradicted" and top_rerank_score >= 0.6)
                if gpt_supports:
                    candidate_status = VerdictStatus.SUPPORTED
                    candidate_confidence = _clamp(0.35 + 0.35 * best_score, 0.15, 0.7)
                    _set_candidate_meta("verdict_source", "gpt_support")
                elif gpt_contradicts:
                    candidate_status = VerdictStatus.CONTRADICTED
                    candidate_confidence = _clamp(0.3 + 0.3 * best_score, 0.15, 0.65)
                    _set_candidate_meta("verdict_source", "gpt_contradict")
                else:
                    confidence = _clamp(0.25 + 0.3 * best_score, 0.1, 0.55)
                    if not year_match:
                        confidence = _clamp(confidence - 0.1, 0.1, 0.5)
                    return VerdictStatus.NOT_FOUND, confidence

        if candidate_status is None:
            if best_score >= 0.45:
                candidate_status = VerdictStatus.SUPPORTED
                candidate_confidence = _clamp(0.4 + 0.5 * best_score, 0.2, 0.8)
            else:
                confidence = _clamp(0.2 + 0.4 * best_score, 0.1, 0.6)
                if not year_match:
                    confidence = _clamp(confidence - 0.1, 0.1, 0.5)
                return VerdictStatus.NOT_FOUND, confidence

    if candidate_status is None:
        confidence = _clamp(0.2 + 0.4 * best_score, 0.1, 0.6)
        if not year_match:
            confidence = _clamp(confidence - 0.1, 0.1, 0.5)
        return VerdictStatus.NOT_FOUND, confidence

    if not year_match:
        if candidate_status == VerdictStatus.SUPPORTED:
            candidate_confidence = _clamp(candidate_confidence - 0.2, 0.1, 0.9)
        elif candidate_status == VerdictStatus.CONTRADICTED:
            candidate_confidence = _clamp(candidate_confidence - 0.1, 0.1, 0.8)

    if not _entity_guard_allows_support(claim.sentence, evaluation_text):
        fallback_confidence = _clamp(0.2 + 0.3 * best_score, 0.1, 0.55)
        return VerdictStatus.NOT_FOUND, fallback_confidence

    alignment_match, alignment_confidence, alignment_reason = _confirm_entity_alignment(claim.sentence, evaluation_text)
    effective_alignment_conf = alignment_confidence if alignment_confidence is not None else ALIGNMENT_DEFAULT_CONFIDENCE
    _set_candidate_meta("alignment_reason", alignment_reason)

    if alignment_match is False:
        override_threshold = max(candidate_confidence - ALIGNMENT_OVERRIDE_MARGIN, 0.3)
        if effective_alignment_conf >= override_threshold:
            _set_candidate_meta("alignment_match", False)
            _set_candidate_meta("alignment_confidence", effective_alignment_conf)
            _set_candidate_meta("alignment_vetoed", True)
            _set_candidate_meta("verdict_source", "alignment_override")
            fallback_confidence = _clamp(min(candidate_confidence, effective_alignment_conf), 0.1, 0.6)
            return VerdictStatus.NOT_FOUND, fallback_confidence
        candidate_confidence = _clamp(candidate_confidence - ALIGNMENT_PENALTY, 0.1, 0.9)
        _set_candidate_meta("alignment_match", False)
        _set_candidate_meta("alignment_confidence", effective_alignment_conf)
        _set_candidate_meta("alignment_vetoed", False)
    elif alignment_match is True:
        if effective_alignment_conf >= candidate_confidence + 0.05:
            candidate_confidence = _clamp((candidate_confidence + effective_alignment_conf) / 2, candidate_confidence, 0.9)
        elif effective_alignment_conf <= 0.35:
            candidate_confidence = _clamp(candidate_confidence - ALIGNMENT_PENALTY / 2, 0.1, 0.9)
        _set_candidate_meta("alignment_match", True)
        _set_candidate_meta("alignment_confidence", effective_alignment_conf)
        _set_candidate_meta("alignment_vetoed", False)
    else:
        _set_candidate_meta("alignment_match", None)
        _set_candidate_meta("alignment_confidence", effective_alignment_conf)

    _set_candidate_meta("final_confidence", candidate_confidence)
    if top_candidate is not None and isinstance(top_candidate, dict) and "verdict_source" not in top_candidate:
        top_candidate["verdict_source"] = "heuristic"

    return candidate_status, candidate_confidence


def _evaluate_numeric_alignment(
    claim: Claim,
    evidence_text: str,
    numbers: List[float] | None = None,
) -> str | None:
    if claim.value is None:
        return None
    if numbers is None:
        numbers = _extract_numbers(evidence_text)
    if not numbers:
        return None

    tolerance_absolute = 0.05
    tolerance_relative = 0.1
    conflict_relative = 0.12

    for number in numbers:
        if not _units_match(claim.units, evidence_text):
            continue
        diff = abs(number - claim.value)
        rel = diff / max(abs(claim.value), 1e-6)
        if diff <= tolerance_absolute or rel <= tolerance_relative:
            return "match"

    if _units_match(claim.units, evidence_text):
        if _numbers_sum_to_claim(
            claim.value,
            numbers,
            tolerance_absolute=tolerance_absolute,
            tolerance_relative=tolerance_relative,
        ):
            return "match_sum"
        for number in numbers:
            diff = abs(number - claim.value)
            rel = diff / max(abs(claim.value), 1e-6)
            if rel <= conflict_relative:
                return "conflict"
    return None


def _extract_numbers(text: str) -> List[float]:
    numbers: List[float] = []
    for token in NUMBER_PATTERN.findall(text):
        parsed = _parse_number(token)
        if parsed is not None:
            numbers.append(parsed)
    return numbers


def _numbers_sum_to_claim(
    claim_value: float,
    numbers: Sequence[float],
    *,
    tolerance_absolute: float,
    tolerance_relative: float,
) -> bool:
    claim_abs = abs(claim_value)
    if claim_abs == 0:
        return False
    upper_bound = max(claim_abs * 1.5, claim_abs + tolerance_absolute)
    filtered = [abs(num) for num in numbers if 0 < abs(num) <= upper_bound]
    if not filtered:
        return False
    max_subset = min(len(filtered), 4)
    for subset_size in range(2, max_subset + 1):
        for combo in combinations(filtered, subset_size):
            total = sum(combo)
            diff = abs(total - claim_abs)
            rel = diff / max(claim_abs, 1e-6)
            if diff <= tolerance_absolute or rel <= tolerance_relative:
                return True
    return False


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
    if "snippet" not in first and isinstance(first.get("text"), str):
        snippet, _ = best_sentence_snippet("", first.get("text", ""))
        if snippet:
            first["snippet"] = snippet
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
