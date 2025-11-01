"""Helpers bridging the CLI core logic with the web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from sqlalchemy import select

from .config import settings
from .db import get_session, init_db
from .gpt import OpenAIUnavailable, generate_explanation
from .table_formatter import format_table_html, ensure_table_classes
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


HIGH_CONFIDENCE_THRESHOLD = 0.75


def _candidate_confidence_value(candidate: Optional[dict]) -> float:
    if not candidate:
        return 0.0
    value = candidate.get("rerank_score")
    if value is not None:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
    value = candidate.get("score")
    if value is not None:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
    value = candidate.get("snippet_score")
    if value is not None:
        try:
            # snippet_score is a fuzz ratio (0-100)
            return max(0.0, min(1.0, float(value) / 100.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _candidate_numeric_label(candidate: Optional[dict]) -> str | None:
    if not candidate:
        return None
    numeric = candidate.get("numeric_match")
    if numeric in {"exact", "sum", "conflict"}:
        return numeric
    return None


def _candidate_method(method: Optional[str], candidate: Optional[dict]) -> str:
    if method:
        return method
    if candidate and candidate.get("rerank_score") is not None:
        return "gpt_rerank"
    return "faiss"


def _method_label(method: str) -> str:
    if method == "gpt_rerank":
        return "GPT rerank"
    if method == "faiss":
        return "FAISS similarity"
    return method.replace("_", " ").title()


def _alignment_label(candidate: Optional[dict]) -> str | None:
    if not candidate:
        return None
    value = candidate.get("alignment_match")
    if value is None:
        return None
    return "match" if value else "mismatch"


def _render_table_html(structured_tables: Optional[List[dict]], snippet: Optional[str], full_text: Optional[str]) -> List[str]:
    tables: List[str] = []
    if structured_tables:
        for meta in structured_tables:
            if not isinstance(meta, dict):
                continue
            table_text = meta.get("text")
            caption = meta.get("caption")
            formatted = None
            if settings.format_tables_with_gpt and table_text:
                try:
                    formatted = format_table_html(table_text, caption=caption)
                except OpenAIUnavailable:
                    formatted = None
            if formatted:
                tables.append(ensure_table_classes(formatted))
                continue
            if table_text:
                tables.extend(ensure_table_classes(html) for html in extract_tables_from_text(table_text))
            html_repr = meta.get("html")
            if html_repr:
                tables.append(ensure_table_classes(html_repr))

    if not tables:
        fallback_text = full_text or snippet or ""
        tables.extend(ensure_table_classes(html) for html in extract_tables_from_text(fallback_text))

    deduped: List[str] = []
    seen: Set[str] = set()
    for item in tables:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)

    if settings.format_tables_with_gpt and deduped:
        return [deduped[0]]
    return deduped


def _format_percent(value: Optional[float], *, default: str = "n/a", decimals: int = 1) -> str:
    if value is None:
        return default
    try:
        scaled = round(float(value) * 100, decimals)
    except (TypeError, ValueError):
        return default
    if decimals == 0:
        return f"{int(scaled)}%"
    return f"{scaled:.{decimals}f}%"


def _describe_relevance(rerank_score: Optional[float], rerank_label: Optional[str]) -> Tuple[str, str]:
    value = _format_percent(rerank_score)
    if rerank_label:
        value = f"{value} ({rerank_label})"
    description = "Combined FAISS + GPT reranker score for this snippet."
    return value, description


def _describe_numeric(numeric_match: Optional[str], verdict_source: Optional[str]) -> Tuple[str, str]:
    mapping = {
        "exact": ("Exact match", "The snippet states the same number as the claim."),
        "sum": ("Sum matches", "Numbers in the snippet add up to the claim value."),
        "conflict": ("Conflict", "The snippet contains a different number."),
    }
    if numeric_match in mapping:
        return mapping[numeric_match]
    if verdict_source == "gpt_support":
        return ("No explicit number", "We relied on GPT reranker support even though the number isn't stated.")
    if verdict_source == "gpt_contradict":
        return ("GPT contradicted", "GPT judged the snippet as contradicting the claim.")
    return ("No numeric evidence", "No matching number was detected in the snippet.")


def _describe_alignment(
    alignment_label: Optional[str],
    alignment_confidence: Optional[float],
    alignment_reason: Optional[str],
    alignment_vetoed: bool,
) -> Tuple[str, str]:
    if alignment_label == "match":
        value = "Match"
        description = "GPT alignment judged the claim and snippet to discuss the same entities."
    elif alignment_label == "mismatch":
        value = "Mismatch"
        if alignment_vetoed:
            description = "Alignment mismatch was strong enough to override the evidence."
        else:
            description = "Alignment mismatch was detected but not strong enough to override the evidence." 
    else:
        value = "Unknown"
        description = "Alignment model was inconclusive or not executed."

    if alignment_confidence is not None:
        value = f"{value} ({_format_percent(alignment_confidence)})"
    if alignment_reason:
        description = f"{description} {alignment_reason}"
    return value, description


def _describe_method(method_label: Optional[str]) -> Tuple[str, str]:
    value = method_label or "n/a"
    description = "Retrieval method that surfaced this snippet."
    if value.lower() == "gpt rerank":
        description = "Snippet was reranked by GPT after FAISS retrieval."
    return value, description


def _build_decision_rationale(
    status: str,
    confidence: Optional[float],
    candidate: Optional[dict],
) -> Optional[str]:
    status = (status or "UNKNOWN").upper()
    if not candidate:
        if status == "NOT_FOUND":
            return "No evidence snippet met the retrieval threshold, so the claim remains NOT_FOUND."
        return None

    rerank_score = candidate.get("rerank_score")
    numeric_match = candidate.get("numeric_match")
    verdict_source = candidate.get("verdict_source")
    alignment_match = candidate.get("alignment_match")
    alignment_conf = candidate.get("alignment_confidence")
    alignment_vetoed = candidate.get("alignment_vetoed")

    rerank_percent = _format_percent(rerank_score)
    final_conf = _format_percent(confidence)

    if alignment_vetoed:
        return (
            f"Alignment judged the snippet as mismatched ({_format_percent(alignment_conf)}), so the claim stays {status} "
            f"despite the snippet scoring {rerank_percent} on relevance."
        )

    if status == "SUPPORTED" and verdict_source == "gpt_support":
        return (
            f"The reranker strongly supported this snippet ({rerank_percent}) even without an explicit number, "
            f"so the claim is SUPPORTED with moderated confidence ({final_conf})."
        )

    if status == "CONTRADICTED" and numeric_match == "conflict":
        return "The snippet reports a different number than the claim, so the verdict is CONTRADICTED."

    if verdict_source == "gpt_contradict":
        return (
            f"GPT judged the snippet as contradictory ({rerank_percent}), leading to a {status} verdict."
        )

    if status == "NOT_FOUND" and rerank_score and rerank_score >= 0.7 and not numeric_match:
        return (
            f"The snippet looks relevant ({rerank_percent}) but lacks numeric confirmation, so the claim remains NOT_FOUND."
        )

    if alignment_match is False and alignment_conf is not None:
        return (
            f"Alignment confidence was low ({_format_percent(alignment_conf)}), so the claim stays {status} with {final_conf} confidence."
        )

    if status == "SUPPORTED" and numeric_match == "sum":
        return "Numbers in the snippet add up to the claim value, supporting the verdict."

    if status == "SUPPORTED" and numeric_match == "exact":
        return "The snippet states the same number as the claim and alignment looks consistent."

    numeric_value, numeric_desc = _describe_numeric(numeric_match, verdict_source)
    _, alignment_desc = _describe_alignment(
        candidate.get("alignment_label"),
        alignment_conf,
        candidate.get("alignment_reason"),
        alignment_vetoed,
    )

    return (
        f"Numeric check: {numeric_value}. {numeric_desc} {alignment_desc} "
        f"Overall confidence is {final_conf}."
    )


def _build_summary_metrics(
    status: str,
    confidence: Optional[float],
    candidate: Optional[dict],
) -> Tuple[List[dict], Optional[str]]:
    metrics: List[dict] = []
    metrics.append(
        {
            "label": "Verdict confidence",
            "value": _format_percent(confidence),
            "description": "Final confidence after numeric heuristics and alignment arbitration.",
        }
    )

    if candidate:
        relevance_value, relevance_desc = _describe_relevance(candidate.get("rerank_score"), candidate.get("rerank_label"))
        metrics.append(
            {
                "label": "Evidence relevance",
                "value": relevance_value,
                "description": relevance_desc,
            }
        )

        numeric_value, numeric_desc = _describe_numeric(candidate.get("numeric_match"), candidate.get("verdict_source"))
        metrics.append(
            {
                "label": "Numeric check",
                "value": numeric_value,
                "description": numeric_desc,
            }
        )

        alignment_value, alignment_desc = _describe_alignment(
            candidate.get("alignment_label"),
            candidate.get("alignment_confidence"),
            candidate.get("alignment_reason"),
            candidate.get("alignment_vetoed"),
        )
        metrics.append(
            {
                "label": "Alignment",
                "value": alignment_value,
                "description": alignment_desc,
            }
        )

        method_value, method_desc = _describe_method(candidate.get("method_label"))
        metrics.append(
            {
                "label": "Retrieval method",
                "value": method_value,
                "description": method_desc,
            }
        )
    else:
        metrics.append(
            {
                "label": "Evidence relevance",
                "value": "n/a",
                "description": "No evidence snippet was retrieved for this claim.",
            }
        )

    rationale = _build_decision_rationale(status, confidence, candidate)
    return metrics, rationale


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


def _build_candidate_payloads(session, verdict: Optional[Verdict]) -> List[dict]:
    payloads: List[dict] = []
    for idx, raw_candidate in enumerate(_candidate_list(verdict)):
        (
            snippet,
            full_text,
            source_title,
            source_doc_id,
            source_author,
            structured_tables,
            rerank_score,
            rerank_label,
        ) = _resolve_candidate_details(session, raw_candidate)

        method = _candidate_method(raw_candidate.get("method"), raw_candidate)
        confidence_value = _candidate_confidence_value(raw_candidate)
        is_table = bool(raw_candidate.get("is_table"))
        table_html = _render_table_html(structured_tables, snippet, full_text)
        confidence_percent = round(confidence_value * 100, 1) if confidence_value is not None else None
        numeric_match_raw = raw_candidate.get("numeric_match")
        numeric_label = _candidate_numeric_label(raw_candidate)
        alignment_match = raw_candidate.get("alignment_match")
        alignment_label = _alignment_label(raw_candidate)
        alignment_confidence = raw_candidate.get("alignment_confidence")
        alignment_confidence_percent = (
            round(float(alignment_confidence) * 100, 1)
            if isinstance(alignment_confidence, (int, float))
            else None
        )
        display_snippet = None if is_table else snippet
        payloads.append(
            {
                "index": idx,
                "snippet": display_snippet,
                "raw_snippet": snippet,
                "full_text": full_text,
                "display_full_text": None if is_table else full_text,
                "context_text": (full_text or snippet or "")[:2000],
                "is_table": is_table,
                "method": method,
                "method_label": _method_label(method),
                "confidence_value": confidence_value,
                "confidence_percent": confidence_percent,
                "rerank_score": rerank_score,
                "rerank_label": rerank_label,
                "numeric_match": numeric_match_raw,
                "numeric_label": numeric_label,
                "alignment_match": alignment_match,
                "alignment_label": alignment_label,
                "alignment_confidence": alignment_confidence,
                "alignment_confidence_percent": alignment_confidence_percent,
                "alignment_reason": raw_candidate.get("alignment_reason"),
                "alignment_vetoed": raw_candidate.get("alignment_vetoed") or False,
                "verdict_source": raw_candidate.get("verdict_source"),
                "score": raw_candidate.get("score"),
                "snippet_score": raw_candidate.get("snippet_score"),
                "source_title": source_title,
                "source_doc_id": source_doc_id,
                "source_author": source_author,
                "source_path": raw_candidate.get("doc_path"),
                "table_html": table_html,
                "structured_tables": structured_tables,
                "doc_label": source_title or raw_candidate.get("doc_path") or "Source",
            }
        )
    return payloads


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

        candidate_payloads = _build_candidate_payloads(session, verdict)

        primary_candidate = candidate_payloads[0] if candidate_payloads else None
        additional_candidates = [
            payload for payload in candidate_payloads[1:] if (payload.get("confidence_value") or 0.0) >= HIGH_CONFIDENCE_THRESHOLD
        ]

        status = verdict.status.value if verdict and verdict.status else "UNKNOWN"
        confidence = float(verdict.confidence) if verdict and verdict.confidence is not None else None

        if primary_candidate:
            detail = {
                "claim_id": claim.id,
                "sentence": claim.sentence,
                "value": claim.value,
                "units": claim.units,
                "year": claim.year,
                "report_doc_id": claim.doc_id,
                "status": status,
                "confidence": confidence,
                "evidence_text": primary_candidate["snippet"],
                "evidence_full_text": primary_candidate["display_full_text"],
                "source_title": primary_candidate["source_title"],
                "source_doc_id": primary_candidate["source_doc_id"],
                "source_author": primary_candidate["source_author"],
                "rerank_score": primary_candidate["rerank_score"],
                "rerank_label": primary_candidate["rerank_label"],
                "explanation": verdict.explanation if verdict else None,
                "is_table": primary_candidate["is_table"],
                "evidence_is_table": primary_candidate["is_table"],
                "evidence_tables": primary_candidate["table_html"],
                "evidence_method": primary_candidate["method"],
                "evidence_method_label": primary_candidate["method_label"],
                "primary_candidate_confidence_percent": primary_candidate["confidence_percent"],
                "primary_candidate_index": primary_candidate["index"],
            }
        else:
            detail = {
                "claim_id": claim.id,
                "sentence": claim.sentence,
                "value": claim.value,
                "units": claim.units,
                "year": claim.year,
                "report_doc_id": claim.doc_id,
                "status": status,
                "confidence": confidence,
                "evidence_text": None,
                "evidence_full_text": None,
                "source_title": None,
                "source_doc_id": None,
                "source_author": None,
                "rerank_score": None,
                "rerank_label": None,
                "explanation": verdict.explanation if verdict else None,
                "is_table": False,
                "evidence_is_table": False,
                "evidence_tables": [],
                "evidence_method": None,
                "evidence_method_label": None,
                "primary_candidate_confidence_percent": None,
                "primary_candidate_index": None,
            }

        candidate_options: List[dict] = []
        if primary_candidate:
            candidate_options.append(
                {
                    "index": primary_candidate["index"],
                    "label": f"Primary • {primary_candidate['method_label']}",
                    "method_label": primary_candidate["method_label"],
                    "confidence_percent": primary_candidate["confidence_percent"],
                    "doc_label": primary_candidate["doc_label"],
                    "is_primary": True,
                    "confidence_value": primary_candidate["confidence_value"],
                }
            )
        for payload in additional_candidates:
            candidate_options.append(
                {
                    "index": payload["index"],
                    "label": f"{payload['method_label']} • {payload['doc_label']}",
                    "method_label": payload["method_label"],
                    "confidence_percent": payload["confidence_percent"],
                    "doc_label": payload["doc_label"],
                    "is_primary": False,
                    "confidence_value": payload["confidence_value"],
                }
            )

        detail["candidates"] = candidate_payloads
        detail["additional_candidates"] = additional_candidates
        detail["candidate_options"] = candidate_options
        summary_metrics, summary_rationale = _build_summary_metrics(status, confidence, primary_candidate)
        detail["summary_metrics"] = summary_metrics
        detail["summary_rationale"] = summary_rationale
        return detail


def fetch_candidate_preview(claim_id: int, candidate_index: int) -> dict:
    """Return data for a specific evidence candidate, including a fresh explanation when requested."""
    init_db()
    with get_session() as session:
        claim = session.get(Claim, claim_id)
        if not claim:
            raise ValueError(f"Claim {claim_id} not found.")

        verdict = claim.verdicts[0] if claim.verdicts else session.scalar(
            select(Verdict).where(Verdict.claim_id == claim_id)
        )
        if not verdict:
            raise ValueError(f"No verdict stored for claim {claim_id}.")

        candidate_payloads = _build_candidate_payloads(session, verdict)
        if not candidate_payloads:
            raise ValueError("No evidence candidates available for this claim.")
        if candidate_index < 0 or candidate_index >= len(candidate_payloads):
            raise ValueError(f"Candidate index {candidate_index} is out of range.")

        target = candidate_payloads[candidate_index]

        verdict_status = verdict.status.value if verdict and verdict.status else None
        verdict_confidence = float(verdict.confidence) if verdict and verdict.confidence is not None else None
        cached_explanation = verdict.explanation if verdict else None

        claim_sentence = claim.sentence
        claim_value = claim.value
        claim_units = claim.units
        claim_year = claim.year

    explanation: Optional[str] = None
    explanation_cached = False

    if candidate_index == 0:
        if cached_explanation:
            explanation = cached_explanation
            explanation_cached = True
        else:
            explanation = get_or_make_explanation(claim_id)
            explanation_cached = True
    else:
        if not settings.openai_api_key:
            raise RuntimeError("OpenAI API key is not configured.")
        payload = {
            "claim_sentence": claim_sentence,
            "verdict_status": verdict_status,
            "confidence": verdict_confidence,
            "evidence_text": target.get("context_text") or "",
            "full_context": target.get("context_text") or "",
            "value": claim_value,
            "units": claim_units,
            "year": claim_year,
        }
        try:
            explanation = generate_explanation(payload)
        except OpenAIUnavailable as exc:
            raise RuntimeError(str(exc)) from exc

    summary_metrics, summary_rationale = _build_summary_metrics(verdict_status or "UNKNOWN", verdict_confidence, target)

    return {
        "candidate": target,
        "explanation": explanation,
        "explanation_cached": explanation_cached,
        "verdict_status": verdict_status,
        "verdict_confidence": verdict_confidence,
        "summary": {"metrics": summary_metrics, "rationale": summary_rationale},
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
        document_json = report.document_json or {}
        sections = document_json.get("sections", []) if isinstance(document_json, dict) else []
        metadata = document_json.get("metadata", {}) if isinstance(document_json, dict) else {}
        executive_summary = next((section for section in sections if section.get("type") == "executive_summary"), None)
        return {
            "id": report.id,
            "title": report.title or Path(report.path).stem,
            "path": report.path,
            "filename": Path(report.path).name,
            "author": report.author,
            "year": report.year,
            "router_label": report.router_label or metadata.get("router_label"),
            "is_scanned": report.is_scanned if report.is_scanned is not None else metadata.get("is_scanned"),
            "extractor_chain": report.extractor_chain or metadata.get("extractor_chain", []),
            "document_sections": sections,
            "executive_summary": executive_summary,
            "document_metadata": metadata,
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


def _candidate_list(verdict: Optional[Verdict]) -> List[dict]:
    if not verdict or not isinstance(verdict.evidence, dict):
        return []
    candidates = verdict.evidence.get("candidates")
    if not isinstance(candidates, list):
        return []
    result: List[dict] = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            result.append(candidate)
    return result


def _top_candidate(verdict: Optional[Verdict]) -> Optional[dict]:
    candidates = _candidate_list(verdict)
    return candidates[0] if candidates else None


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
