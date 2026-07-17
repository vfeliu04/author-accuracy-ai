"""
High-level accuracy pipeline coordinator.
"""

from __future__ import annotations

import functools
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

from ..models import Claim, ClaimEvidence, _now_iso
from ..storage.database import Repository
from ..config import get_settings
from ..services.logger import setup_logger
from ..services.vector_store import VectorStore
from ..services.reranker import EvidenceReranker
from ..services.verdict_classifier import VerdictClassifier
from ..services.section_indexer import SECTION_INDEXER
from .ingestion import IngestionPipeline


logger = setup_logger(__name__)


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_COMPOUND_SPLIT = re.compile(r",\s+(?:while|whereas|but|however|although|though|yet|and)\s+", re.IGNORECASE)
_NON_VERIFIABLE_STARTS = re.compile(
    r"^(the report|this report|figure\s+\d|table\s+\d|as shown in|see figure|see table)",
    re.IGNORECASE,
)
_VAGUE_ONLY = re.compile(
    r"^(data shows?|research shows?|it is noted|it was found)",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"-?\d[\d,\.]*")
VERB_HINTS = re.compile(
    r"\b(is|are|was|were|has|have|had|be|been|being|shows?|shown|reported|reports?|estimates?|estimated|projects?|projected|expects?|expected|forecasts?|forecasted|increase|increased|decrease|decreased|rise|rose|fell|falling|grew|grow|remains?)\b",
    re.IGNORECASE,
)
UNIT_OR_YEAR = re.compile(r"(percent|%|million|billion|thousand|20\d{2}|19\d{2})", re.IGNORECASE)


class AccuracyPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.ingestion = IngestionPipeline()
        self.repo = Repository()
        self.vector_store = VectorStore("sources")  # stores embedded source chunks for retrieval
        self.reranker = EvidenceReranker()  # GPT-based reranker (haystack cross-encoder removed)
        self.verdict_classifier = VerdictClassifier()
        self._section_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._section_cache_lock = threading.Lock()

    def index_source(self, upload: dict) -> Dict[str, Any]:
        pdf_path = Path(upload["path"]).resolve()
        payload = self.ingestion.ingest(pdf_path, doc_id=upload["upload_id"], doc_type="SOURCE")
        self._persist_ingestion_payload(payload, upload, doc_type="SOURCE")
        self._index_chunks(payload["chunks"])
        logger.info("Indexed source %s with %d chunks", pdf_path, len(payload["chunks"]))
        return payload

    def verify_report(self, upload: dict) -> Dict[str, Any]:
        pdf_path = Path(upload["path"]).resolve()
        report_id = upload["upload_id"]
        payload = self.ingestion.ingest(pdf_path, doc_id=report_id, doc_type="REPORT")
        self._persist_ingestion_payload(payload, upload, doc_type="REPORT")
        sections = (payload.get("document") or {}).get("sections") or []
        claims = self._extract_claims(payload["body_text"], report_id, sections)
        claims = self._filter_verifiable_claims(claims)
        claims = self._decompose_claims(claims)
        source_doc_ids = self.repo.list_source_doc_ids()
        evidence_rows: List[Dict[str, Any]] = []
        max_workers = self.settings.pipeline_max_workers
        retrieve_fn = functools.partial(self._retrieve_evidence, source_doc_ids=source_doc_ids)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(retrieve_fn, claim) for claim in claims]
            for future in as_completed(futures):
                evidence_rows.extend(future.result())

        claim_dicts = [
            {
                "claim_id": claim.claim_id,
                "report_id": report_id,
                "text": claim.text,
                "verdict": claim.verdict,
                "confidence": claim.confidence,
                "confidence_band": claim.confidence_band,
                "explanation": claim.explanation,
                "processing_mode": getattr(claim, "processing_mode", "unknown"),
                "metadata": claim.metadata,
            }
            for claim in claims
        ]
        self.repo.insert_claims(claim_dicts)
        if evidence_rows:
            self.repo.insert_evidence(evidence_rows)

        self._persist_claim_index(report_id, claims)
        self._persist_source_index(report_id, evidence_rows)

        supported = sum(1 for claim in claims if claim.verdict == "SUPPORTED")
        contradicted = sum(1 for claim in claims if claim.verdict == "CONTRADICTED")
        not_found = len(claims) - supported - contradicted
        accuracy_pct = (supported / len(claims) * 100) if claims else 0.0
        summary_text = (
            f"Accuracy is {accuracy_pct:.1f}% ({supported} of {len(claims)} claims supported, "
            f"{contradicted} contradicted, {not_found} not found)."
        )
        self.repo.update_document_metadata(
            report_id,
            {
                "accuracy_summary": summary_text,
                "accuracy_supported": supported,
                "accuracy_contradicted": contradicted,
                "accuracy_not_found": not_found,
            },
        )
        logger.info("Extracted %d claims from %s", len(claims), pdf_path)
        return {"report_id": report_id, "claims": [claim.to_summary() for claim in claims]}

    def _rule_split_claim(self, text: str) -> List[str]:
        parts = _COMPOUND_SPLIT.split(text)
        if len(parts) <= 1:
            return [text]
        parts_with_verb = [p for p in parts if VERB_HINTS.search(p)]
        if len(parts_with_verb) < 2:
            return [text]
        return parts

    def _decompose_claims(self, claims: List[Claim]) -> List[Claim]:
        if not self.settings.claim_decompose_enabled:
            return claims
        result: List[Claim] = []
        for claim in claims:
            parts = self._rule_split_claim(claim.text)
            if len(parts) > 1:
                decomposed_any = False
                for part in parts:
                    part = part.strip()
                    if len(part.split()) >= 4:
                        new_claim = Claim(
                            claim_id=str(uuid.uuid4()),
                            text=part,
                            verdict="NOT_EVALUATED",
                            confidence=0.4,
                            confidence_band="LOW",
                            processing_mode="unknown",
                            explanation="Awaiting retrieval alignment.",
                            metadata={
                                **claim.metadata,
                                "decomposed_from": claim.claim_id,
                            },
                            evidence=[],
                        )
                        result.append(new_claim)
                        decomposed_any = True
                if decomposed_any:
                    logger.debug("Decomposed claim %s into %d parts.", claim.claim_id, len(parts))
                    continue
            result.append(claim)
        return result

    def _filter_verifiable_claims(self, claims: List[Claim]) -> List[Claim]:
        kept = []
        dropped = 0
        for claim in claims:
            text = claim.text
            if _NON_VERIFIABLE_STARTS.match(text):
                dropped += 1
                continue
            if _VAGUE_ONLY.match(text) and not any(ch.isdigit() for ch in text):
                dropped += 1
                continue
            kept.append(claim)
        if dropped:
            logger.info("Claim quality pre-filter dropped %d claim(s), %d kept.", dropped, len(kept))
        if self.settings.claim_quality_llm_filter and kept:
            kept = self._llm_batch_filter(kept)
        return kept

    def _llm_batch_filter(self, claims: List[Claim]) -> List[Claim]:
        import json as _json
        numbered = "\n".join(f"{i+1}. {c.text}" for i, c in enumerate(claims))
        system_prompt = (
            "You are a claim verifiability classifier. "
            "For each numbered claim below, decide if it is VERIFIABLE (a specific, checkable factual assertion) "
            "or NOT_VERIFIABLE (vague, subjective, procedural, or not checkable). "
            "Respond with a JSON array only: [{\"n\": 1, \"label\": \"VERIFIABLE\"}, ...]"
        )
        user_content = f"Claims:\n{numbered}\n\nJSON array:"
        try:
            client = self.verdict_classifier.client
            if not client:
                return claims
            response = client.messages.create(
                model=self.settings.claim_classifier_model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            text_blocks = [b.text for b in response.content if b.type == "text"]
            content = text_blocks[0].strip() if text_blocks else ""
            results = _json.loads(content)
            verifiable_indices = {item["n"] - 1 for item in results if (item.get("label") or "").upper() == "VERIFIABLE"}
            filtered = [c for i, c in enumerate(claims) if i in verifiable_indices]
            logger.info("LLM batch filter kept %d of %d claims.", len(filtered), len(claims))
            return filtered
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM batch filter failed, returning all claims: %s", exc)
            return claims

    def _extract_claims(self, text: str, report_id: str, sections: List[Dict[str, Any]]) -> List[Claim]:
        def _strip_bullet_prefix(value: str) -> str:
            # Remove common list/section markers like "1.", "2)", "5.2" when they look like headings.
            trimmed = value.lstrip()
            m = re.match(r"^([0-9]{1,3}(?:\.[0-9]{1,3})*)([.)]?)\s+([A-Z][A-Za-z].*)$", trimmed)
            if m:
                remainder = m.group(3)
                return remainder.strip()
            return re.sub(r"^\s*(?:[\-\u2022\*])\s*", "", value)

        def _heading_prefix_without_numbers(value: str) -> bool:
            """
            Detect sentences that start with a section-like heading (e.g., '3.3 Supply Chain...')
            where the remainder has no digits; these should be skipped entirely.
            """
            trimmed = value.lstrip()
            match = re.match(r"^([0-9]{1,3}(?:\.[0-9]{1,3})*)([.)]?\s+)(.*)$", trimmed)
            if not match:
                return False
            remainder = match.group(3)
            return not any(ch.isdigit() for ch in remainder)

        def _has_alpha(value: str) -> bool:
            return any(ch.isalpha() for ch in value)

        def _find_parent(sentence: str) -> Dict[str, Any] | None:
            for section in sections:
                if sentence in (section.get("text") or ""):
                    return {
                        "id": section.get("id"),
                        "title": section.get("title"),
                        "page": section.get("page"),
                    }
            return None

        doc_metadata = (self.repo.get_document(report_id) or {}).get("metadata") or {}

        def _find_parent_summary(parent_id: str | None) -> str | None:
            if not parent_id:
                return None
            sections_detail = doc_metadata.get("sections_detail") or []
            for section in sections_detail:
                if section.get("id") == parent_id:
                    summary = section.get("summary")
                    if summary:
                        return summary
                    if self.settings.section_summary_mode.lower() == "lazy" and SECTION_INDEXER.enabled:
                        try:
                            generated = SECTION_INDEXER.summarize_sections([section]).get(parent_id)
                        except Exception:
                            generated = None
                        if generated:
                            section["summary"] = generated
                            self.repo.update_document_metadata(
                                report_id,
                                {"sections_detail": sections_detail},
                            )
                            return generated
                    return summary
            return None

        def _claim_score(sentence: str) -> float:
            # Simple weighted scorer to keep numeric, sentence-like claims and drop headers/list numbers.
            has_digit = any(ch.isdigit() for ch in sentence)
            if not has_digit:
                return 0.0
            score = 0.0
            if has_digit:
                score += 2.0
            if UNIT_OR_YEAR.search(sentence):
                score += 2.0
            if VERB_HINTS.search(sentence):
                score += 1.0
            tokens = sentence.split()
            if len(tokens) >= 4:
                score += 0.5
            alpha_ratio = sum(1 for ch in sentence if ch.isalpha()) / max(1, len(sentence))
            if alpha_ratio < self.settings.claim_alpha_ratio_min:
                score -= 2.0
            numbers = NUMBER_PATTERN.findall(sentence)
            stripped = sentence.rstrip(". ").strip()
            if len(numbers) == 1 and stripped.endswith(numbers[0]) and not UNIT_OR_YEAR.search(sentence):
                score -= 1.5
            if ":" in sentence and not UNIT_OR_YEAR.search(sentence):
                score -= 1.0
            return score

        import os as _os
        _doc = self.repo.get_document(report_id) or {}
        _report_display_name = _os.path.basename(_doc.get("path") or "")

        sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
        claims: List[Claim] = []
        for sentence in sentences:
            if self._is_title_or_header(sentence, _report_display_name):
                logger.debug("Title/header filter dropped: %s", sentence[:80])
                continue
            if _heading_prefix_without_numbers(sentence):
                continue
            cleaned = _strip_bullet_prefix(sentence)
            if not cleaned:
                continue
            if not _has_alpha(cleaned):
                continue
            if len(cleaned.split()) < 3:
                continue
            # Skip sentences where the only numeric token is a leading heading-like number (e.g., "3.2 Climate...")
            num_matches = list(NUMBER_PATTERN.finditer(sentence))
            if num_matches:
                first = num_matches[0]
                heading_like = re.fullmatch(r"[0-9]{1,3}(?:\.[0-9]{1,3})*", first.group())
                no_other_numbers = len(num_matches) == 1
                remainder_has_units = bool(UNIT_OR_YEAR.search(sentence[first.end():]))
                if heading_like and first.start() == 0 and no_other_numbers and not remainder_has_units:
                    continue
            has_digits_cleaned = any(char.isdigit() for char in cleaned)
            if not has_digits_cleaned:
                # Secondary path: include non-numeric sentences that have strong verb signals.
                verb_matches = VERB_HINTS.findall(cleaned)
                non_numeric_score = 0.0
                if VERB_HINTS.search(cleaned):
                    non_numeric_score += 1.0
                if len(verb_matches) >= 2:
                    non_numeric_score += 0.5
                tokens_cleaned = cleaned.split()
                if len(tokens_cleaned) >= 4:
                    non_numeric_score += 0.5
                alpha_ratio_cleaned = sum(1 for ch in cleaned if ch.isalpha()) / max(1, len(cleaned))
                if alpha_ratio_cleaned >= 0.5:
                    non_numeric_score += 0.5
                if len(verb_matches) >= 2 and non_numeric_score >= self.settings.claim_non_numeric_score_min:
                    pass  # allow this sentence through
                else:
                    continue
            elif has_digits_cleaned:
                if _claim_score(cleaned) < self.settings.claim_score_min:
                    continue
            claim_id = str(uuid.uuid4())
            number_match = re.search(r"(-?\d[\d,\.]*)(?:\s?(percent|%|million|billion))?", cleaned, re.IGNORECASE)
            year_match = re.search(r"(20\d{2}|19\d{2})", cleaned)
            parent = _find_parent(cleaned)
            parent_summary = _find_parent_summary(parent.get("id") if parent else None)
            metadata = {
                "primary_value": number_match.group(1) if number_match else None,
                "units": (number_match.group(2) or "count") if number_match else None,
                "year": year_match.group(1) if year_match else None,
                "parent_id": parent.get("id") if parent else None,
                "parent_title": parent.get("title") if parent else None,
                "parent_page": parent.get("page") if parent else None,
                "parent_summary": parent_summary,
            }
            claim = Claim(
                claim_id=claim_id,
                text=cleaned,
                verdict="NOT_EVALUATED",
                confidence=0.4,
                confidence_band="LOW",
                processing_mode="unknown",
                explanation="Awaiting retrieval alignment.",
                metadata=metadata,
                evidence=[],
            )
            claims.append(claim)
        return claims

    def _index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        texts: List[str] = []
        metadata: List[Dict[str, Any]] = []
        for chunk in chunks:
            text = chunk.get("text") or ""
            if not text.strip():
                continue
            texts.append(text)
            chunk_meta = chunk.get("metadata") or {}
            metadata.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "snippet": chunk["text"][:400],
                    "parent_id": chunk_meta.get("parent_id"),
                    "parent_title": chunk_meta.get("parent_title"),
                    "parent_page": chunk_meta.get("parent_page"),
                    "chunk_type": chunk.get("chunk_type") or chunk_meta.get("chunk_type"),
                    "chart_id": chunk.get("chart_id") or chunk_meta.get("chart_id"),
                    "figure_label": chunk_meta.get("figure_label"),
                    "chart_type": chunk_meta.get("chart_type"),
                    "x_value": chunk.get("x_value") or chunk_meta.get("x_value"),
                    "y_value": chunk.get("y_value") or chunk_meta.get("y_value"),
                    "series_name": chunk.get("series_name") or chunk_meta.get("series_name"),
                }
            )
        if texts:
            self.vector_store.add_texts(texts, metadata)

    def _retrieve_evidence(self, claim: Claim, source_doc_ids: list[str] | None = None) -> List[Dict[str, Any]]:
        if source_doc_ids:
            per_source = self.settings.retrieval_per_source_cap
            hits: List[Dict[str, Any]] = []
            for doc_id in source_doc_ids:
                hits.extend(self.vector_store.similarity_search_by_doc(claim.text, doc_id, top_k=per_source))
            hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
        else:
            hits = self.vector_store.similarity_search(claim.text, top_k=self.settings.retrieval_top_k)
        logger.debug("Retrieved %d vector hits for claim %s", len(hits), claim.claim_id)
        if not hits:
            # Fallback when the vector index is empty or unavailable: use raw chunks from storage.
            fallback_chunks = self.repo.list_chunks()
            logger.debug("Falling back to %d stored chunks for claim %s", len(fallback_chunks), claim.claim_id)
            for chunk in fallback_chunks[:5]:
                meta = chunk.get("metadata") or {}
                hits.append(
                    {
                        "doc_id": chunk.get("doc_id"),
                        "chunk_id": chunk.get("chunk_id"),
                        "snippet": chunk.get("text"),
                        "score": self.settings.retrieval_support_threshold * 0.9,
                        "chunk_type": chunk.get("chunk_type") or meta.get("chunk_type"),
                        "chart_id": chunk.get("chart_id") or meta.get("chart_id"),
                        "figure_label": meta.get("figure_label"),
                        "chart_type": meta.get("chart_type"),
                        "x_value": chunk.get("x_value") or meta.get("x_value"),
                        "y_value": chunk.get("y_value") or meta.get("y_value"),
                        "series_name": chunk.get("series_name") or meta.get("series_name"),
                        "parent": {
                            "id": meta.get("parent_id"),
                            "title": meta.get("parent_title"),
                            "page": meta.get("parent_page"),
                            "text": None,
                            "summary": None,
                        },
                    }
                )

        # H3: Batch-fetch all unique (doc_id, parent_id) pairs before assigning to hits.
        unique_pairs = {
            (hit.get("doc_id"), hit.get("parent_id"))
            for hit in hits
            if hit.get("doc_id") and hit.get("parent_id")
        }
        parent_section_map: Dict[tuple, Dict[str, Any]] = {}
        for doc_id, parent_id in unique_pairs:
            section = self._get_parent_section(doc_id, parent_id)
            if section:
                parent_section_map[(doc_id, parent_id)] = section
        for hit in hits:
            parent_section = parent_section_map.get((hit.get("doc_id"), hit.get("parent_id")))
            if parent_section:
                hit["parent"] = {
                    "id": parent_section.get("id"),
                    "title": parent_section.get("title"),
                    "page": parent_section.get("page"),
                    "text": parent_section.get("text"),
                    "summary": parent_section.get("summary"),
                }
        if len(hits) > 1:
            hits = self.reranker.rerank(claim.text, hits)
        evidence_rows = []
        if not hits:
            claim.verdict = "NOT_FOUND"
            claim.explanation = "No supporting evidence retrieved."
            return evidence_rows

        best = hits[0]
        score = float(best.get("score") or 0.0)
        threshold = self.settings.retrieval_support_threshold
        snippet_text = (best.get("snippet") or best.get("text") or "").strip()
        override_threshold = self._should_override_threshold(score, threshold, claim.text, snippet_text)
        logger.debug(
            "Claim %s best score=%.4f threshold=%.4f override=%s",
            claim.claim_id,
            score,
            threshold,
            override_threshold,
        )
        classification = None
        if score >= threshold or override_threshold:
            multi_cap = self.settings.verdict_multi_evidence_cap
            classification = self.verdict_classifier.classify_multi(claim, hits[:multi_cap])
            label = (classification.get("label") or "SUPPORTED").upper()
            claim.processing_mode = classification.get("mode", "heuristic")
            if label == "CONTRADICTED":
                verdict = "CONTRADICTED"
            elif label == "SUPPORTED":
                verdict = "SUPPORTED"
            else:
                verdict = "NOT_FOUND"
        else:
            verdict = "NOT_FOUND"
            claim.processing_mode = "heuristic"

        # Plan 6: Temporal grounding — detect year mismatches between claim and evidence
        year_mismatch = False
        mismatch_detail = ""
        best_snippet = (best.get("snippet") or best.get("text") or "").strip()
        claim_years = self._extract_years(claim.text)
        evidence_years = self._extract_years(best_snippet)
        if claim_years and evidence_years:
            min_gap = min(abs(cy - ey) for cy in claim_years for ey in evidence_years)
            if min_gap > self.settings.temporal_mismatch_tolerance_years:
                year_mismatch = True
                mismatch_detail = f"[Year mismatch: claim references {sorted(claim_years)}, evidence references {sorted(evidence_years)}]"

        if year_mismatch and verdict == "SUPPORTED":
            verdict = "NOT_FOUND"
            if classification:
                classification["reason"] = mismatch_detail + " " + (classification.get("reason") or "")
                orig_conf = float(classification.get("confidence") or 0.5)
                classification["confidence"] = max(0.05, orig_conf - 0.2)
        elif year_mismatch and verdict == "CONTRADICTED":
            if classification:
                classification["reason"] = (classification.get("reason") or "") + " " + mismatch_detail

        claim.verdict = verdict
        confidence_candidates = [score]
        if classification and isinstance(classification.get("confidence"), (int, float)):
            confidence_candidates.append(float(classification["confidence"]))

        # Plan 4: Evidence count bonus for SUPPORTED label
        llm_confidence = float(classification["confidence"]) if classification and isinstance(classification.get("confidence"), (int, float)) else 0.0
        evidence_count_bonus = 0.0
        label_for_bonus = (classification.get("label") or "").upper() if classification else ""
        if label_for_bonus == "SUPPORTED":
            for h in hits[1:]:
                if float(h.get("score") or 0.0) > threshold:
                    evidence_count_bonus += 0.03
            evidence_count_bonus = min(evidence_count_bonus, 0.10)

        claim.confidence = float(min(0.99, max(0.05, max(score, llm_confidence) + evidence_count_bonus)))
        claim.confidence_band = self._band_from_confidence(claim.confidence)

        if verdict == "SUPPORTED":
            parent = best.get("parent") or {}
            page_hint = f" (page {parent.get('page')})" if parent.get("page") else ""
            reason = (classification or {}).get("reason") or f"Similarity {score:.2f}."
            if override_threshold and score < threshold:
                reason = f"{reason} (numeric overlap override; similarity {score:.2f} < {threshold:.2f})."
            claim.explanation = f"Supported by {best['doc_id']}{page_hint}: {reason}"
        elif verdict == "CONTRADICTED":
            parent = best.get("parent") or {}
            page_hint = f" (page {parent.get('page')})" if parent.get("page") else ""
            reason = (classification or {}).get("reason") or f"Similarity {score:.2f} but contradicting values detected."
            if override_threshold and score < threshold:
                reason = f"{reason} (numeric overlap override; similarity {score:.2f} < {threshold:.2f})."
            claim.explanation = f"Contradicted by {best['doc_id']}{page_hint}: {reason}"
        else:
            if score < threshold:
                claim.explanation = "Similarity below support threshold."
            else:
                reason = (classification or {}).get("reason") or "Evidence inconclusive."
                claim.explanation = f"No definitive evidence: {reason}"

        for hit in hits:
            evidence_rows.append(
                {
                    "evidence_id": str(uuid.uuid4()),
                    "claim_id": claim.claim_id,
                    "source_id": hit["doc_id"],
                    "chunk_id": hit.get("chunk_id"),
                    "verdict_label": verdict if hit is best else "ALTERNATIVE",
                    "metadata": {
                        "snippet": hit.get("snippet"),
                        "score": hit.get("score"),
                        "rerank_score": hit.get("rerank_score"),
                        "haystack_score": hit.get("haystack_score"),
                        "parent": hit.get("parent"),
                        "classification": classification if hit is best else None,
                        "chunk_type": hit.get("chunk_type"),
                        "chart_id": hit.get("chart_id"),
                        "figure_label": hit.get("figure_label"),
                        "chart_type": hit.get("chart_type"),
                        "x_value": hit.get("x_value"),
                        "y_value": hit.get("y_value"),
                        "series_name": hit.get("series_name"),
                    },
                }
            )
        return evidence_rows

    @staticmethod
    def _band_from_confidence(value: float) -> str:
        if value >= 0.75:
            return "HIGH"
        if value >= 0.45:
            return "MEDIUM"
        return "LOW"

    def _get_parent_section(self, doc_id: str | None, parent_id: str | None) -> Dict[str, Any] | None:
        if not doc_id or not parent_id:
            return None
        with self._section_cache_lock:
            # H4: Evict entire cache when it exceeds 500 entries to prevent unbounded memory growth.
            if len(self._section_cache) > 500:
                self._section_cache.clear()
            if doc_id not in self._section_cache:
                document = self.repo.get_document(doc_id) or {}
                metadata = document.get("metadata") or {}
                sections = metadata.get("sections_detail") or []
                self._section_cache[doc_id] = {section["id"]: section for section in sections}
            sections_map = self._section_cache.get(doc_id, {})
        section = sections_map.get(parent_id)
        if not section:
            return None
        if (
            self.settings.section_summary_mode.lower() == "lazy"
            and SECTION_INDEXER.enabled
            and not section.get("summary")
        ):
            try:
                generated = SECTION_INDEXER.summarize_sections([section]).get(parent_id)
            except Exception:
                generated = None
            if generated:
                updated = dict(section)
                updated["summary"] = generated
                with self._section_cache_lock:
                    sections_map[parent_id] = updated
                    self._section_cache[doc_id] = sections_map
                document = self.repo.get_document(doc_id) or {}
                metadata = document.get("metadata") or {}
                existing = metadata.get("sections_detail") or []
                replaced = []
                for entry in existing:
                    if entry.get("id") == parent_id:
                        replaced.append({**entry, "summary": generated})
                    else:
                        replaced.append(entry)
                self.repo.update_document_metadata(doc_id, {"sections_detail": replaced})
                return updated
        return section

    @staticmethod
    def _is_title_or_header(sentence: str, report_name: str | None = None) -> bool:
        import os as _os
        text = sentence.strip()
        # Rule 1: sentence contains the cleaned report name
        if report_name:
            cleaned_name = _os.path.splitext(report_name)[0].replace("_", " ").replace("-", " ").lower()
            if len(cleaned_name) > 5 and cleaned_name in text.lower():
                return True
        # Rule 2: high capitalisation ratio, no verbs, no stats pattern
        _STOP_WORDS = {"a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for", "with", "by", "from", "into", "about", "as", "is", "are"}
        words = text.split()
        content_words = [w for w in words if w.lower() not in _STOP_WORDS]
        if 3 <= len(words) <= 20 and content_words:
            cap_ratio = sum(1 for w in content_words if w and w[0].isupper()) / len(content_words)
            _STAT_PATTERN = re.compile(r"\d+\s*(%|percent|million|billion|thousand|kg|mt|usd|\$)", re.IGNORECASE)
            if cap_ratio >= 0.60 and not VERB_HINTS.search(text) and not _STAT_PATTERN.search(text):
                return True
        # Rule 3: colon with left side being 1-6 title-cased words and no verbs
        if ":" in text:
            left = text.split(":")[0].strip()
            left_words = left.split()
            if 1 <= len(left_words) <= 6 and all(w[0].isupper() for w in left_words if w) and not VERB_HINTS.search(text):
                return True
        # Rule 4: ends with a bare number 1-20 without stat context
        _TRAILING_NUM = re.compile(r"\s+(\d{1,2})[.)]*$")
        _STAT_CONTEXT = re.compile(r"percent|million|grew|fell|rose|increased|decreased|19\d{2}|20\d{2}", re.IGNORECASE)
        m = _TRAILING_NUM.search(text)
        if m:
            num = int(m.group(1))
            if 1 <= num <= 20:
                preceding = text[:m.start()]
                if not _STAT_CONTEXT.search(preceding):
                    return True
        return False

    @staticmethod
    def _extract_years(text: str | None) -> set[int]:
        if not text:
            return set()
        return {int(m) for m in _YEAR_RE.findall(text)}

    @staticmethod
    def _numbers_in_text(text: str | None) -> set[str]:
        """Return the set of numeric strings found in *text* via NUMBER_PATTERN.

        A ``set`` is returned intentionally so callers can use set intersection
        (``&``) for O(1) overlap checks (see ``_should_override_threshold``).
        """
        if not text:
            return set()
        return set(NUMBER_PATTERN.findall(text))

    def _should_override_threshold(self, score: float, threshold: float, claim_text: str, snippet: str) -> bool:
        if score >= threshold:
            return False
        claim_numbers = self._numbers_in_text(claim_text)
        snippet_numbers = self._numbers_in_text(snippet)
        if not claim_numbers or not snippet_numbers:
            return False
        return bool(claim_numbers & snippet_numbers)

    def _persist_claim_index(self, report_id: str, claims: List[Claim]) -> None:
        store = VectorStore(report_id, base_dir=self.settings.claim_vector_path)
        if not claims:
            store.overwrite([], [])
            return
        texts = [
            f"Claim: {claim.text}\nVerdict: {claim.verdict}\nExplanation: {claim.explanation}"
            for claim in claims
        ]
        metadata = [
            {
                "claim_id": claim.claim_id,
                "verdict": claim.verdict,
                "confidence": claim.confidence,
            }
            for claim in claims
        ]
        store.overwrite(texts, metadata)

    def _persist_source_index(self, report_id: str, evidence_rows: List[Dict[str, Any]]) -> None:
        snippets: List[str] = []
        metadata: List[Dict[str, Any]] = []
        for row in evidence_rows:
            snippet = (row.get("metadata") or {}).get("snippet")
            if not snippet:
                continue
            snippets.append(snippet)
            metadata.append(
                {
                    "claim_id": row["claim_id"],
                    "source_id": row.get("source_id"),
                    "chunk_id": row.get("chunk_id"),
                    "verdict_label": row.get("verdict_label"),
                    "score": (row.get("metadata") or {}).get("score"),
                    "snippet": snippet[:600],
                }
            )
        store = VectorStore(report_id, base_dir=self.settings.source_vector_path)
        if not snippets:
            store.overwrite([], [])
            return
        store.overwrite(snippets, metadata)

    def _persist_ingestion_payload(self, payload: Dict[str, Any], upload: dict, doc_type: str) -> None:
        """
        Ensure documents/chunks are stored in SQLite even when ingestion is stubbed (e.g., in tests).
        """
        doc_id = upload.get("upload_id")
        if not doc_id:
            return
        existing = self.repo.get_document(doc_id)
        body_text = payload.get("body_text") or ""
        if not existing:
            self.repo.upsert_document(
                doc_id=doc_id,
                doc_type=doc_type,
                path=str(upload.get("path", "")),
                metadata={},
                body_text=body_text,
                created_at=_now_iso(),
            )
        chunks = payload.get("chunks") or []
        if chunks:
            chunk_records = []
            for chunk in chunks:
                chunk_records.append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "doc_id": chunk.get("doc_id") or doc_id,
                        "text": chunk.get("text") or "",
                        "page_start": None,
                        "page_end": None,
                        "metadata": chunk.get("metadata") or {},
                    }
                )
            self.repo.insert_chunks(chunk_records)
