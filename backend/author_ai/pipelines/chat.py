"""
Claim-first chat assistant orchestrator.
"""

from __future__ import annotations

from typing import Dict, Any, List, DefaultDict, Optional, Tuple
from collections import defaultdict
import re

try:
    import anthropic as _anthropic
except ImportError:  # pragma: no cover
    _anthropic = None  # type: ignore

from ..storage.database import Repository
from ..config import get_settings
from ..services.logger import setup_logger
from ..models import _now_iso
from ..services.embedding import embed_texts
from ..services.vector_store import VectorStore

GENERAL_INTENTS = {"hey", "hello", "hi", "thanks", "thank you", "who are you", "help", "good morning", "good afternoon"}
REPORT_KEYWORDS = {
    "claim",
    "claims",
    "credibility",
    "validity",
    "accuracy",
    "source",
    "author",
    "publication",
    "report",
    "evidence",
    "support",
    "contradict",
    "table",
}

CHAT_MODES = ("evidence", "guidance", "creative")
GUIDANCE_KEYWORDS = {"improve", "improvement", "fix", "revise", "revamp", "better", "enhance"}
CREATIVE_KEYWORDS = {"brainstorm", "idea", "ideas", "scenario", "plan", "future", "summary", "recommend"}
MODE_HELP_KEYWORDS = {"mode", "modes", "model", "models", "setting", "settings", "assistant mode"}
MODE_EXPLANATIONS = {
    "evidence": "Evidence — strictly summarizes verified claims and metrics with no speculation.",
    "guidance": "Guidance — still grounded in claims but explains verdict gaps and suggests next steps to improve the report.",
    "creative": "Creative — brainstorms advisory ideas based on the verified context, clearly labelling suggestions as advisory.",
}


logger = setup_logger(__name__)


class ChatService:
    def __init__(self):
        self.settings = get_settings()
        self.repo = Repository()
        self.default_mode = CHAT_MODES[0]
        if _anthropic and self.settings.anthropic_api_key:
            self.client = _anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        else:
            self.client = None
            logger.info("ChatService running in heuristic mode (set ANTHROPIC_API_KEY for LLM answers).")

    def respond(
        self,
        question: str,
        report_id: str,
        session_id: str | None = None,
        mode: Optional[str] = None,
        mode_locked: bool = False,
    ) -> Dict[str, Any]:
        cleaned_question = (question or "").strip()
        requested_mode = (mode or "").lower()
        active_mode = requested_mode if requested_mode in CHAT_MODES else self.default_mode
        inferred_mode = self._infer_mode(cleaned_question)
        suggested_mode = None
        if not mode_locked and inferred_mode and inferred_mode != active_mode:
            suggested_mode = inferred_mode

        mode_help_requested = self._is_mode_help_question(cleaned_question)

        claims = self.repo.list_claims_by_report(report_id)
        claim_map = {claim["claim_id"]: claim for claim in claims}
        claim_index_map = {idx + 1: claim for idx, claim in enumerate(claims)}
        claim_number_map = {claim["claim_id"]: idx + 1 for idx, claim in enumerate(claims)}

        raw_history = self.repo.get_chat_history(report_id, limit=self.settings.chat_history_length)
        history_context = self._build_history_context(self._trim_history(raw_history, cleaned_question))
        intent = self._detect_intent(cleaned_question)

        forced_claims: Optional[List[Dict[str, Any]]] = None
        claim_request = self._extract_claim_request(cleaned_question)
        if claim_request:
            resolved: List[Dict[str, Any]] = []
            missing: List[str] = []
            if claim_request.get("all"):
                resolved = claims
            else:
                ids = claim_request.get("ids") or []
                nums = claim_request.get("numbers") or []
                for cid in ids:
                    claim = claim_map.get(cid)
                    if claim:
                        resolved.append(claim)
                    else:
                        missing.append(cid)
                for num in nums:
                    claim = claim_index_map.get(num)
                    if claim:
                        resolved.append(claim)
                    else:
                        missing.append(str(num))
            if not resolved:
                answer = "No matching claims were found for your request."
                return self._finalize_response(
                    answer,
                    session_id,
                    report_id,
                    question,
                    [],
                    [],
                    mode=self.default_mode,
                    suggested_mode=None,
                )
            forced_claims = resolved

        if intent != "report":
            answer = self._small_talk_answer(cleaned_question or None, history_context)
            return self._finalize_response(
                answer,
                session_id,
                report_id,
                question,
                [],
                [],
                active_mode,
                suggested_mode,
            )

        report_doc = self.repo.get_document(report_id)
        validity_record = self.repo.get_validity(report_id)
        credibility_breakdown = self.repo.source_usage(report_id)
        latest_job = self.repo.get_latest_job_for_report(report_id)
        credibility_overall = (
            latest_job.get("result_json", {}).get("credibility", {}).get("overall")
            if latest_job
            else None
        )
        recommended_sources = (
            (latest_job.get("result_json") or {}).get("recommended_sources") if latest_job else None
        )

        question_vector = self._embed_question(cleaned_question)
        if forced_claims is not None:
            claim_context = [{"claim": claim, "score": None} for claim in forced_claims]
        else:
            claim_context = self._select_claim_context(report_id, claim_map, question_vector, active_mode)
        claim_ids = [entry["claim"]["claim_id"] for entry in claim_context]
        evidence_map = self._group_evidence_by_claim(claim_ids)

        source_context = self._select_source_context(
            report_id,
            question_vector,
            active_mode,
            claim_ids,
            evidence_map,
        )
        source_ids = {entry.get("source_id") for entry in source_context if entry.get("source_id")}
        source_docs = self.repo.list_documents([sid for sid in source_ids if sid]) if source_ids else {}

        metrics_context = self._build_metric_context(
            claims=claims,
            validity=validity_record,
            credibility_usage=credibility_breakdown,
            source_docs=source_docs,
            credibility_overall=credibility_overall,
            mode=active_mode,
            recommended_sources=recommended_sources,
        )
        core_context = self._build_core_context(report_doc)
        mode_help_context = self._build_mode_help_context(active_mode) if mode_help_requested else None

        answer = (
            self._llm_answer(
                question=cleaned_question,
                claim_context=claim_context,
                evidence_map=evidence_map,
                source_context=source_context,
                source_docs=source_docs,
                report_doc=report_doc,
                metrics_context=metrics_context,
                history_context=history_context,
                core_context=core_context,
                mode_help_context=mode_help_context,
                mode=active_mode,
                claim_number_map=claim_number_map,
            )
            if self.client
            else self._compose_answer(
                claim_context,
                source_context,
                metrics_context,
                history_context,
                core_context,
                mode_help_context,
                active_mode,
                claim_number_map,
            )
        )

        return self._finalize_response(
            answer,
            session_id,
            report_id,
            question,
            [entry["claim"] for entry in claim_context],
            [entry for entry in source_context if entry.get("source_id")],
            active_mode,
            suggested_mode,
        )

    def _llm_answer(
        self,
        question: str,
        claim_context: List[Dict[str, Any]],
        evidence_map: DefaultDict[str, List[dict]],
        source_context: List[Dict[str, Any]],
        source_docs: Dict[str, Dict[str, Any]],
        report_doc: Dict[str, Any] | None,
        metrics_context: Optional[str],
        history_context: Optional[str],
        core_context: Optional[str],
        mode_help_context: Optional[str],
        mode: str,
        claim_number_map: Optional[Dict[str, int]],
    ) -> str:
        claim_blocks = self._render_claim_findings(claim_context, evidence_map, source_docs, claim_number_map)
        support_blocks = self._render_supporting_context(source_context, source_docs)
        if not any([claim_blocks, support_blocks, metrics_context, core_context]):
            return self._format_response(
                "I do not yet have enough verified evidence to answer. Please run the pipeline or provide more sources."
            )

        context_sections = []
        report_summary = (
            (report_doc or {}).get("metadata", {}).get("summary")
            if report_doc and report_doc.get("metadata")
            else None
        )
        if report_summary:
            context_sections.append(f"Report Summary:\n{report_summary}")
        if claim_blocks:
            context_sections.append("Claim Findings:\n" + "\n\n".join(claim_blocks))
        if support_blocks:
            context_sections.append("Supporting Context:\n" + "\n\n".join(support_blocks))
        if core_context:
            context_sections.append(core_context)
        if metrics_context:
            context_sections.append(metrics_context)
        if history_context:
            context_sections.append(history_context)
        if mode_help_context:
            context_sections.append(mode_help_context)
        context_sections.append(f"Assistant mode: {mode.upper()}")
        context_sections.append("User Question:\n" + question)
        prompt = "\n\n".join(context_sections)

        system_prompt = self._mode_system_prompt(mode)
        try:
            response = self.client.messages.create(  # type: ignore[union-attr]
                model=self.settings.llm_chat_model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            text_blocks = [b.text for b in response.content if b.type == "text"]
            choice = text_blocks[0] if text_blocks else None
            if choice:
                return self._format_response(choice)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChatService LLM call failed; falling back to heuristic answer: %s", exc)
        return self._compose_answer(
            claim_context,
            source_context,
            metrics_context,
            history_context,
            core_context,
            mode_help_context,
            mode,
            claim_number_map,
        )

    def _compose_answer(
        self,
        claim_context: List[Dict[str, Any]],
        source_context: List[Dict[str, Any]],
        metrics_context: Optional[str],
        history_context: Optional[str],
        core_context: Optional[str],
        mode_help_context: Optional[str],
        mode: str,
        claim_number_map: Optional[Dict[str, int]] = None,
    ) -> str:
        lines: List[str] = []
        if claim_context:
            lines.append("Claims considered:")
            for entry in claim_context:
                claim = entry["claim"]
                number = None
                if claim_number_map:
                    number = claim_number_map.get(claim.get("claim_id"))
                number_text = f"(#{number}) " if number else ""
                verdict = claim.get("verdict", "UNKNOWN")
                explanation = claim.get("explanation") or claim.get("text")
                lines.append(f"- {number_text}{verdict}: {explanation}")
        if source_context:
            lines.append("\nSupporting snippets:")
            for ctx in source_context:
                snippet = ctx.get("snippet")
                if snippet:
                    label = ctx.get("source_id") or "source"
                    lines.append(f"- {label}: {snippet}")
        if metrics_context:
            lines.append("\nMetrics:")
            lines.append(metrics_context)
        if core_context:
            lines.append("\nReport context:")
            lines.append(core_context)
        if history_context:
            lines.append("\nConversation context:")
            lines.append(history_context)
        if mode_help_context:
            lines.append("")
            lines.append(mode_help_context)
        if not lines:
            return self._format_response(
                "I do not yet have verified data to answer this. Please ensure the report has been processed."
            )
        lines.append(f"\nMode: {mode.title()}. Response generated without the LLM fallback.")
        return self._format_response("\n".join(lines))

    def _small_talk_answer(self, question: Optional[str], history_context: Optional[str]) -> str:
        if not self.client:
            return "Hello! I’m here to help with your report’s accuracy, credibility, and validity. Ask me about metrics, claims, or sources when you’re ready."
        system_prompt = (
            "You are a friendly Author AI assistant. Engage in light small talk, "
            "but keep the conversation professional and guide the user back to report quality topics when possible. "
            "Avoid speculating about unrelated world knowledge."
        )
        user_content = "\n\n".join(filter(None, [history_context, question or "Just say hello to the user."]))
        try:
            response = self.client.messages.create(  # type: ignore[union-attr]
                model=self.settings.llm_chat_model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            text_blocks = [b.text for b in response.content if b.type == "text"]
            choice = text_blocks[0] if text_blocks else None
            if choice:
                return self._format_response(choice)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Small talk LLM call failed: %s", exc)
        return "Hello! I’m ready to help whenever you have a question about your report."

    def _detect_intent(self, question: str) -> str:
        lowered = question.lower().strip()
        for keyword in REPORT_KEYWORDS:
            if keyword in lowered:
                return "report"
        for keyword in GENERAL_INTENTS:
            if keyword in lowered:
                return "general"
        return "general" if len(lowered.split()) <= 2 else "report"

    def _build_metric_context(
        self,
        claims: List[Dict[str, Any]],
        validity: Optional[dict],
        credibility_usage: List[dict],
        source_docs: Dict[str, Dict[str, Any]],
        credibility_overall: Optional[float],
        mode: str,
        recommended_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not claims and not validity and not credibility_usage:
            return ""

        supported = sum(1 for claim in claims if claim.get("verdict") == "SUPPORTED")
        contradicted = sum(1 for claim in claims if claim.get("verdict") == "CONTRADICTED")
        total_claims = len(claims)
        metrics_lines = [
            "Metric Diagnostics:",
            f"- Accuracy: {supported}/{total_claims or 1} claims supported; {contradicted} contradicted.",
        ]

        improvement_notes: List[str] = []

        if validity:
            diagnostics = validity.get("diagnostics") or {}
            coverage = diagnostics.get("missing_topics")
            methodology = diagnostics.get("methodology_elements")
            metrics_lines.append(
                f"- Validity ({validity.get('overall', 0)}): coverage {validity.get('coverage')}%, "
                f"consistency {validity.get('consistency')}%, methodology {validity.get('methodology')}%, "
                f"context {validity.get('context')}%, recency {validity.get('recency')}%."
            )
            if coverage:
                metrics_lines.append(f"  Missing topics: {', '.join(coverage)}.")
                improvement_notes.append(
                    f"Add evidence or sections covering: {', '.join(coverage)}."
                )
            if methodology:
                missed = [k for k, v in methodology.items() if not v]
                if missed:
                    metrics_lines.append(f"  Methodology gaps: {', '.join(missed)}.")
                    improvement_notes.append(
                        f"Document methodology details for: {', '.join(missed)}."
                    )

        if credibility_overall is not None:
            metrics_lines.append(f"- Credibility overall: {credibility_overall:.2f}")
            if credibility_overall < 60:
                improvement_notes.append("Introduce higher-credibility sources or refresh outdated data.")

        if credibility_usage:
            lines = ["- Credibility sources:"]
            for entry in credibility_usage:
                cred = self.repo.get_credibility(entry["source_id"])
                source_meta = source_docs.get(entry["source_id"]) or {}
                upload = self.repo.get_upload(entry["source_id"])
                label = (
                    (source_meta.get("metadata") or {}).get("title")
                    or (upload or {}).get("file_name")
                    or source_meta.get("doc_id")
                    or entry["source_id"]
                )
                score = cred.get("score") if cred else None
                components_data = cred.get("components") if cred else None
                if components_data:
                    components = ", ".join(f"{key}:{value}" for key, value in components_data.items())
                else:
                    components = "components unavailable"
                metadata = source_meta.get("metadata") or {}
                authors = metadata.get("authors")
                pub_date = metadata.get("publication_date")
                lines.append(
                    f"  • {label} — score {score if score is not None else 'N/A'} ({components}); "
                    f"used in {entry['usage_count']} claims."
                )
                if authors or pub_date:
                    author_text = ", ".join(authors) if isinstance(authors, list) else authors
                    lines.append(
                        f"    metadata: author(s) {author_text or 'unknown'}, published {pub_date or 'unknown'}."
                    )
            metrics_lines.extend(lines)

        if recommended_sources:
            metrics_lines.append("- Recommended external sources:")
            for entry in recommended_sources[:5]:
                title = entry.get("title") or entry.get("name") or "Source"
                note = entry.get("summary") or entry.get("host_venue")
                metadata = entry.get("publication_year")
                citation = entry.get("cited_by_count")
                detail_parts = []
                if metadata:
                    detail_parts.append(str(metadata))
                if isinstance(citation, int):
                    detail_parts.append(f"{citation} citations")
                detail_text = f" ({', '.join(detail_parts)})" if detail_parts else ""
                context_note = f" — {note}" if note else ""
                metrics_lines.append(f"  • {title}{detail_text}{context_note}")

        if mode in {"guidance", "creative"} and improvement_notes:
            metrics_lines.append("- Suggested improvements:")
            for note in improvement_notes:
                metrics_lines.append(f"  • {note}")

        return "\n".join(metrics_lines)

    def _build_history_context(self, history: List[dict]) -> str:
        if not history:
            return ""
        lines = ["Recent conversation:"]
        for turn in history:
            prefix = "User" if turn["role"].lower() == "user" else "Assistant"
            lines.append(f"{prefix}: {turn['message']}")
        return "\n".join(lines)

    def _trim_history(self, history: List[dict], question: str, max_tokens: int = 1200) -> List[dict]:
        if not history:
            return []
        def estimate_tokens(text: str) -> int:
            return max(1, len(text) // 4)

        budget = max_tokens - estimate_tokens(question)
        trimmed: List[dict] = []
        accumulated = 0
        for turn in reversed(history):
            cost = estimate_tokens(turn.get("message", "")) + 20
            if accumulated + cost > budget and trimmed:
                break
            trimmed.append(turn)
            accumulated += cost
        trimmed.reverse()
        return trimmed

    def _build_core_context(self, report_doc: Optional[dict]) -> Optional[str]:
        if not report_doc:
            return None
        metadata = report_doc.get("metadata") or {}
        parts = []
        summary = metadata.get("summary")
        if summary:
            parts.append(f"Summary: {summary}")
        accuracy_summary = metadata.get("accuracy_summary")
        if accuracy_summary:
            parts.append(f"Accuracy: {accuracy_summary}")
        validity_summary = metadata.get("validity_summary")
        if validity_summary:
            parts.append(f"Validity: {validity_summary}")
        credibility_summary = metadata.get("credibility_summary")
        if credibility_summary:
            parts.append(f"Credibility: {credibility_summary}")
        return "\n".join(parts) if parts else None

    def _format_response(self, text: str) -> str:
        formatted = text.strip()
        formatted = formatted.replace("**", "")
        formatted = formatted.replace("* ", "• ")
        return formatted

    def _embed_question(self, question: str) -> Optional[List[float]]:
        if not question:
            return None
        vectors = embed_texts([question])
        return vectors[0] if vectors else None

    def _select_claim_context(
        self,
        report_id: str,
        claim_map: Dict[str, Dict[str, Any]],
        question_vector: Optional[List[float]],
        mode: str,
    ) -> List[Dict[str, Any]]:
        if not claim_map:
            return []
        # Use all claims; order by semantic score when available, otherwise by priority.
        if question_vector is not None:
            store = VectorStore(report_id, base_dir=self.settings.claim_vector_path)
            hits = store.search(question_vector, top_k=max(len(claim_map), self.settings.claim_context_limit * 2))
            ranked: List[Dict[str, Any]] = []
            for hit in hits:
                claim = claim_map.get(hit.get("claim_id"))
                if not claim:
                    continue
                ranked.append({"claim": claim, "score": hit.get("score")})
            if ranked:
                ranked.sort(key=lambda entry: entry.get("score") or 0, reverse=True)
                return ranked
        fallback_claims = sorted(
            claim_map.values(),
            key=self._claim_priority_sort,
        )
        return [{"claim": claim, "score": None} for claim in fallback_claims]

    def _select_source_context(
        self,
        report_id: str,
        question_vector: Optional[List[float]],
        mode: str,
        fallback_claim_ids: List[str],
        evidence_map: DefaultDict[str, List[dict]],
    ) -> List[Dict[str, Any]]:
        contexts: List[Dict[str, Any]] = []
        if question_vector is not None:
            store = VectorStore(report_id, base_dir=self.settings.source_vector_path)
            hits = store.search(question_vector, top_k=self.settings.source_context_limit * 3)
            threshold = self._threshold_for_mode(mode, self.settings.claim_relevance_min * 0.8)
            seen_sources: set[str] = set()
            for hit in hits:
                score = hit.get("score", 0)
                if score < threshold:
                    continue
                source_id = hit.get("source_id")
                if source_id in seen_sources:
                    continue
                seen_sources.add(source_id)
                contexts.append(hit)
                if len(contexts) >= self.settings.source_context_limit:
                    break
            if contexts:
                return contexts
        if not fallback_claim_ids:
            return []
        fallback_contexts: List[Dict[str, Any]] = []
        for claim_id in fallback_claim_ids:
            for evidence in evidence_map.get(claim_id, [])[:1]:
                snippet = (evidence.get("metadata") or {}).get("snippet")
                if not snippet:
                    continue
                fallback_contexts.append(
                    {
                        "source_id": evidence.get("source_id"),
                        "claim_id": claim_id,
                        "snippet": snippet,
                        "score": evidence.get("metadata", {}).get("score"),
                    }
                )
                if len(fallback_contexts) >= self.settings.source_context_limit:
                    return fallback_contexts
        return fallback_contexts

    def _group_evidence_by_claim(self, claim_ids: List[str]) -> DefaultDict[str, List[dict]]:
        evidence_map: DefaultDict[str, List[dict]] = defaultdict(list)
        if not claim_ids:
            return evidence_map
        rows = self.repo.list_evidence_for_claims(claim_ids)
        for row in rows:
            evidence_map[row["claim_id"]].append(row)
        return evidence_map

    @staticmethod
    def _claim_priority_sort(claim: Dict[str, Any]) -> Tuple[int, float]:
        verdict_rank = {"SUPPORTED": 0, "CONTRADICTED": 1, "NOT_FOUND": 2}.get(
            (claim.get("verdict") or "").upper(),
            3,
        )
        confidence = float(-(claim.get("confidence") or 0.0))
        return (verdict_rank, confidence)

    def _threshold_for_mode(self, mode: str, base: float) -> float:
        if mode == "guidance":
            return max(0.05, base * 0.8)
        if mode == "creative":
            return max(0.05, base * 0.7)
        return base

    def _infer_mode(self, question: str) -> Optional[str]:
        lowered = (question or "").lower()
        if any(keyword in lowered for keyword in GUIDANCE_KEYWORDS):
            return "guidance"
        if any(keyword in lowered for keyword in CREATIVE_KEYWORDS):
            return "creative"
        return None

    def _is_mode_help_question(self, question: str) -> bool:
        lowered = (question or "").lower()
        if not lowered:
            return False
        if any(keyword in lowered for keyword in MODE_HELP_KEYWORDS):
            return True
        return "mode" in lowered and "what" in lowered

    def _build_mode_help_context(self, active_mode: str) -> str:
        descriptions = [MODE_EXPLANATIONS[mode] for mode in CHAT_MODES]
        autoprefix = (
            "Auto — lets the assistant pick whichever behavior best fits your question."
        )
        entries = descriptions + [autoprefix, f"Current preference: {active_mode.title()} mode."]
        return "Assistant Modes:\n- " + "\n- ".join(entries)

    def _mode_system_prompt(self, mode: str) -> str:
        if mode == "guidance":
            return (
                "You are an Author AI report coach. Use verified Claim Findings as the source of truth, "
                "and translate diagnostics into actionable recommendations. Highlight gaps, suggest next "
                "steps, and ask for missing data when needed."
            )
        if mode == "creative":
            return (
                "You are an Author AI brainstorming assistant. Ground statements in the provided claims and "
                "context, but you may offer clearly labeled advisory ideas or next steps. Never fabricate "
                "data; mark general recommendations as 'Advisory'."
            )
        return (
            "You are an Author AI verification assistant. Answer strictly from Claim Findings, supporting "
            "context, and diagnostics. If evidence is missing, say so and do not speculate."
        )

    def _render_claim_findings(
        self,
        claim_context: List[Dict[str, Any]],
        evidence_map: DefaultDict[str, List[dict]],
        source_docs: Dict[str, Dict[str, Any]],
        claim_number_map: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        blocks: List[str] = []
        for idx, entry in enumerate(claim_context, start=1):
            claim = entry["claim"]
            ordinal = claim_number_map.get(claim.get("claim_id")) if claim_number_map else None
            label = f"[Claim {ordinal}]" if ordinal else f"[Claim {idx}]"
            block_lines = [
                f"{label} {claim.get('text')}",
                f"ID: {claim.get('claim_id')}",
                f"Verdict: {claim.get('verdict')} (confidence {claim.get('confidence')})",
                f"Explanation: {claim.get('explanation')}",
            ]
            for evidence in evidence_map.get(claim["claim_id"], []):
                snippet = (evidence.get("metadata") or {}).get("snippet")
                if not snippet:
                    continue
                source_id = evidence.get("source_id")
                source_meta = source_docs.get(source_id, {})
                label = (
                    (source_meta.get("metadata") or {}).get("title")
                    or evidence.get("file_name")
                    or source_id
                    or "source"
                )
                block_lines.append(f"Evidence ({label}): {snippet}")
            blocks.append("\n".join(block_lines))
        return blocks

    def _render_supporting_context(
        self,
        source_context: List[Dict[str, Any]],
        source_docs: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        blocks: List[str] = []
        for entry in source_context:
            snippet = entry.get("snippet")
            if not snippet:
                continue
            source_id = entry.get("source_id")
            doc = source_docs.get(source_id or "") or {}
            label = (
                (doc.get("metadata") or {}).get("title")
                or doc.get("doc_id")
                or source_id
                or "source"
            )
            score = entry.get("score")
            meta = f" (score {score:.2f})" if isinstance(score, float) else ""
            blocks.append(f"{label}{meta}: {snippet}")
        return blocks

    def _finalize_response(
        self,
        answer: str,
        session_id: Optional[str],
        report_id: str,
        question: str,
        claims_used: List[dict],
        sources_used: List[dict],
        mode: str,
        suggested_mode: Optional[str],
    ) -> Dict[str, Any]:
        timestamp = _now_iso()
        session = session_id or "anonymous"
        self.repo.record_chat_turn(
            {
                "session_id": session,
                "report_id": report_id,
                "role": "user",
                "message": question,
                "timestamp": timestamp,
                "context_ids": {"mode": mode},
            }
        )
        self.repo.record_chat_turn(
            {
                "session_id": session,
                "report_id": report_id,
                "role": "assistant",
                "message": answer,
                "timestamp": _now_iso(),
                "context_ids": {
                    "mode": mode,
                    "claims": [claim.get("claim_id") for claim in claims_used],
                    "sources": [source.get("source_id") for source in sources_used],
                },
            }
        )
        return {
            "answer": answer,
            "claims_used": claims_used,
            "sources_used": sources_used,
            "mode": mode,
            "suggested_mode": suggested_mode,
        }

    def _extract_claim_request(self, question: str) -> Dict[str, Any] | None:
        lower = (question or "").lower()
        if not lower:
            return None
        if "all claims" in lower or "list all claims" in lower:
            return {"all": True}
        # capture patterns like "claim 3", "claims 3,4 and 5", "claim id <uuid>"
        ids: List[str] = []
        numbers: List[int] = []

        # claim id explicit
        for match in re.findall(r"claim[_\s-]?id[:\s]+([0-9a-f\-]{6,})", lower):
            ids.append(match.strip())

        # numeric references
        for block in re.findall(r"claims?\s+([0-9,\sand]+)", lower):
            tokens = re.split(r"[,\sand]+", block)
            for tok in tokens:
                tok = tok.strip()
                if tok.isdigit():
                    numbers.append(int(tok))

        for match in re.findall(r"claim\s+([0-9]+)", lower):
            numbers.append(int(match))

        if ids or numbers:
            return {"ids": ids, "numbers": numbers}
        return None

    def _deterministic_claim_response(
        self,
        request: Dict[str, Any],
        *,
        claims: List[Dict[str, Any]],
        claim_map: Dict[str, Dict[str, Any]],
        claim_index_map: Dict[int, Dict[str, Any]],
        session_id: str,
        report_id: str,
        question: str,
    ) -> Dict[str, Any]:
        resolved: List[Dict[str, Any]] = []
        missing: List[str] = []

        if request.get("all"):
            resolved = claims
        else:
            ids = request.get("ids") or []
            nums = request.get("numbers") or []
            for cid in ids:
                claim = claim_map.get(cid)
                if claim:
                    resolved.append(claim)
                else:
                    missing.append(cid)
            for num in nums:
                claim = claim_index_map.get(num)
                if claim:
                    resolved.append(claim)
                else:
                    missing.append(str(num))

        if not resolved:
            answer = "No matching claims were found for your request."
            return self._finalize_response(
                answer,
                session_id,
                report_id,
                question,
                [],
                [],
                mode="evidence",
                suggested_mode=None,
            )

        lines: List[str] = []
        for claim in resolved:
            lines.append(
                f"Claim {claim.get('claim_id')}: {claim.get('text')}\n"
                f"Verdict: {claim.get('verdict')} (confidence {claim.get('confidence')})\n"
                f"Explanation: {claim.get('explanation')}"
            )
        if missing:
            lines.append(f"Not found: {', '.join(missing)}")

        answer = "\n\n".join(lines)
        return self._finalize_response(
            answer,
            session_id,
            report_id,
            question,
            resolved,
            [],
            mode="evidence",
            suggested_mode=None,
        )
