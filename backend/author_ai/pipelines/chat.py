"""
Claim-first chat assistant orchestrator.
"""

from __future__ import annotations

from typing import Dict, Any, List, DefaultDict, Optional
from collections import defaultdict
import json

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

from ..storage.database import Repository
from ..config import get_settings
from ..services.logger import setup_logger
from ..models import _now_iso

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


logger = setup_logger(__name__)


class ChatService:
    def __init__(self):
        self.settings = get_settings()
        self.repo = Repository()
        if OpenAI and self.settings.openai_api_key:
            self.client = OpenAI(api_key=self.settings.openai_api_key)
        else:
            self.client = None
            logger.info("ChatService running in heuristic mode (set OPENAI_API_KEY for LLM answers).")

    def respond(self, question: str, report_id: str, session_id: str | None = None) -> Dict[str, Any]:
        intent = self._detect_intent(question)
        claims = self.repo.list_claims_by_report(report_id)
        claim_context = claims[: self.settings.claim_context_limit] if intent == "report" else []

        claim_ids = [claim["claim_id"] for claim in claim_context]
        evidences = self.repo.list_evidence_for_claims(claim_ids)
        evidence_map: DefaultDict[str, List[dict]] = defaultdict(list)
        source_ids: set[str] = set()
        for row in evidences:
            evidence_map[row["claim_id"]].append(row)
            if row.get("source_id"):
                source_ids.add(row["source_id"])

        source_docs = self.repo.list_documents(list(source_ids))
        report_doc = self.repo.get_document(report_id)
        validity_record = self.repo.get_validity(report_id)
        credibility_breakdown = self.repo.source_usage(report_id)
        latest_job = self.repo.get_latest_job_for_report(report_id)
        credibility_overall = (
            latest_job.get("result_json", {}).get("credibility", {}).get("overall")
            if latest_job
            else None
        )

        metrics_context = (
            self._build_metric_context(
                claims=claims,
                validity=validity_record,
                credibility_usage=credibility_breakdown,
                source_docs=source_docs,
                credibility_overall=credibility_overall,
            )
            if intent == "report"
            else None
        )
        history = self.repo.get_chat_history(report_id, limit=self.settings.chat_history_length)
        history_context = self._build_history_context(history)
        core_context = self._build_core_context(report_doc)

        answer = (
            self._llm_answer(
                question,
                claim_context,
                evidence_map,
                source_docs,
                report_doc,
                metrics_context,
                history_context,
                core_context,
                intent,
            )
            if self.client
            else self._compose_answer(claim_context, metrics_context, history_context, core_context, intent)
        )

        timestamp = _now_iso()
        session = session_id or "anonymous"
        self.repo.record_chat_turn(
            {
                "session_id": session,
                "report_id": report_id,
                "role": "user",
                "message": question,
                "timestamp": timestamp,
                "context_ids": {},
            }
        )
        self.repo.record_chat_turn(
            {
                "session_id": session,
                "report_id": report_id,
                "role": "assistant",
                "message": answer,
                "timestamp": _now_iso(),
                "context_ids": {"claims": claim_ids, "sources": list(source_ids)},
            }
        )

        return {
            "answer": answer,
            "claims_used": claim_context,
            "sources_used": list(source_ids),
        }

    def _llm_answer(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        evidence_map: DefaultDict[str, List[dict]],
        source_docs: Dict[str, Dict[str, Any]],
        report_doc: Dict[str, Any] | None,
        metrics_context: Optional[str],
        history_context: Optional[str],
        core_context: Optional[str],
        intent: str,
    ) -> str:
        if intent != "report":
            return self._small_talk_answer(question, history_context)
        if not claims:
            return self._compose_answer(claims, metrics_context, history_context, core_context, intent)

        report_summary = (
            (report_doc or {}).get("metadata", {}).get("summary")
            if report_doc and report_doc.get("metadata")
            else None
        )

        claim_blocks: List[str] = []
        for idx, claim in enumerate(claims, start=1):
            block_lines = [
                f"[Claim {idx}]",
                f"Text: {claim.get('text')}",
                f"Verdict: {claim.get('verdict')}",
                f"Confidence: {claim.get('confidence')}",
                f"Explanation: {claim.get('explanation')}",
            ]
            for evidence in evidence_map.get(claim["claim_id"], []):
                snippet = evidence.get("metadata", {}).get("snippet")
                source_id = evidence.get("source_id")
                doc_meta = source_docs.get(source_id, {})
                source_label = evidence.get("file_name") or source_id or "source"
                if snippet:
                    block_lines.append(f"Evidence ({source_label}): {snippet}")
                tables = (doc_meta.get("metadata") or {}).get("table_preview") or []
                if tables:
                    table_json = json.dumps(tables[0])[:500]
                    block_lines.append(f"Table snippet ({source_label}): {table_json}")
            claim_blocks.append("\n".join(block_lines))

        context_sections = []
        if report_summary:
            context_sections.append(f"Report Summary:\n{report_summary}")
        context_sections.append("Verified Claims and Evidence:\n" + "\n\n".join(claim_blocks))
        if core_context:
            context_sections.append(core_context)
        if metrics_context:
            context_sections.append(metrics_context)
        if history_context:
            context_sections.append(history_context)
        context_sections.append("User Question:\n" + question)
        prompt = "\n\n".join(context_sections)

        prompt = (
            "You are an assistant helping users understand a report verification run.\n"
            "Use ONLY the provided claims, evidence snippets, tables, and metric diagnostics to answer the question. "
            "If the claims do not cover the topic, say you do not have evidence.\n\n"
            f"{prompt}"
        )
        try:
            response = self.client.chat.completions.create(  # type: ignore[union-attr]
                model=self.settings.llm_chat_model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You ground answers strictly on verified claims supplied by the backend. "
                            "Never fabricate evidence. Cite claim numbers or verdicts when helpful."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            choice = response.choices[0].message.content if response.choices else None
            if choice:
                return self._format_response(choice)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChatService LLM call failed; falling back to heuristic answer: %s", exc)
        return self._compose_answer(claims, metrics_context, history_context, core_context, intent)

    def _compose_answer(
        self,
        claims: List[Dict[str, Any]],
        metrics_context: Optional[str],
        history_context: Optional[str],
        core_context: Optional[str],
        intent: str,
    ) -> str:
        if intent != "report":
            return self._small_talk_answer(None, history_context)
        lines = []
        if claims:
            lines.append("Key verified claims relevant to your question:")
            for claim in claims:
                verdict = claim.get("verdict", "UNKNOWN")
                explanation = claim.get("explanation") or claim.get("text")
                lines.append(f"- {verdict}: {explanation}")
        if core_context:
            lines.append("\nReport context:")
            lines.append(core_context)
        if metrics_context:
            lines.append("\nMetric diagnostics:")
            lines.append(metrics_context)
        if history_context:
            lines.append("\nConversation context:")
            lines.append(history_context)
        if not lines:
            return self._format_response(
                "I do not yet have verified claims or diagnostics for this report. Please run the pipeline first."
            )
        lines.append(
            "\nAnswer: Based on these findings, the evidence indicates the report's statements are handled as above."
        )
        return self._format_response("\n".join(lines))

    def _small_talk_answer(self, question: Optional[str], history_context: Optional[str]) -> str:
        if not self.client:
            return "Hello! I’m here to help with your report’s accuracy, credibility, and validity. Ask me about metrics, claims, or sources when you’re ready."
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly Author AI assistant. Engage in light small talk, "
                    "but keep the conversation professional and guide the user back to report quality topics when possible. "
                    "Avoid speculating about unrelated world knowledge."
                ),
            }
        ]
        if history_context:
            messages.append({"role": "user", "content": history_context})
        messages.append({"role": "user", "content": question or "Just say hello to the user."})
        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_chat_model,
                temperature=0.4,
                messages=messages,
            )
            choice = response.choices[0].message.content if response.choices else None
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
            if methodology:
                missed = [k for k, v in methodology.items() if not v]
                if missed:
                    metrics_lines.append(f"  Methodology gaps: {', '.join(missed)}.")

        if credibility_overall is not None:
            metrics_lines.append(f"- Credibility overall: {credibility_overall:.2f}")

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

        return "\n".join(metrics_lines)

    def _build_history_context(self, history: List[dict]) -> str:
        if not history:
            return ""
        lines = ["Recent conversation:"]
        for turn in history:
            prefix = "User" if turn["role"].lower() == "user" else "Assistant"
            lines.append(f"{prefix}: {turn['message']}")
        return "\n".join(lines)

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
