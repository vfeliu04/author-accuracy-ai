"""
LLM/heuristic verdict classifier that distinguishes supported vs contradicted evidence.
"""

from __future__ import annotations

import re
from typing import Dict, Any

try:
    import anthropic as _anthropic
except ImportError:  # pragma: no cover
    _anthropic = None  # type: ignore

from ..config import get_settings
from ..services.logger import setup_logger
import time


logger = setup_logger(__name__)


NUMBER_PATTERN = re.compile(r"-?\d[\d,\.]*")


class VerdictClassifier:
    def __init__(self):
        self.settings = get_settings()
        if _anthropic and self.settings.anthropic_api_key:
            try:
                self.client = _anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("VerdictClassifier failed to init Anthropic client: %s", exc)
                self.client = None
        else:
            self.client = None

    def classify(self, claim: Any, evidence: Dict[str, Any]) -> Dict[str, Any]:
        snippet = (evidence.get("snippet") or evidence.get("text") or "").strip()
        parent = evidence.get("parent") or {}
        if not snippet:
            return {"label": "NOT_FOUND", "confidence": 0.0, "reason": "No evidence text provided.", "mode": "heuristic"}

        parent_summary = parent.get("summary") or parent.get("text")
        claim_metadata = claim.metadata if hasattr(claim, "metadata") else {}
        evidence_meta = evidence.get("metadata") or {}
        chunk_type = (evidence.get("chunk_type") or evidence_meta.get("chunk_type") or "") or ""
        is_chart = chunk_type.startswith("chart_")
        figure_label = evidence.get("figure_label") or evidence_meta.get("figure_label") or "unknown"
        structured_bits = []
        for key, label in (
            ("x_value", "x"),
            ("y_value", "y"),
            ("series_name", "series"),
            ("y_unit", "unit"),
        ):
            value = evidence.get(key) or evidence_meta.get(key)
            if value is not None:
                structured_bits.append(f"{label}={value}")
        structured_line = ", ".join(structured_bits) if structured_bits else None
        evidence_text = (
            f"Evidence from chart (Figure {figure_label}, Page {parent.get('page') or 'unknown'}):\n"
            f"Summary: {snippet}"
        ) if is_chart else snippet
        if structured_line:
            evidence_text = f"{evidence_text}\nStructured data: {structured_line}"
        heuristic_hint = self._heuristic(claim_metadata, claim.text, snippet)

        if self.client:
            try:
                _system_prompt = (
                    "You are an accuracy evaluator. Given a claim from a report and an evidence passage from a source document, "
                    "decide whether the evidence genuinely SUPPORTS, CONTRADICTS, or is INCONCLUSIVE about the claim. "
                    "Respond with a single JSON line: {\"label\": \"SUPPORTED|CONTRADICTED|INCONCLUSIVE\", \"reason\": \"...\"}. "
                    "Rules: "
                    "1) SUPPORTED: The evidence explicitly and factually asserts the same information as the claim. "
                    "   Numeric values must closely align AND the evidence must endorse the claim (not merely mention a similar number in a different context). "
                    "2) CONTRADICTED: The evidence explicitly conflicts with the claim — different numbers, opposite conclusions, or the evidence calls the claim false/fabricated. "
                    "   If the claim text itself uses words like 'fabricated', 'falsely', 'fake', or 'does not exist', mark CONTRADICTED. "
                    "3) INCONCLUSIVE: The evidence is tangentially related but does not clearly confirm or deny the claim. "
                    "   When in doubt between SUPPORTED and INCONCLUSIVE, choose INCONCLUSIVE. "
                    "   Do NOT mark SUPPORTED just because a number appears in both the claim and the evidence. "
                    "4) If the claim year is unspecified, a single evidence year is acceptable unless clearly far off. "
                    "5) Some evidence comes from charts; compare claim against chart data just like textual evidence. "
                    "6) Pay close attention to years. If the claim references year X and evidence references year Y where |X-Y| > 2, mark INCONCLUSIVE unless the evidence explicitly says 'as of year X' or makes a direct comparison. Different years = different facts. "
                    "Confidence guidance: 0.9-1.0: Evidence explicitly states the exact same fact, numbers match exactly. 0.7-0.9: Evidence clearly supports/contradicts with only minor differences. 0.5-0.7: Evidence partially supports/contradicts. 0.3-0.5: Weak or indirect relationship. 0.0-0.3: Very tenuous."
                )
                _user_content = (
                    f"Claim: {claim.text}\n"
                    f"Claim metadata: {claim_metadata}\n"
                    f"Evidence: {evidence_text}\n"
                    f"Additional context: {parent_summary}\n"
                    f"Heuristic suggestion: {heuristic_hint}\n"
                    "Answer with JSON only."
                )

                def _try_classify(retries: int = 2, delay: float = 1.0):
                    attempt = 0
                    while True:
                        try:
                            return self.client.messages.create(  # type: ignore[union-attr]
                                model=self.settings.explanation_model,
                                max_tokens=256,
                                system=_system_prompt,
                                messages=[{"role": "user", "content": _user_content}],
                            )
                        except Exception as exc:  # noqa: BLE001
                            attempt += 1
                            if attempt > retries:
                                raise
                            logger.warning("LLM verdict request failed (attempt %d/%d): %s", attempt, retries + 1, exc)
                            time.sleep(delay * attempt)

                response = _try_classify()
                text_blocks = [b.text for b in response.content if b.type == "text"]
                content = text_blocks[0] if text_blocks else None
                if content:
                    parsed = self._parse_json_line(content.strip())
                    if parsed:
                        parsed["mode"] = "llm"
                        return parsed
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM verdict classification failed, falling back to heuristics: %s", exc)

        # Fall back to heuristic when LLM is unavailable or inconclusive.
        return {**heuristic_hint, "mode": "heuristic"}

    def classify_multi(self, claim: Any, evidence_list: list[Dict[str, Any]]) -> Dict[str, Any]:
        if not evidence_list:
            return {"label": "NOT_FOUND", "confidence": 0.0, "reason": "No evidence provided.", "mode": "heuristic"}
        if len(evidence_list) == 1:
            return self.classify(claim, evidence_list[0])

        # Build a numbered evidence block for each item
        evidence_blocks = []
        for i, ev in enumerate(evidence_list, start=1):
            snippet = (ev.get("snippet") or ev.get("text") or "").strip()
            doc_id = ev.get("doc_id") or "unknown"
            parent = ev.get("parent") or {}
            page = parent.get("page") or ev.get("parent_page") or "unknown"
            evidence_blocks.append(f"[{i}] Source: {doc_id}, Page: {page}\n{snippet}")
        evidence_text = "\n\n".join(evidence_blocks)

        if self.client:
            try:
                _system_prompt = (
                    "You are an accuracy evaluator. Given a claim from a report and multiple evidence passages from source documents, "
                    "decide whether the evidence as a whole SUPPORTS, CONTRADICTS, or is INCONCLUSIVE about the claim. "
                    "Respond with a single JSON line: {\"label\": \"SUPPORTED|CONTRADICTED|INCONCLUSIVE\", \"confidence\": 0.0-1.0, \"reason\": \"...\"}. "
                    "Rules: "
                    "1) SUPPORTED: The evidence explicitly and factually asserts the same information as the claim. "
                    "   Numeric values must closely align AND the evidence must endorse the claim (not merely mention a similar number in a different context). "
                    "2) CONTRADICTED: The evidence explicitly conflicts with the claim — different numbers, opposite conclusions, or the evidence calls the claim false/fabricated. "
                    "   If the claim text itself uses words like 'fabricated', 'falsely', 'fake', or 'does not exist', mark CONTRADICTED. "
                    "3) INCONCLUSIVE: The evidence is tangentially related but does not clearly confirm or deny the claim. "
                    "   When in doubt between SUPPORTED and INCONCLUSIVE, choose INCONCLUSIVE. "
                    "   Do NOT mark SUPPORTED just because a number appears in both the claim and the evidence. "
                    "4) If the claim year is unspecified, a single evidence year is acceptable unless clearly far off. "
                    "5) Some evidence comes from charts; compare claim against chart data just like textual evidence. "
                    "6) Pay close attention to years. If the claim references year X and evidence references year Y where |X-Y| > 2, mark INCONCLUSIVE unless the evidence explicitly says 'as of year X' or makes a direct comparison. Different years = different facts. "
                    "Confidence guidance: 0.9-1.0: Evidence explicitly states the exact same fact, numbers match exactly. 0.7-0.9: Evidence clearly supports/contradicts with only minor differences. 0.5-0.7: Evidence partially supports/contradicts. 0.3-0.5: Weak or indirect relationship. 0.0-0.3: Very tenuous."
                )
                _user_content = (
                    f"Claim: {claim.text}\n"
                    f"Claim metadata: {claim.metadata if hasattr(claim, 'metadata') else {}}\n\n"
                    f"Evidence passages:\n{evidence_text}\n"
                    "Answer with JSON only."
                )

                def _try_classify_multi(retries: int = 2, delay: float = 1.0):
                    attempt = 0
                    while True:
                        try:
                            return self.client.messages.create(  # type: ignore[union-attr]
                                model=self.settings.explanation_model,
                                max_tokens=256,
                                system=_system_prompt,
                                messages=[{"role": "user", "content": _user_content}],
                            )
                        except Exception as exc:  # noqa: BLE001
                            attempt += 1
                            if attempt > retries:
                                raise
                            logger.warning("LLM multi-verdict request failed (attempt %d/%d): %s", attempt, retries + 1, exc)
                            time.sleep(delay * attempt)

                response = _try_classify_multi()
                text_blocks = [b.text for b in response.content if b.type == "text"]
                content = text_blocks[0] if text_blocks else None
                if content:
                    parsed = self._parse_json_line(content.strip())
                    if parsed:
                        parsed["mode"] = "llm_multi"
                        return parsed
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM multi-verdict classification failed, falling back to heuristics: %s", exc)

        # Fall back to heuristic on the best hit (first evidence item)
        best = evidence_list[0]
        snippet = (best.get("snippet") or best.get("text") or "").strip()
        claim_metadata = claim.metadata if hasattr(claim, "metadata") else {}
        heuristic_hint = self._heuristic(claim_metadata, claim.text, snippet)
        return {**heuristic_hint, "mode": "heuristic"}

    @staticmethod
    def _parse_json_line(content: str) -> Dict[str, Any] | None:
        content = content.strip()
        if not content:
            return None
        # naive parsing to avoid importing json for malformed text
        import json

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
        label = (data.get("label") or "").upper()
        if label not in {"SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"}:
            return None
        return {
            "label": label,
            "confidence": float(data.get("confidence", 0.5)),
            "reason": data.get("reason") or "",
        }

    @staticmethod
    def _heuristic(metadata: dict, claim_text: str, snippet: str) -> Dict[str, Any]:
        claim_value = metadata.get("primary_value") or _extract_first_number(claim_text)
        evidence_value = _extract_first_number(snippet)
        if claim_value is None or evidence_value is None:
            return {"label": "INCONCLUSIVE", "confidence": 0.5, "reason": "Heuristic fallback could not compare numbers."}
        try:
            claim_num = float(claim_value.replace(",", ""))
            evidence_num = float(evidence_value.replace(",", ""))
        except ValueError:
            return {"label": "INCONCLUSIVE", "confidence": 0.5, "reason": "Failed to parse numeric values."}
        if claim_num == 0:
            diff_ratio = abs(evidence_num - claim_num)
        else:
            diff_ratio = abs(evidence_num - claim_num) / abs(claim_num)
        if diff_ratio >= 0.15:
            return {
                "label": "CONTRADICTED",
                "confidence": 0.65,
                "reason": f"Heuristic numeric comparison differs by {diff_ratio:.0%}.",
            }
        return {
            "label": "SUPPORTED",
            "confidence": 0.6,
            "reason": "Heuristic numeric comparison within tolerance.",
        }


def _extract_first_number(text: str | None) -> str | None:
    if not text:
        return None
    match = NUMBER_PATTERN.search(text)
    return match.group(0) if match else None
