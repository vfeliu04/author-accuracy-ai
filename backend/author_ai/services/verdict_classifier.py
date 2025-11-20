"""
LLM/heuristic verdict classifier that distinguishes supported vs contradicted evidence.
"""

from __future__ import annotations

import re
from typing import Dict, Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

from ..config import get_settings
from ..services.logger import setup_logger


logger = setup_logger(__name__)


NUMBER_PATTERN = re.compile(r"-?\d[\d,\.]*")


class VerdictClassifier:
    def __init__(self):
        self.settings = get_settings()
        if OpenAI and self.settings.openai_api_key:
            try:
                self.client = OpenAI(api_key=self.settings.openai_api_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("VerdictClassifier failed to init OpenAI client: %s", exc)
                self.client = None
        else:
            self.client = None

    def classify(self, claim: Any, evidence: Dict[str, Any]) -> Dict[str, Any]:
        snippet = (evidence.get("snippet") or evidence.get("text") or "").strip()
        parent = evidence.get("parent") or {}
        if not snippet:
            return {"label": "NOT_FOUND", "confidence": 0.0, "reason": "No evidence text provided."}

        parent_summary = parent.get("summary") or parent.get("text")
        metadata = claim.metadata if hasattr(claim, "metadata") else {}
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.explanation_model,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an accuracy evaluator. Given a numerical claim and an evidence passage, "
                                "respond with a single JSON line: {\"label\": \"SUPPORTED|CONTRADICTED|INCONCLUSIVE\", \"reason\": \"...\"}. "
                                "Label CONTRADICTED only when the evidence clearly disagrees with the claim's numeric assertion."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Claim: {claim.text}\n"
                                f"Claim metadata: {metadata}\n"
                                f"Evidence: {snippet}\n"
                                f"Additional context: {parent_summary}\n"
                                "Answer with JSON only."
                            ),
                        },
                    ],
                )
                content = response.choices[0].message.content if response.choices else None
                if content:
                    parsed = self._parse_json_line(content.strip())
                    if parsed:
                        return parsed
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM verdict classification failed, falling back to heuristics: %s", exc)
        return self._heuristic(metadata, claim.text, snippet)

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
            "confidence": float(data.get("confidence", 0.75 if label == "SUPPORTED" else 0.65)),
            "reason": data.get("reason") or "",
        }

    @staticmethod
    def _heuristic(metadata: dict, claim_text: str, snippet: str) -> Dict[str, Any]:
        claim_value = metadata.get("primary_value") or _extract_first_number(claim_text)
        evidence_value = _extract_first_number(snippet)
        if claim_value is None or evidence_value is None:
            return {"label": "SUPPORTED", "confidence": 0.55, "reason": "Heuristic fallback could not compare numbers."}
        try:
            claim_num = float(claim_value.replace(",", ""))
            evidence_num = float(evidence_value.replace(",", ""))
        except ValueError:
            return {"label": "SUPPORTED", "confidence": 0.55, "reason": "Failed to parse numeric values."}
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
