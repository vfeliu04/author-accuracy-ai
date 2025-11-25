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
import time


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
            return {"label": "NOT_FOUND", "confidence": 0.0, "reason": "No evidence text provided.", "mode": "heuristic"}

        parent_summary = parent.get("summary") or parent.get("text")
        metadata = claim.metadata if hasattr(claim, "metadata") else {}
        heuristic_hint = self._heuristic(metadata, claim.text, snippet)

        if self.client:
            try:
                def _try_classify(retries: int = 2, delay: float = 1.0):
                    attempt = 0
                    while True:
                        try:
                            return self.client.chat.completions.create(  # type: ignore[union-attr]
                                model=self.settings.explanation_model,
                                temperature=0.0,
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are an accuracy evaluator. Given a numerical claim and an evidence passage, "
                                            "respond with a single JSON line: {\"label\": \"SUPPORTED|CONTRADICTED|INCONCLUSIVE\", \"reason\": \"...\"}. "
                                            "Tie-breakers: "
                                            "1) If the claim year is unspecified, treat a single evidence year as acceptable unless it is clearly far in the past/future or contradicts the claim context; do not mark INCONCLUSIVE solely because the claim omits a year. "
                                            "2) Prefer SUPPORTED when numeric values align closely and timeframe is acceptable (see 1). "
                                            "3) Mark CONTRADICTED only when numbers or stated timeframe clearly conflict. "
                                            "4) If a heuristic suggestion is provided, follow it unless the evidence strongly conflicts with it."
                                        ),
                                    },
                                    {
                                        "role": "user",
                                        "content": (
                                            f"Claim: {claim.text}\n"
                                            f"Claim metadata: {metadata}\n"
                                            f"Evidence: {snippet}\n"
                                            f"Additional context: {parent_summary}\n"
                                            f"Heuristic suggestion: {heuristic_hint}\n"
                                            "Answer with JSON only."
                                        ),
                                    },
                                ],
                            )
                        except Exception as exc:  # noqa: BLE001
                            attempt += 1
                            if attempt > retries:
                                raise
                            logger.warning("LLM verdict request failed (attempt %d/%d): %s", attempt, retries + 1, exc)
                            time.sleep(delay * attempt)

                response = _try_classify()
                content = response.choices[0].message.content if response.choices else None
                if content:
                    parsed = self._parse_json_line(content.strip())
                    if parsed:
                        parsed["mode"] = "llm"
                        return parsed
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM verdict classification failed, falling back to heuristics: %s", exc)

        # Fall back to heuristic when LLM is unavailable or inconclusive.
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
