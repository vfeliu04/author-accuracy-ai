"""
Lightweight document summarisation helper with optional Anthropic support.
"""

from __future__ import annotations

import re
from typing import Optional
import time

try:
    import anthropic as _anthropic
except ImportError:  # pragma: no cover - optional dependency
    _anthropic = None  # type: ignore

from ..config import get_settings
from .logger import setup_logger


logger = setup_logger(__name__)


class Summarizer:
    def __init__(self):
        self.settings = get_settings()
        if _anthropic and self.settings.anthropic_api_key:
            self.client = _anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        else:
            self.client = None
            logger.info("Summaries default to heuristic mode (set ANTHROPIC_API_KEY for LLM-powered summaries).")

    def summarize(self, text: str, max_sentences: int = 3, word_limit: Optional[int] = None) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        snippet = text[:4000]
        desired_words = word_limit or max_sentences * 25
        if self.client:
            try:
                def _try_summarize(retries: int = 2, delay: float = 1.0):
                    attempt = 0
                    while True:
                        try:
                            return self.client.messages.create(  # type: ignore[union-attr]
                                model=self.settings.explanation_model,
                                max_tokens=512,
                                system=(
                                    "You are summarising source documents for an accuracy pipeline. "
                                    "Return a concise paragraph highlighting the document's main claims."
                                ),
                                messages=[{
                                    "role": "user",
                                    "content": (
                                        "Summarise the following document in a single paragraph of approximately "
                                        f"{desired_words} words (max {desired_words + 20} words). "
                                        "Highlight why it is relevant for improving a food security report.\n"
                                        f"{snippet}"
                                    ),
                                }],
                            )
                        except Exception as exc:  # noqa: BLE001
                            attempt += 1
                            if attempt > retries:
                                raise
                            logger.warning("LLM summary request failed (attempt %d/%d): %s", attempt, retries + 1, exc)
                            time.sleep(delay * attempt)

                response = _try_summarize()
                text_blocks = [b.text for b in response.content if b.type == "text"]
                choice = text_blocks[0] if text_blocks else None
                if choice:
                    return choice.strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM summary failed, falling back to heuristic summary: %s", exc)

        logger.info("Summarizer using heuristic fallback for text length %d", len(snippet))
        if word_limit:
            return self._fallback_word_limit(snippet, word_limit)
        return self._fallback(snippet, max_sentences)

    @staticmethod
    def _fallback(text: str, max_sentences: int) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:max_sentences]).strip()


    @staticmethod
    def _fallback_word_limit(text: str, word_limit: int) -> str:
        words = text.split()
        if not words:
            return ""
        limited = words[:word_limit]
        return " ".join(limited).strip()


_SUMMARIZER = Summarizer()


def summarize_text(text: str, max_sentences: int = 3, word_limit: Optional[int] = None) -> str:
    return _SUMMARIZER.summarize(text, max_sentences=max_sentences, word_limit=word_limit)
