"""
Lightweight document summarisation helper with optional OpenAI support.
"""

from __future__ import annotations

import re
from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from ..config import get_settings
from .logger import setup_logger


logger = setup_logger(__name__)


class Summarizer:
    def __init__(self):
        self.settings = get_settings()
        if OpenAI and self.settings.openai_api_key:
            self.client: Optional[OpenAI] = OpenAI(api_key=self.settings.openai_api_key)
        else:
            self.client = None
            logger.info("Summaries default to heuristic mode (set OPENAI_API_KEY for LLM-powered summaries).")

    def summarize(self, text: str, max_sentences: int = 3, word_limit: Optional[int] = None) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        snippet = text[:4000]
        desired_words = word_limit or max_sentences * 25
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.explanation_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are summarising source documents for an accuracy pipeline. "
                                "Return a concise paragraph highlighting the document's main claims."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Summarise the following document in a single paragraph of approximately "
                                f"{desired_words} words (max {desired_words + 20} words). "
                                "Highlight why it is relevant for improving a food security report.\n"
                                f"{snippet}"
                            ),
                        },
                    ],
                    temperature=0.2,
                )
                choice = response.choices[0].message.content if response.choices else None
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
