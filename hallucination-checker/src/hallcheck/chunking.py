"""Chunking utilities for splitting documents."""

from __future__ import annotations

import re
from typing import List

import nltk


def sent_tokenize(text: str) -> List[str]:
    """Wrapper to tokenize sentences."""
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        # Fallback for environments without punkt resources.
        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                continue
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]


def chunk_text(text: str, target_tokens: int, overlap: int) -> List[str]:
    """Split text into sentence-aware chunks using approximate token lengths."""
    if not text.strip():
        return []

    sentences = sent_tokenize(text)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def count_tokens(sentence: str) -> int:
        return max(1, len(sentence.split()))

    for sentence in sentences:
        sent_tokens = count_tokens(sentence)
        if current and current_tokens + sent_tokens > target_tokens:
            chunks.append(" ".join(current).strip())
            if overlap > 0:
                overlap_tokens = 0
                overlap_buffer: List[str] = []
                for prev_sentence in reversed(current):
                    overlap_buffer.insert(0, prev_sentence)
                    overlap_tokens += count_tokens(prev_sentence)
                    if overlap_tokens >= overlap:
                        break
                current = overlap_buffer.copy()
                current_tokens = sum(count_tokens(s) for s in current)
            else:
                current = []
                current_tokens = 0
        current.append(sentence.strip())
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]
