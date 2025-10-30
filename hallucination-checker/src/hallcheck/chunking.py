"""Chunking utilities for splitting documents."""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Sequence, Tuple

import nltk
from rapidfuzz import fuzz


def sent_tokenize(text: str) -> List[str]:
    """Wrapper to tokenize sentences with graceful fallbacks."""
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                continue
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]


def chunk_text(
    text: str,
    target_tokens: int,
    overlap: int,
    *,
    min_overlap: int | None = None,
    max_overlap: int | None = None,
    topic_threshold: float | None = None,
    stable_threshold: float | None = None,
) -> List[str]:
    """Split text into coherent, sentence-aware chunks using adaptive windows."""
    if not text.strip():
        return []

    segments = _segment_text(text)
    chunks: List[str] = []

    min_overlap = min_overlap if min_overlap is not None else max(5, overlap // 3)
    max_overlap = max_overlap if max_overlap is not None else max(overlap, min_overlap)
    topic_threshold = topic_threshold if topic_threshold is not None else 0.72
    stable_threshold = stable_threshold if stable_threshold is not None else 0.86

    for segment in segments:
        segment_chunks = _adaptive_chunk_segment(
            segment,
            target_tokens=target_tokens,
            base_overlap=overlap,
            min_overlap=min_overlap,
            max_overlap=max_overlap,
            topic_threshold=topic_threshold,
            stable_threshold=stable_threshold,
        )
        chunks.extend(segment_chunks)

    return [chunk for chunk in chunks if chunk.strip()]


def _segment_text(text: str) -> List[str]:
    """Coarse segmentation using TextTiling with fallbacks to fixed-size blocks."""
    cleaned = _normalize_whitespace(text)
    sentences = sent_tokenize(cleaned)
    if len(sentences) < 4:
        return [cleaned]

    try:
        from nltk.tokenize import texttiling

        tokenizer = texttiling.TextTilingTokenizer()
        tiles = tokenizer.tokenize(cleaned)
        if tiles:
            return [tile.strip() for tile in tiles if tile.strip()]
    except Exception:
        pass

    blocks: List[str] = []
    buffer: List[str] = []
    for sentence in sentences:
        buffer.append(sentence)
        if len(buffer) >= 10:
            blocks.append(" ".join(buffer).strip())
            buffer = []
    if buffer:
        blocks.append(" ".join(buffer).strip())
    return [block for block in blocks if block]


def _adaptive_chunk_segment(
    segment: str,
    *,
    target_tokens: int,
    base_overlap: int,
    min_overlap: int,
    max_overlap: int,
    topic_threshold: float,
    stable_threshold: float,
) -> List[str]:
    sentences = [s.strip() for s in sent_tokenize(segment) if s.strip()]
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_tokens = 0
    current_vocab: Counter[str] = Counter()
    similarity_history: List[float] = []

    for sentence in sentences:
        sentence_tokens = _tokenize_words(sentence)
        sentence_token_count = max(1, len(sentence_tokens))
        similarity = _bag_similarity(current_vocab, sentence_tokens) if current_sentences else 1.0

        exceeds_limit = current_sentences and (current_tokens + sentence_token_count > target_tokens)
        topic_shift = current_sentences and similarity < topic_threshold

        if current_sentences and (exceeds_limit or topic_shift):
            chunks.append(" ".join(current_sentences).strip())
            overlap_tokens = _determine_overlap(
                similarity_history,
                min_overlap=min_overlap,
                max_overlap=max_overlap,
                base_overlap=base_overlap,
                stable_threshold=stable_threshold,
            )
            current_sentences = _carry_overlap(current_sentences, overlap_tokens)
            current_tokens = sum(max(1, len(_tokenize_words(s))) for s in current_sentences)
            current_vocab = Counter()
            for s in current_sentences:
                current_vocab.update(_tokenize_words(s))
            similarity_history = []

        current_sentences.append(sentence)
        current_tokens += sentence_token_count
        current_vocab.update(sentence_tokens)
        if current_sentences and similarity is not None:
            similarity_history.append(similarity)

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return chunks


def _tokenize_words(sentence: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", sentence)]


def _bag_similarity(current_vocab: Counter[str], sentence_tokens: Sequence[str]) -> float:
    if not current_vocab or not sentence_tokens:
        return 0.0
    chunk_terms = set(current_vocab.keys())
    sentence_terms = set(sentence_tokens)
    if not chunk_terms or not sentence_terms:
        return 0.0
    intersection = len(chunk_terms & sentence_terms)
    union = len(chunk_terms | sentence_terms)
    if union == 0:
        return 0.0
    return intersection / union


def _determine_overlap(
    similarity_history: Sequence[float],
    *,
    min_overlap: int,
    max_overlap: int,
    base_overlap: int,
    stable_threshold: float,
) -> int:
    if not similarity_history:
        return base_overlap
    avg_similarity = sum(similarity_history) / max(1, len(similarity_history))
    if avg_similarity >= stable_threshold:
        return min_overlap
    if avg_similarity <= 0.5:
        return max_overlap
    ratio = (stable_threshold - avg_similarity) / max(stable_threshold - 0.5, 1e-6)
    overlap = min_overlap + ratio * (max_overlap - min_overlap)
    return int(round(overlap))


def _carry_overlap(sentences: Sequence[str], overlap_tokens: int) -> List[str]:
    if overlap_tokens <= 0 or not sentences:
        return []
    carried: List[str] = []
    token_budget = 0
    for sentence in reversed(sentences):
        carried.insert(0, sentence)
        token_budget += max(1, len(_tokenize_words(sentence)))
        if token_budget >= overlap_tokens:
            break
    return carried


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def best_sentence_snippet(claim: str, chunk_text: str, max_sentences: int = 3) -> Tuple[str | None, float]:
    """Return the most relevant sentence snippet for a claim from a chunk."""
    sentences = [s.strip() for s in sent_tokenize(chunk_text) if s.strip()]
    if not sentences:
        return None, 0.0

    scores = [
        (fuzz.token_set_ratio(claim, sentence), idx, sentence)
        for idx, sentence in enumerate(sentences)
    ]
    scores.sort(reverse=True)
    top_score, best_idx, best_sentence = scores[0]
    snippet_sentences = [best_sentence]

    for offset in (-1, 1):
        neighbour_idx = best_idx + offset
        if 0 <= neighbour_idx < len(sentences) and len(snippet_sentences) < max_sentences:
            neighbour = sentences[neighbour_idx]
            neighbour_score = fuzz.partial_ratio(claim, neighbour)
            if neighbour_score > max(55, 0.6 * top_score):
                snippet_sentences.append(neighbour)

    snippet = " ".join(snippet_sentences).strip()
    return snippet, float(top_score)
