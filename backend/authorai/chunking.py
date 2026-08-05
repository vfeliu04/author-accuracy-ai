"""Plain-code text chunking: pack paragraphs into bounded, overlapping chunks.

No LLM framework needed for this — it is pure bookkeeping. Paragraphs are
packed greedily up to `max_chars`; a paragraph that alone exceeds the cap is
split on sentence boundaries; consecutive chunks share `overlap` characters of
context so a fact straddling a boundary is retrievable from either side.

Document order is an invariant: chunks always contain contiguous source text
in source order — a chunk must never splice together non-adjacent passages,
because retrieved chunk text is quoted as evidence downstream.
"""

import re

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    pieces: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
        else:
            pieces.extend(_split_long(paragraph, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        # `current` is never empty here: every piece is pre-bounded to
        # max_chars, so a lone piece always fits.
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        current = f"{tail}\n\n{piece}" if tail else piece
        if len(current) > max_chars:
            # Carrying the overlap tail would exceed the cap; drop the tail.
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries, hard-wrapping any
    single sentence that still exceeds the cap. Emits parts strictly in
    document order."""
    parts: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            parts.append(current)
            current = ""

    for sentence in _SENTENCE_SPLIT.split(paragraph):
        if len(sentence) > max_chars:
            # Flush accumulated sentences FIRST so the wrapped segments of
            # this long sentence land after them, never before.
            flush()
            while len(sentence) > max_chars:
                parts.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            current = sentence
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            flush()
            current = sentence
    flush()
    return parts
