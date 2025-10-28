from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .tokens import count_tokens


# A type alias to make the code more readable.
# Represents a single sentence as a tuple of (page_number, text, token_count).
Sentence = Tuple[int, str, int]


# A simple data class to hold the information for a single chunk of text.
@dataclass
class Chunk:
    page_start: int      # The page number where the chunk begins.
    page_end: int        # The page number where the chunk ends.
    chunk_text: str      # The actual text content of the chunk.
    token_count: int     # The number of tokens in the chunk.
    chunk_id: str        # A unique identifier for the chunk.


# A helper function to split a block of text into sentences.
# This uses a regular expression to split on periods, question marks, and exclamation points.
def _split_sentences(text: str) -> List[str]:
    """Split text into rough sentence-sized units."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


# A helper function to combine a sequence of sentences into a single `Chunk` object.
def _make_chunk(sentences: Sequence[Sentence]) -> Chunk:
    page_start = sentences[0][0]
    page_end = sentences[-1][0]
    text = " ".join(sentence for _, sentence, _ in sentences).strip()
    token_count = sum(tokens for _, _, tokens in sentences)
    # Create a unique hash for the chunk based on its content and location.
    # This is used for deduplication.
    chunk_hash = hashlib.sha256(f"{page_start}:{page_end}:{text}".encode("utf-8")).hexdigest()
    return Chunk(
        page_start=page_start,
        page_end=page_end,
        chunk_text=text,
        token_count=token_count,
        chunk_id=chunk_hash,
    )


# This is the main function for chunking the text.
# It takes text from PDF pages and splits it into chunks of a target size with some overlap.
def chunk_text(
    pages: Iterable[Tuple[int, str]],
    target_tokens: int = 900,    # The desired size of each chunk in tokens.
    overlap_tokens: int = 100,  # The number of tokens to overlap between consecutive chunks.
) -> List[Dict[str, int | str]]:
    """Chunk PDF page text into embedding-ready payloads."""
    chunks: List[Chunk] = []
    buffer: List[Sentence] = []  # A temporary list to build up sentences for the next chunk.
    buffer_tokens = 0

    # Iterate through each page from the PDF.
    for page_number, raw_text in pages:
        if not raw_text:
            continue
        # Split the page text into sentences.
        sentences = _split_sentences(raw_text)
        if not sentences:
            sentences = [raw_text.strip()]  # Handle cases where a page has no clear sentences.

        # Process each sentence on the page.
        for sentence in sentences:
            token_count = count_tokens(sentence)
            sentence_tuple: Sentence = (page_number, sentence, token_count)

            # Special case: If a single sentence is larger than our target,
            # we make it its own chunk and handle the overlap separately.
            if token_count >= target_tokens:
                if buffer:
                    chunks.append(_make_chunk(buffer))
                    buffer = _prepare_overlap(buffer, overlap_tokens)
                    buffer_tokens = sum(tokens for _, _, tokens in buffer)
                oversized_chunk = _make_chunk([sentence_tuple])
                chunks.append(oversized_chunk)
                buffer = _prepare_overlap([sentence_tuple], overlap_tokens)
                buffer_tokens = sum(tokens for _, _, tokens in buffer)
                continue

            # If adding the next sentence would push the buffer over the target size,
            # finalize the current buffer as a chunk.
            if buffer_tokens + token_count > target_tokens and buffer:
                chunks.append(_make_chunk(buffer))
                # Start the new buffer with the overlapping sentences from the end of the last chunk.
                buffer = _prepare_overlap(buffer, overlap_tokens)
                buffer_tokens = sum(tokens for _, _, tokens in buffer)

            # Add the current sentence to the buffer.
            buffer.append(sentence_tuple)
            buffer_tokens += token_count

    # After the loop, if there's anything left in the buffer, make it the final chunk.
    if buffer:
        chunks.append(_make_chunk(buffer))

    # Convert the list of `Chunk` objects into a list of dictionaries,
    # which is a more common format for JSON serialization.
    return [
        {
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "chunk_text": chunk.chunk_text,
            "token_count": chunk.token_count,
            "chunk_id": chunk.chunk_id,
        }
        for chunk in chunks
    ]


# This helper function creates the overlapping portion for the next chunk.
def _prepare_overlap(sentences: Sequence[Sentence], overlap_tokens: int) -> List[Sentence]:
    """Return trailing sentences that satisfy the overlap token target."""
    overlap: List[Sentence] = []
    total_tokens = 0
    # Go through the sentences of the last chunk in reverse order.
    for sentence in reversed(sentences):
        total_tokens += sentence[2]
        overlap.insert(0, sentence)  # Add to the beginning to maintain original order.
        # Stop once we have enough tokens for the desired overlap.
        if total_tokens >= overlap_tokens:
            break
    return overlap
