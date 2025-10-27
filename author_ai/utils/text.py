"""Utility helpers for working with text spans safely."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

# A simple list of locations to check for.
# This could be expanded or replaced with a more sophisticated NER approach.
KNOWN_LOCATIONS = [
    "UK",
    "United Kingdom",
    "USA",
    "United States",
    "China",
    "India",
    "Germany",
    "France",
]


def extract_location_from_subject(subject: Optional[str]) -> Optional[str]:
    """Extract a known location from the subject text."""
    if not subject:
        return None
    for location in KNOWN_LOCATIONS:
        if location in subject:
            return location
    return None


def spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Return True if two spans overlap."""

    return max(a[0], b[0]) < min(a[1], b[1])


def is_span_free(spans: Iterable[Tuple[int, int]], candidate: Tuple[int, int]) -> bool:
    """Check if a candidate span does not overlap existing spans."""

    for span in spans:
        if spans_overlap(span, candidate):
            return False
    return True


def expand_to_word_boundaries(text: str, start: int, end: int) -> Tuple[int, int]:
    """Expand a span to token boundaries, keeping indices within the text."""

    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end : end + 1].isalnum():
        end += 1
    return start, end


def extract_window(text: str, start: int, end: int, width: int = 48) -> str:
    """Return a snippet of text surrounding a span."""

    left = max(start - width, 0)
    right = min(end + width, len(text))
    return text[left:right]


def slice_text(text: str, span: Tuple[int, int]) -> str:
    """Return a safe substring for the provided span."""

    start, end = span
    start = max(0, min(len(text), start))
    end = max(start, min(len(text), end))
    return text[start:end]


def extract_subject(text: str, span: Tuple[int, int]) -> Optional[str]:
    """Extract the subject of a claim."""
    # This is a placeholder implementation. A more sophisticated implementation
    # would use NLP to identify the true subject of the claim.
    start, end = span
    window = extract_window(text, start, end)
    # For now, we'll just return the window as the subject.
    return window
