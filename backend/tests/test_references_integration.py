"""Real-PDF check of the bibliography text slice — excluded in CI (needs the
example PDFs). Run locally with: python -m pytest -m integration -q

Where each report's reference list is found was measured on the real files;
these bounds hold the measurement so a regression in the heading rule, the
run-gap rule or the page reader is visible on the real files. Re-measure
and re-pin when a rule or a cap changes.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

EXAMPLES = Path(__file__).resolve().parents[2] / "example_sources"

DROUGHT = "example source two/Drought Hotspots 2023-2025_ENG.pdf"

CASES = [
    # (relative path, expected text_source, (min chars, max chars), the text
    # the slice must begin with and end with — None for a tail slice)
    #
    # The Drought report's Works Cited list runs 11 pages (38 to 48 of the
    # 51 read) and repeats its heading on each as a running header, so the
    # headings are one page apart (within HEADING_RUN_PAGES) and the slice
    # starts at the FIRST of the 11, on the entry "Addis Standard". From
    # that heading the list runs 61,616 characters to the end of the file —
    # under REFERENCE_MAX_CHARS, so the slice is the ENTIRE list, through
    # its last entry (Zkhiri, DOI 10.1007/s00704-018-2388-6) and the back
    # cover's imprint after it. Before the running-header rule the slice
    # was the last page alone: 4,443 characters; under the 60,000 cap it
    # lost the last page's final seven entries.
    (
        DROUGHT,
        "heading",
        (61_616, 61_616),
        "WORKS CITED\nAddis Standard. (2024, January 10). News: Four million",
        "https://doi.org/10.1007/s00704-018-2388-6  \n\n"
        "International Drought\nResilience AllianceIDRA",
    ),
    # One heading, 7,436 characters from it to the end of the file: a 7,435-character slice.
    (
        "example source two/Defining_domestic_water_consumption_based_on_perso.pdf",
        "heading",
        (7_000, 7_500),
        "REFERENCES\nAdler, I. 2011 Domestic water demand management",
        None,
    ),
    # No heading: the whole (short) document is the tail, 4,519 characters stripped.
    ("Water_Stress_Fake_Report.pdf", "tail", (4_000, 5_000), None, None),
    # No heading: the whole (short) document is the tail, 5,664 characters stripped.
    ("example source one/World_Hunger_Fake.pdf", "tail", (5_000, 6_000), None, None),
]


def _slice(relative: str):
    from authorai.references import read_pages, reference_text

    path = EXAMPLES / relative
    if not path.exists():
        pytest.skip(f"example PDF not present: {path}")
    with path.open("rb") as handle:
        pages = read_pages(handle)
    assert pages, "expected at least one page"
    return reference_text(pages)


@pytest.mark.parametrize(
    ("relative", "expected_source", "bounds", "starts_with", "ends_with"), CASES
)
def test_real_reports_slice_where_measured(
    relative, expected_source, bounds, starts_with, ends_with
):
    text, source = _slice(relative)
    assert source == expected_source
    low, high = bounds
    assert low <= len(text) <= high, f"{relative}: {len(text)} chars from {source!r}"
    if starts_with is not None:
        assert text.startswith(starts_with), f"{relative} begins {text[:80]!r}"
    if ends_with is not None:
        assert text.endswith(ends_with), f"{relative} ends {text[-120:]!r}"


def test_the_largest_real_list_splits_into_the_chunks_the_cost_comment_counts():
    """The worst case the constants' comment prices: the Drought list is six
    chunks — five near REFERENCE_CHUNK_CHARS and the remainder — cut at
    line breaks only, every line of the list in exactly one chunk, and the
    last entry (Zkhiri) in the last chunk."""
    from authorai.references import REFERENCE_CHUNK_CHARS, split_reference_text

    text, _ = _slice(DROUGHT)
    chunks = split_reference_text(text)
    assert len(chunks) == 6
    assert all(len(chunk) <= REFERENCE_CHUNK_CHARS for chunk in chunks)
    assert all(len(chunk) > REFERENCE_CHUNK_CHARS * 0.9 for chunk in chunks[:-1])
    lines = [line for line in text.split("\n") if line.strip()]
    assert [line for chunk in chunks for line in chunk.split("\n") if line.strip()] == lines
    assert "Zkhiri, W., Y. Tramblay" in chunks[-1]
