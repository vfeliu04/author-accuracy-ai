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

CASES = [
    # (relative path, expected text_source, (min chars, max chars), the text
    # the slice must begin with — None for a tail slice)
    #
    # The Drought report's Works Cited list runs 11 pages and repeats its
    # heading on each as a running header, the headings 5,072 to 6,064
    # characters apart (all within HEADING_RUN_GAP), so the slice starts at
    # the FIRST of the 11 headings, on the entry "Addis Standard". From that
    # heading the list runs 61,616 characters to the end of the file (57,173
    # to the last running header, then that last page); REFERENCE_MAX_CHARS
    # binds at 60,000, so the slice is exactly the cap and the last page's
    # final seven entries (World Meteorological Organization to Zkhiri) fall
    # outside it. Before the running-header rule the slice was the last page
    # alone: 4,443 characters.
    (
        "example source two/Drought Hotspots 2023-2025_ENG.pdf",
        "heading",
        (60_000, 60_000),
        "WORKS CITED\nAddis Standard. (2024, January 10). News: Four million",
    ),
    # One heading, 7,436 characters from it to the end of the file: a 7,435-character slice.
    (
        "example source two/Defining_domestic_water_consumption_based_on_perso.pdf",
        "heading",
        (7_000, 7_500),
        "REFERENCES\nAdler, I. 2011 Domestic water demand management",
    ),
    # No heading: the whole (short) document is the tail, 4,519 characters stripped.
    ("Water_Stress_Fake_Report.pdf", "tail", (4_000, 5_000), None),
    # No heading: the whole (short) document is the tail, 5,664 characters stripped.
    ("example source one/World_Hunger_Fake.pdf", "tail", (5_000, 6_000), None),
]


@pytest.mark.parametrize(("relative", "expected_source", "bounds", "starts_with"), CASES)
def test_real_reports_slice_where_measured(relative, expected_source, bounds, starts_with):
    from authorai.references import read_pages, reference_text

    path = EXAMPLES / relative
    if not path.exists():
        pytest.skip(f"example PDF not present: {path}")
    with path.open("rb") as handle:
        pages = read_pages(handle)
    assert pages, "expected at least one page"
    text, source = reference_text(pages)
    assert source == expected_source
    low, high = bounds
    assert low <= len(text) <= high, f"{relative}: {len(text)} chars from {source!r}"
    if starts_with is not None:
        assert text.startswith(starts_with), f"{relative} begins {text[:80]!r}"
