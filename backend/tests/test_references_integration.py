"""Real-PDF check of the bibliography text slice — excluded in CI (needs the
example PDFs). Run locally with: python -m pytest -m integration -q

Where each report's reference list is found was measured before the feature
was planned; these bounds hold the measurement so a regression in the
heading rule or the page reader is visible on the real files.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

EXAMPLES = Path(__file__).resolve().parents[2] / "example_sources"

CASES = [
    # (relative path, expected text_source, (min chars, max chars))
    # The Drought report's Works Cited list runs 11 pages and repeats its
    # heading on each as a running header; the slice starts at the first of
    # them and reaches the cap (before the running-header rule it read the
    # last page alone: 4,443 characters, ten pages of entries lost).
    ("example source two/Drought Hotspots 2023-2025_ENG.pdf", "heading", (29_000, 30_000)),
    (
        "example source two/Defining_domestic_water_consumption_based_on_perso.pdf",
        "heading",
        (6_000, 9_000),
    ),
    ("Water_Stress_Fake_Report.pdf", "tail", (200, 30_000)),
    ("example source one/World_Hunger_Fake.pdf", "tail", (200, 30_000)),
]


@pytest.mark.parametrize(("relative", "expected_source", "bounds"), CASES)
def test_real_reports_slice_where_measured(relative, expected_source, bounds):
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
