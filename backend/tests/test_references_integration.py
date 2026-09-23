"""Real-PDF check of the bibliography text slice — excluded in CI (needs the
example PDFs). Run locally with: python -m pytest -m integration -q

EVERY example PDF is swept: where each report's reference list is found was
measured on the real files, and these bounds hold the measurement so a
regression in the heading rule, the run-gap rule or the page reader is
visible on the real files — a rule that parses PDF text is judged on the
spread of real shapes, not on one report. Re-measure and re-pin when a rule
or a cap changes.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

EXAMPLES = Path(__file__).resolve().parents[2] / "example_sources"

DROUGHT = "example source two/Drought Hotspots 2023-2025_ENG.pdf"

GHI_2024 = "example source one/2024 Global Wolrd Hunger Index.pdf"
GHI_2025 = "example source two/2025.pdf"

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
    # The two Global Hunger Index reports print no heading LINE: their
    # bibliography (pages 52-58 of 66, and 52-56 of 62) carries a running
    # footer on each page — "2024 Global Hunger Index | Bibliography  51",
    # "52 Bibliography  | 2024 Global Hunger Index" — which pypdf emits at
    # the END of the page's text, after the page's entries and after the
    # section title glued to the last entry ("…175–179. BIBLIOGRAPHY", not
    # a heading). A running header names its page, so the slice starts at
    # the first footer's PAGE start: the first entry under "A". Before the
    # rule both fell to the tail: 65,000 characters starting in the
    # appendix tables (105 and 16,685 characters of them) — and the 2024
    # list, 47,005 characters plus the imprint pages after it, is cut by
    # the cap at 64,895 (the cap, stripped), the 2025 list reaches its end.
    (
        GHI_2024,
        "heading",
        (64_895, 64_895),
        "A\nAgarwal, B. 2019. “Does Group Farming Empower Rural Women?",
        None,
    ),
    (
        GHI_2025,
        "heading",
        (48_315, 48_315),
        "A\nACLED (Armed Conflict Location and Event Data). 2024.",
        None,
    ),
    # A plain heading mid-page (after the body's last paragraph), 25,085 characters to the end.
    (
        "example source one/disruptions_in_the_food_supply_chain.pdf",
        "heading",
        (25_085, 25_085),
        "References \n[1]L. Abuabara, K. Werner-Masters",
        None,
    ),
    # The page number glued BEFORE the heading ("39Literature Cited", page 46
    # of 50) is read as a numbered heading; the contents line "Literature
    # Cited 39" on page 4 is a running-header form but 42 pages back, so it
    # stays excluded. 8,148 characters to the end.
    (
        "example source two/REPORT19.pdf",
        "heading",
        (8_148, 8_148),
        "39Literature Cited\nAlcamo, Joseph, Petra Doll",
        None,
    ),
    # The page number on the line BEFORE the heading ("8 ") is not part of
    # it: the heading rule reads one line at a time, so the slice starts at
    # the heading word, 7,675 characters to the end (a pattern that ran
    # across the newline used to absorb the "8 " as a numbered heading).
    (
        "example source two/Status_Brief_C_TRN_11.pdf",
        "heading",
        (7_675, 7_675),
        "References \n \nBallesteros, C., Lincke, D.",
        None,
    ),
    # No heading LINE at all (a footnote-style report, 56 pages): the last
    # 65,000 characters of the document.
    ("example source one/HH_Nov24-May25_FINAL.pdf", "tail", (65_000, 65_000), None, None),
    # No heading: the whole (short) document is the tail.
    ("example source one/2025_world_hunger.pdf", "tail", (33_000, 33_200), None, None),
    (
        "example source two/WSG-Water-Consumption-Report-2021-FV.pdf",
        "tail",
        (2_600, 2_800),
        None,
        None,
    ),
    # No heading: the whole (short) document is the tail, 4,519 characters stripped.
    ("Water_Stress_Fake_Report.pdf", "tail", (4_000, 5_000), None, None),
    # No heading: the whole (short) document is the tail, 5,664 characters stripped.
    ("example source one/World_Hunger_Fake.pdf", "tail", (5_000, 6_000), None, None),
]


def test_every_example_pdf_is_pinned():
    """The sweep covers every example file: a new sample gets a measured
    case, never a silent miss."""
    present = {str(path.relative_to(EXAMPLES)) for path in EXAMPLES.rglob("*.pdf")}
    assert present == {relative for relative, *_ in CASES}


def _slice(relative: str):
    from authorai.references import read_pages, reference_text

    path = EXAMPLES / relative
    if not path.exists():
        pytest.skip(f"example PDF not present: {path}")
    with path.open("rb") as handle:
        pages = read_pages(handle, timeout=60)
    assert pages, "expected at least one page"
    return reference_text(pages)


@pytest.mark.parametrize(
    ("relative", "expected_source", "bounds", "starts_with", "ends_with"), CASES
)
def test_real_reports_slice_where_measured(
    relative, expected_source, bounds, starts_with, ends_with
):
    text, source, _ = _slice(relative)
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

    text, _, _ = _slice(DROUGHT)
    chunks = split_reference_text(text)
    assert len(chunks) == 6
    assert all(len(chunk) <= REFERENCE_CHUNK_CHARS for chunk in chunks)
    assert all(len(chunk) > REFERENCE_CHUNK_CHARS * 0.9 for chunk in chunks[:-1])
    lines = [line for line in text.split("\n") if line.strip()]
    assert [line for chunk in chunks for line in chunk.split("\n") if line.strip()] == lines
    assert "Zkhiri, W., Y. Tramblay" in chunks[-1]


# The completeness guard's yardstick (references.entry_starts), measured on
# every real slice: (relative path, the count of entries it must land within
# 40 % of, where that count comes from). The live model's good answers where
# they were measured (the reviewer's six calls: 37 for the article, 225 for
# the Drought list); elsewhere a count independent of the heuristic — the
# ". YYYY. " token the Global Hunger Index and REPORT19 styles print once
# per entry, the highest bracket number of the numbered list, the
# blank-line-separated blocks of the status brief. Measured 2026-09-23:
# 247, 34, 190, 117, 116, 37, 23.
ENTRY_COUNTS = [
    (DROUGHT, 225),
    ("example source two/Defining_domestic_water_consumption_based_on_perso.pdf", 37),
    (GHI_2024, 199),
    (GHI_2025, 131),
    ("example source one/disruptions_in_the_food_supply_chain.pdf", 106),
    ("example source two/REPORT19.pdf", 30),
    ("example source two/Status_Brief_C_TRN_11.pdf", 22),
]

# The reports with no reference list — the two fakes, the two short reports
# and the footnote-numbered Hunger Hotspots report, whose "199  UNHCR.
# 2024." lines the heuristic does not read (a known limit) — must stay
# under the guard's floor, so a scan of them is never asked twice or
# flagged for the empty list that is its right answer. Measured: 1, 2, 0,
# 2, 1.
NO_LIST = [
    "Water_Stress_Fake_Report.pdf",
    "example source one/World_Hunger_Fake.pdf",
    "example source two/WSG-Water-Consumption-Report-2021-FV.pdf",
    "example source one/2025_world_hunger.pdf",
    "example source one/HH_Nov24-May25_FINAL.pdf",
]


def test_every_example_pdf_is_calibrated():
    assert {relative for relative, _ in ENTRY_COUNTS} | set(NO_LIST) == {
        relative for relative, *_ in CASES
    }


@pytest.mark.parametrize(("relative", "count"), ENTRY_COUNTS)
def test_entry_starts_lands_within_forty_percent_of_the_real_lists_count(relative, count):
    from authorai.references import entry_starts

    text, _, _ = _slice(relative)
    starts = entry_starts(text)
    assert 0.6 * count <= starts <= 1.4 * count, f"{relative}: {starts} starts for {count} entries"


@pytest.mark.parametrize("relative", NO_LIST)
def test_entry_starts_stays_under_the_guards_floor_on_a_report_without_a_list(relative):
    from authorai.references import ENTRY_STARTS_FLOOR, entry_starts

    text, _, _ = _slice(relative)
    assert entry_starts(text) < ENTRY_STARTS_FLOOR, f"{relative}: {entry_starts(text)}"
