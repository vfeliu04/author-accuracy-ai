"""Bibliography scan (references.py): closing-page text, the reference models,
the Haiku call, the Unpaywall client and retrievability — all offline (respx
for HTTP, FakeLLM for the model, a hand-built PDF for pypdf)."""

import functools
import io
import json
import multiprocessing
import os
import re
import resource
import signal
import subprocess
import sys
import threading
import time
import types
import zlib
from concurrent.futures import wait

import httpx
import pytest
import respx

from authorai import references
from authorai.fetch import MAX_URL_LENGTH
from authorai.llm import PARSE_MAX_TOKENS
from authorai.references import (
    DOI_FIELD_MAX_CHARS,
    ENTRY_PREFIX_CHARS,
    ENTRY_STARTS_FLOOR,
    HEADING_RUN_PAGES,
    MAX_REFERENCES,
    PAGE_TEXT_MAX_CHARS,
    PAGE_TREE_CEILING,
    READER_ADDRESS_SPACE_BYTES,
    READER_MEMORY_BYTES,
    READER_MEMORY_EXIT_CODE,
    REFERENCE_CHUNK_CHARS,
    REFERENCE_MAX_CHARS,
    REFERENCES_SYSTEM,
    UNPAYWALL_BASE,
    URL_FIELD_MAX_CHARS,
    Reference,
    ReferenceList,
    RegistryUnavailable,
    UnpaywallClient,
    entry_starts,
    extract_references,
    lookup_retrievability,
    printed_url,
    read_pages,
    reference_text,
    retrievability,
    split_reference_text,
)
from authorai.web import ExtractionTimeoutError, handover_file, in_bounded_child
from tests.conftest import FakeLLM, pdf_from_objects, pdf_with_pages, reader_that_never_answers

ENTRY = "Smith, J. (2020). Water stress and cities. Journal of Hydrology, 12(3), 1-9."


# --- reference_text: where the bibliography is ---------------------------------

SECOND_ENTRY = "Jones, K. (2019). Rivers under stress. Nature Water, 1(1), 2-3."
BODY = "Body text continues. " * 200  # a page of prose with no heading


def test_slices_from_the_last_references_heading():
    """A report can print 'References' more than once (a chapter's own list
    far earlier); the bibliography starts at the LAST one when the earlier
    one lies more than HEADING_RUN_PAGES pages back — the cap plays no part
    in where the slice starts."""
    pages = [
        "Intro. See References for details.",
        "References\nChapter 1's own short list.",
        BODY,
        BODY,
        BODY,  # three pages between the two headings: more than a run
        "References\n" + ENTRY,
    ]
    text, source, _ = reference_text(pages, max_chars=100)
    assert source == "heading"
    assert text == "References\n" + ENTRY


def test_a_running_header_does_not_cut_a_multi_page_bibliography_short():
    """Each page of a long bibliography carries 'References' as a running
    header. The LAST match is on the last page; slicing from it alone would
    lose every earlier entry. Headings at most HEADING_RUN_PAGES pages apart
    are one section, so the slice starts at the earliest of them."""
    first = "References\n" + ENTRY
    second = "45\nReferences\n" + SECOND_ENTRY
    text, source, _ = reference_text([first, second])
    assert source == "heading"
    assert text == first + "\n" + second


def test_a_run_of_headers_joins_by_chaining_each_page_to_the_one_before():
    """Four consecutive pages each carry the running header. The first sits
    three pages before the last — more than HEADING_RUN_PAGES — so a rule
    measuring every heading against the LAST one would drop page 0, and a
    rule stepping back once would start at page 2. Only chaining, each
    heading judged against the one before it, starts at page 0 and keeps
    all four pages' entries."""
    pages = [f"References\nEntry on page {i}. " + ENTRY for i in range(4)]
    text, source, _ = reference_text(pages)
    assert source == "heading"
    assert text == "\n".join(pages)
    assert all(f"Entry on page {i}." in text for i in range(4))


def test_a_chapter_list_several_pages_back_stays_excluded():
    """A chapter-end reference list four pages before the closing bibliography
    is a separate section: the slice starts at the LAST heading and keeps
    every one of the final list's 120 entries, with the chapter list out.
    The pages between are short, so a rule measured in characters would
    have joined the two lists — page distance does not depend on how dense
    the pages are."""
    chapter_list = "References\n" + SECOND_ENTRY
    final_entries = [f"Final ref {i}. " + ENTRY for i in range(120)]
    final_list = "References\n" + "\n".join(final_entries)
    sparse_page = "A short page.\n"
    text, source, _ = reference_text(
        [chapter_list, sparse_page, sparse_page, sparse_page, final_list]
    )
    assert source == "heading"
    assert text == final_list
    assert all(entry in text for entry in final_entries)
    assert SECOND_ENTRY not in text


def _pages_with_headings_apart(pages_apart: int) -> list[str]:
    """Pages whose two headings sit exactly `pages_apart` pages apart: one on
    the first page, the other on page `pages_apart`, short pages between."""
    return ["References\n" + SECOND_ENTRY, *["Body."] * (pages_apart - 1), "References\n" + ENTRY]


def test_headings_exactly_one_run_apart_are_one_section():
    """The boundary is inclusive: headings HEADING_RUN_PAGES pages apart (a
    header printed on alternate pages — recto or verso) join, so the slice
    starts at the earlier heading. A mutant that compares with "<" instead
    of "<=" fails here."""
    pages = _pages_with_headings_apart(HEADING_RUN_PAGES)
    text, source, _ = reference_text(pages)
    assert source == "heading"
    assert text == "\n".join(pages)


def test_headings_one_page_more_than_a_run_apart_are_two_sections():
    """One page past the run and the earlier heading is a separate section:
    the slice starts at the last heading."""
    pages = _pages_with_headings_apart(HEADING_RUN_PAGES + 1)
    text, source, _ = reference_text(pages)
    assert source == "heading"
    assert text == "References\n" + ENTRY


def test_a_heading_that_begins_a_page_belongs_to_that_page():
    """The joined text puts one newline between pages, and each page's start
    is counted with it: a heading at the very first character of page 3 is
    on page 3, not page 2. Observable through the run rule — the earlier
    heading is on page 0, so the two are HEADING_RUN_PAGES + 1 apart, two
    sections, and the slice starts at the later one. A page start counted
    one character late, or a bisect that puts a position equal to a page
    start on the page before, reads them as one run and keeps page 0."""
    assert HEADING_RUN_PAGES + 1 == 3
    pages = ["Body.\nReferences\n" + SECOND_ENTRY, "Body.", "Body.", "References\n" + ENTRY]
    text, source, _ = reference_text(pages)
    assert source == "heading"
    assert text == "References\n" + ENTRY


def test_a_heading_that_ends_a_page_belongs_to_that_page():
    """The other side of the same count: a heading printed at the foot of a
    page (its entries begin on the next page, and pypdf ends a page's text
    without a newline) sits in that page's last characters. With the join
    newline left out of the count every recorded page start drifts one
    character earlier per page, so by page 12 the page's last twelve
    characters — the whole heading — would read as page 13: three pages
    after the running header on page 10 instead of two, the run would
    break, and the header's page would be dropped."""
    pages = [
        *[BODY] * 10,
        "References\n" + SECOND_ENTRY,  # page 10: the running header
        "Entries continue.",
        BODY + "\nReferences",  # page 12: the heading at the foot, HEADING_RUN_PAGES later
        ENTRY,
    ]
    text, source, _ = reference_text(pages)
    assert source == "heading"
    assert text == "\n".join(pages[10:])


def test_a_contents_page_mention_far_earlier_stays_excluded():
    """A table of contents prints 'References' on a line of its own, many
    pages before the bibliography: more than HEADING_RUN_PAGES pages back
    from the last heading, so the slice still starts at the bibliography."""
    pages = ["Contents\nReferences\n45", *[BODY] * 10, "References\n" + ENTRY]
    text, source, _ = reference_text(pages)
    assert source == "heading"
    assert text == "References\n" + ENTRY


def test_the_slice_and_its_cap_start_at_the_heading_word():
    """Blank lines and indentation before the heading are matched by the
    pattern but are not bibliography: they do not count against the cap."""
    text, source, _ = reference_text(["Body", "\n\n   References\n" + "x" * 100], max_chars=20)
    assert source == "heading"
    assert text == ("References\n" + "x" * 100)[:20]


# Lines exactly as pypdf prints them from the example reports (file, page
# index in the PDF), so the rule is judged on real running headers, contents
# lines and page numbers, not on hand-typed headings alone.
REAL_HEADING_LINES = [
    "BIBLIOGRAPHY    51",  # 2024 Global Wolrd Hunger Index.pdf p5: the contents line
    "Literature Cited 39",  # REPORT19.pdf p4: the contents line
    "39Literature Cited",  # REPORT19.pdf p46: the page number glued before the heading
    "2024 Global Hunger Index | Bibliography  51",  # 2024 GHI p52: a running footer
    "52 Bibliography  | 2024 Global Hunger Index",  # 2024 GHI p53: the verso footer
    "2025 Global Hunger Index | Bibliography  51",  # example source two/2025.pdf p52
    "References ",  # disruptions_in_the_food_supply_chain.pdf p18
    "WORKS CITED",  # Drought Hotspots 2023-2025_ENG.pdf p38
]
REAL_NON_HEADING_LINES = [
    # 2024 GHI p52: the section title glued to the page's last entry — the
    # shape of a sentence-end mention, so not a heading; the page is found
    # through its footer (test_a_running_footer_anchors_the_slice_at_its_page_start)
    "Nutrition  22 (1): 175–179. BIBLIOGRAPHY",
    "the-state-of-food-security-and-nutrition-in-the-world/en .BIBLIOGRAPHY",  # 2025.pdf p52
    "Source: Authors, based on sources listed in Appendix A and previous GHI "
    "publications included in the bibliography.",  # 2024 GHI p43
    "Works Cited  ...............................  40TABLE OF CONTENTS",  # Drought p3
    "Demand Consumer preferences [51,89] ",  # disruptions p6: "references" inside a word
    "Intro. See References for details.",
]


@pytest.mark.parametrize(
    "heading",
    [
        "References",
        "REFERENCES",
        "7. References",
        "7 References",
        "Bibliography",
        "Works Cited",
        "Reference list:",
        "Literature cited",
        "   References   ",
        *REAL_HEADING_LINES,
    ],
)
def test_every_heading_form_is_recognised(heading):
    text, source, _ = reference_text(["Body text.", heading + "\n" + ENTRY])
    assert source == "heading"
    assert text.startswith(heading.strip())
    assert text.endswith(ENTRY)


@pytest.mark.parametrize("line", REAL_NON_HEADING_LINES)
def test_a_mid_line_mention_is_not_a_heading(line):
    """Only a line that IS the heading counts — 'see References' in prose,
    a title glued to a sentence's end, a contents leader — is not where the
    bibliography starts."""
    text, source, _ = reference_text(["Body text.", line + "\n" + ENTRY])
    assert source == "tail"


FOOTER_RECTO = "2024 Global Hunger Index | Bibliography  51"
FOOTER_VERSO = "52 Bibliography  | 2024 Global Hunger Index"


def test_a_running_footer_anchors_the_slice_at_its_page_start():
    """The GHI reports print no heading line: the bibliography's pages carry
    a running footer, which pypdf emits at the END of each page's text. A
    heading with a page number or a running-header decoration names the
    PAGE, so the slice starts where that page starts — at its entries, not
    after them — and the run of footers reaches back to the first page."""
    first = ENTRY + "\nNutrition  22 (1): 175–179. BIBLIOGRAPHY\n" + FOOTER_RECTO
    second = SECOND_ENTRY + "\n" + FOOTER_VERSO
    text, source, _ = reference_text([BODY, first, second])
    assert source == "heading"
    assert text.startswith(ENTRY)
    assert text.endswith(FOOTER_VERSO)
    assert BODY not in text


def test_a_plain_heading_still_anchors_at_the_heading_word():
    """A heading line without a page number or decoration is where the list
    starts, whatever precedes it on the page."""
    text, source, _ = reference_text([BODY, "Conclusions end here.\nReferences\n" + ENTRY])
    assert source == "heading"
    assert text == "References\n" + ENTRY


def test_a_contents_line_with_a_page_number_far_from_the_list_stays_excluded():
    pages = ["Contents\nBIBLIOGRAPHY    51", BODY, BODY, BODY, ENTRY + "\n" + FOOTER_RECTO]
    text, source, _ = reference_text(pages)
    assert source == "heading"
    assert text == ENTRY + "\n" + FOOTER_RECTO


def test_no_heading_falls_back_to_the_document_tail():
    text, source, _ = reference_text(["A" * 100, "B" * 100], max_chars=50)
    assert source == "tail"
    assert text == "B" * 50


def test_the_heading_slice_is_capped_from_the_heading_forward():
    text, source, _ = reference_text(["References\n" + "x" * 100], max_chars=20)
    assert source == "heading"
    assert text == ("References\n" + "x" * 100)[:20]


def test_the_default_cap_is_the_code_constant():
    text, source, _ = reference_text(["y" * (REFERENCE_MAX_CHARS + 500)])
    assert source == "tail"
    assert len(text) == REFERENCE_MAX_CHARS


def test_no_text_at_all_means_no_model_call():
    """A scanned PDF yields no text; the answer is 'none', never an invented
    bibliography — the endpoint skips the model on this value."""
    assert reference_text([]) == ("", "none", False)
    assert reference_text(["", "  \n\t"]) == ("", "none", False)


def test_the_closing_text_says_when_the_cap_cut_it():
    """The dialog presents the list as the whole reference list; a cap that
    cut the text is reported, not only logged, so it can say otherwise."""
    assert reference_text(["References\n" + "x" * 100], max_chars=20).truncated is True
    assert reference_text(["References\n" + "x" * 100], max_chars=200).truncated is False
    assert reference_text(["A" * 100, "B" * 100], max_chars=50).truncated is True
    assert reference_text(["A" * 100, "B" * 100], max_chars=500).truncated is False
    assert reference_text(["References\n" + "x" * 9], max_chars=20).truncated is False


# Hostile whitespace: the scan is one pass over each line — the pattern is
# tested per line, with possessive quantifiers and only " " and "\t" as
# space, never \s, which runs across newlines — so no run of spaces or "|"
# can make it re-read what it already read. Measured before the rule: a
# 100,000-space line took 33 seconds in the request thread, AFTER the
# bounded child had returned the pages.
@pytest.mark.parametrize(
    ("page", "expected_source", "expected_start"),
    [
        (" " * 100_000, "tail", None),
        (" " * 100_000 + "References", "heading", "References"),
        ("\n".join(["|"] * 10_000), "tail", None),
        (" |" * 32_500, "tail", None),
    ],
    ids=["100k spaces", "100k spaces then References", "10k lines of |", "65k of space and |"],
)
def test_the_heading_scan_is_linear_over_hostile_whitespace(page, expected_source, expected_start):
    started = time.perf_counter()
    text, source, _ = reference_text(["Body text.", page])
    elapsed = time.perf_counter() - started
    assert elapsed < 0.1, f"the heading scan took {elapsed:.2f} s"
    assert source == expected_source
    if expected_start is not None:
        assert text.startswith(expected_start)


# --- read_pages: pypdf over the spooled upload ---------------------------------


def test_read_pages_extracts_the_text_of_each_page():
    pdf = pdf_with_pages(["Page one", "Page two", "References\n" + ENTRY])
    pages = read_pages(io.BytesIO(pdf), timeout=30)
    assert [page.strip() for page in pages] == ["Page one", "Page two", "References\n" + ENTRY]


def test_read_pages_reads_only_the_last_max_pages():
    pdf = pdf_with_pages(["one", "two", "three"])
    assert [p.strip() for p in read_pages(io.BytesIO(pdf), timeout=30, max_pages=2)] == [
        "two",
        "three",
    ]


def test_read_pages_keeps_a_textless_page_as_an_empty_string():
    assert [p.strip() for p in read_pages(io.BytesIO(pdf_with_pages([""])), timeout=30)] == [""]


@pytest.mark.parametrize(
    "junk",
    [b"", b"not a pdf", b"%PDF-1.4 fake pdf content", b"%PDF-1.7\n1 0 obj\n<< >>\nendobj\n"],
)
def test_read_pages_turns_any_pypdf_failure_into_a_value_error(junk):
    """The endpoint answers 400 for what it cannot read; pypdf's own exception
    family is not its contract."""
    with pytest.raises(ValueError, match="could not read"):
        read_pages(io.BytesIO(junk), timeout=30)


def test_read_pages_opens_a_pdf_encrypted_with_an_empty_password():
    """Publishers often encrypt a PDF with an empty user password (print/copy
    restrictions) — such a file must be read, not refused."""
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(pdf_with_pages(["Restricted references"]))))
    writer.encrypt(user_password="", owner_password="owner-only")
    encrypted = io.BytesIO()
    writer.write(encrypted)
    assert PdfReader(io.BytesIO(encrypted.getvalue())).is_encrypted
    assert [p.strip() for p in read_pages(encrypted, timeout=30)] == ["Restricted references"]


def _encrypted(pages: list[str]) -> bytes:
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(pdf_with_pages(pages))))
    writer.encrypt(user_password="", owner_password="owner-only")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


@pytest.mark.parametrize("declared", [b"/Count 2", b"/Count 9"])
def test_an_encrypted_pdf_is_paged_by_its_real_page_list_not_the_declared_count(declared):
    """pypdf's page list is a view over the trailer's /Count for an
    encrypted file (it flattens the real tree only for an unencrypted one),
    so a stale count silently dropped the closing pages — where the
    bibliography lives — or made a file every viewer opens unreadable. The
    count is a claim; the pages are what the tree holds."""
    encrypted = _encrypted(["one", "two", "three"])
    assert encrypted.count(b"/Count 3") == 1
    patched = encrypted.replace(b"/Count 3", declared)  # same length: the xref still holds
    pages = read_pages(io.BytesIO(patched), timeout=30)
    assert [page.strip() for page in pages] == ["one", "two", "three"]


# --- read_pages: a bounded child, and a page tree bounded before it is walked --


def _doubled_page_tree(depth: int, *, count: int) -> bytes:
    """The hostile page tree the security review built: every interior
    /Pages node lists ONE child twice, so a flatten with no visited set
    (pypdf 3.17.4's) yields 2**depth pages from depth + 3 objects — 1,024
    pages from a 1,149-byte file at depth 10, a billion at depth 30. `count`
    is what the root DECLARES, which such a file may set to anything."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    for level in range(depth):
        number = 2 + level
        child = number + 1
        declared = f" /Count {count}" if level == 0 else f" /Parent {number - 1} 0 R"
        objects.append(f"<< /Type /Pages /Kids [{child} 0 R {child} 0 R]{declared} >>".encode())
    leaf = 2 + depth
    objects.append(f"<< /Type /Page /Parent {leaf - 1} 0 R /MediaBox [0 0 612 792] >>".encode())
    return pdf_from_objects(objects)


def _flat_page_tree(leaves: int, *, count: int) -> bytes:
    """`leaves` textless pages under one root that declares `count` pages."""
    kids = " ".join(f"{3 + index} 0 R" for index in range(leaves))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode(),
        *[b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>"] * leaves,
    ]
    return pdf_from_objects(objects)


def _stream(data: bytes, *, deflate: bool) -> bytes:
    """A stream object body, deflated when asked (as real PDFs store them)."""
    if deflate:
        data = zlib.compress(data, 9)
        head = b"<< /Length " + str(len(data)).encode() + b" /Filter /FlateDecode >>"
    else:
        head = b"<< /Length " + str(len(data)).encode() + b" >>"
    return head + b"\nstream\n" + data + b"\nendstream"


def _cmap_pdf(
    bfchars: str,
    content: bytes = b"BT /F1 12 Tf 72 720 Td <000100020002> Tj ET",
    *,
    pages: int = 1,
    deflate: bool = False,
) -> bytes:
    """`pages` pages, each drawing `content`, set in a Type0 font whose
    ToUnicode CMap maps glyphs as `bfchars` says — the shape of a real
    malformed font, which pypdf decodes with `surrogatepass` (strict=False)
    rather than refuse."""
    cmap = (
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
        "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        f"{bfchars}\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
    ).encode()
    kids = " ".join(f"{7 + index} 0 R" for index in range(pages))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode(),
        b"<< /Type /Font /Subtype /Type0 /BaseFont /Fake /Encoding /Identity-H "
        b"/DescendantFonts [4 0 R] /ToUnicode 5 0 R >>",
        b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Fake /CIDSystemInfo "
        b"<< /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /DW 500 >>",
        _stream(cmap, deflate=deflate),
        _stream(content, deflate=deflate),
        *[
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents 6 0 R >>"
        ]
        * pages,
    ]
    return pdf_from_objects(objects)


def _amplifier_pdf(destination_chars: int, uses: int, *, pages: int = 1) -> bytes:
    """The security review's text amplifier: one glyph code whose ToUnicode
    destination is `destination_chars` characters ("A"), drawn `uses` times
    on each of `pages` pages, streams deflated — a file of ~1.3 KB that
    pypdf expands to destination_chars × uses characters a page, by dict
    lookup, at memcpy speed."""
    return _cmap_pdf(
        f"1 beginbfchar\n<0001> <{'0041' * destination_chars}>\nendbfchar",
        b"BT /F1 12 Tf 72 720 Td <" + b"0001" * uses + b"> Tj ET",
        pages=pages,
        deflate=True,
    )


def _resident_bytes() -> int:
    """This process's CURRENT resident size (`ps`, on macOS and Linux alike):
    ru_maxrss is a peak, which an earlier test may already have raised."""
    kilobytes = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True
    ).stdout
    return int(kilobytes.strip()) * 1024


def test_a_lone_surrogate_from_a_broken_font_is_replaced_so_the_text_can_be_sent():
    """A glyph mapped to an unpaired surrogate (<D800>) reaches the model
    call as a str the SDK cannot encode as UTF-8 — a 500 raised before any
    request. The reader replaces it; a valid pair (an emoji) is untouched."""
    lone = _cmap_pdf("2 beginbfchar\n<0001> <D800>\n<0002> <0041>\nendbfchar")
    (page,) = read_pages(io.BytesIO(lone), timeout=30)
    assert page.strip() == "?AA"
    page.encode("utf-8")  # what the SDK does with the prompt
    pair = _cmap_pdf("1 beginbfchar\n<0001> <D83DDE00>\n<0002> <0041>\nendbfchar")
    (page,) = read_pages(io.BytesIO(pair), timeout=30)
    assert page.strip() == "\U0001f600AA"


def _reader_the_kernel_ends(sender, path, *_args, signum: int):
    """A reader the kernel ends by signal: the CPU limit's SIGXCPU, an
    out-of-memory kill."""
    os.kill(os.getpid(), signum)
    threading.Event().wait(60)  # the signal lands; nothing here should run


def _reader_that_cannot_import(sender, path, *_args):
    """A child whose start-up fails before any read — a module edited on
    disk under a running server, a missing dependency."""
    raise ImportError("No module named 'pypdf'")


def _reader_that_returns_nothing(sender, path, *_args):
    """A reader that returns without reporting: a bug, not a bound."""


def test_read_pages_reads_in_a_child_that_is_gone_when_it_answers(web_log):
    before = set(multiprocessing.active_children())
    pdf = pdf_with_pages(["Page one", "References\n" + ENTRY])
    pages = read_pages(io.BytesIO(pdf), timeout=30)
    assert [page.strip() for page in pages] == ["Page one", "References\n" + ENTRY]
    assert set(multiprocessing.active_children()) == before
    assert "exited without a result" not in web_log.text  # the watchdog let it finish


@pytest.mark.parametrize(("depth", "declared"), [(10, 1024), (30, 1)])
def test_a_page_tree_that_doubles_at_every_level_is_refused_before_it_is_flattened(depth, declared):
    """Refused by the walk, whatever the root declares: at depth 30 the
    flatten would never finish, and the deadline would then read the file as
    too costly — this is the refusal itself, in the time a child takes."""
    started = time.perf_counter()
    with pytest.raises(ValueError, match="could not read the PDF: .*page tree is not a tree"):
        read_pages(io.BytesIO(_doubled_page_tree(depth, count=declared)), timeout=30)
    assert time.perf_counter() - started < 10  # the child's start-up, not the tree


def test_a_declared_page_count_over_the_ceiling_is_refused_before_the_tree_is_walked():
    pdf = _flat_page_tree(2, count=PAGE_TREE_CEILING + 1)
    with pytest.raises(ValueError, match=f"declares {PAGE_TREE_CEILING + 1} pages"):
        read_pages(io.BytesIO(pdf), timeout=30)


def test_a_page_tree_with_more_nodes_than_the_ceiling_is_refused_whatever_it_declares():
    """The declared count is a claim; the walk is the bound."""
    pdf = _flat_page_tree(PAGE_TREE_CEILING + 1, count=1)
    with pytest.raises(ValueError, match=f"more than {PAGE_TREE_CEILING} nodes"):
        read_pages(io.BytesIO(pdf), timeout=30)


def test_a_page_tree_at_the_ceiling_still_reads_its_last_pages():
    leaves = PAGE_TREE_CEILING - 1  # plus the root: exactly the ceiling
    pages = read_pages(io.BytesIO(_flat_page_tree(leaves, count=leaves)), timeout=30)
    assert pages == [""] * references.REFERENCE_MAX_PAGES


def test_a_reader_that_overruns_its_budget_is_stopped_and_reads_as_too_costly(monkeypatch):
    monkeypatch.setattr(references, "_read_in_child", reader_that_never_answers)
    before = set(multiprocessing.active_children())
    started = time.perf_counter()
    with pytest.raises(ValueError, match="could not read the PDF: it was too costly to read") as c:
        read_pages(io.BytesIO(pdf_with_pages(["x"])), timeout=0.5)
    assert "0.5 seconds" in str(c.value)
    assert time.perf_counter() - started < 15  # stopped at the deadline, never awaited
    assert set(multiprocessing.active_children()) == before


@pytest.mark.parametrize(
    "signum", [signal.SIGKILL, signal.SIGXCPU], ids=["out-of-memory kill", "CPU limit"]
)
def test_a_reader_the_kernel_ends_reads_as_too_costly(monkeypatch, web_log, signum):
    reader = functools.partial(_reader_the_kernel_ends, signum=int(signum))
    monkeypatch.setattr(references, "_read_in_child", reader)
    with pytest.raises(ValueError, match="too costly to read") as caught:
        read_pages(io.BytesIO(pdf_with_pages(["x"])), timeout=30)
    assert "the reader was stopped" in str(caught.value)
    assert f"exited without a result (killed by signal {signum.name})" in web_log.text


@pytest.mark.parametrize(
    ("reader", "status"),
    [(_reader_that_cannot_import, "exit code 1"), (_reader_that_returns_nothing, "exit code 0")],
    ids=["an import error", "no result"],
)
def test_a_reader_that_fails_on_its_own_is_the_servers_fault_not_the_files(
    monkeypatch, web_log, capfd, reader, status
):
    """A child that ends by no bound of its own — an unhandled exception at
    start-up, a return with nothing reported — is a bug: the RuntimeError
    propagates (the endpoint's 500) rather than the 400 that blames the
    file, with the exit status in the log and, for the exception, the
    child's own traceback on stderr."""
    monkeypatch.setattr(references, "_read_in_child", reader)
    with pytest.raises(RuntimeError) as caught:
        read_pages(io.BytesIO(pdf_with_pages(["x"])), timeout=30)
    assert not isinstance(caught.value, ValueError)
    assert str(caught.value) == (
        "the report PDF could not be read: the reader process exited without a result"
    )
    assert f"exited without a result ({status})" in web_log.text
    if reader is _reader_that_cannot_import:
        assert "ImportError: No module named 'pypdf'" in capfd.readouterr().err


def test_the_reader_caps_its_address_space_without_loosening_an_inherited_cap(monkeypatch):
    """Linux enforces RLIMIT_AS; the reader asks for the tighter of its own
    cap and the one it inherited, and a platform that refuses the request
    (macOS answers EINVAL, a ValueError from the resource module) leaves the
    wall-clock and CPU bounds as the reader's limits — never a failed read."""
    calls = []
    monkeypatch.setattr(resource, "getrlimit", lambda which: (2**29, resource.RLIM_INFINITY))
    monkeypatch.setattr(resource, "setrlimit", lambda which, limits: calls.append((which, limits)))
    references._limit_address_space(READER_ADDRESS_SPACE_BYTES)
    assert READER_ADDRESS_SPACE_BYTES > 2**29
    assert calls == [(resource.RLIMIT_AS, (2**29, resource.RLIM_INFINITY))]
    calls.clear()
    monkeypatch.setattr(
        resource, "getrlimit", lambda which: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )
    references._limit_address_space(READER_ADDRESS_SPACE_BYTES)
    assert calls == [(resource.RLIMIT_AS, (READER_ADDRESS_SPACE_BYTES, resource.RLIM_INFINITY))]

    def refused(which, limits):
        raise ValueError("current limit exceeds maximum limit")

    monkeypatch.setattr(resource, "setrlimit", refused)
    references._limit_address_space(READER_ADDRESS_SPACE_BYTES)  # tolerated


def test_the_address_space_cap_is_set_only_when_it_leaves_the_watchdog_its_headroom(
    monkeypatch, references_log
):
    """RLIMIT_AS counts every mapping — the interpreter, the native libraries
    pypdf's imports pull in, thread stacks, glibc's per-thread arenas — so a
    Linux child maps hundreds of MB before pypdf runs, and a fixed 1 GiB
    cap would fall on a legitimate report before the watchdog (which reads
    RESIDENT memory) could. The watchdog is the memory bound; the cap is a
    backstop set only when what the process maps now plus the watchdog's
    bound fits under it, so it can never fire first. Where the usage cannot
    be read (macOS has no /proc) it is asked for as before, and refused."""
    calls: list = []
    monkeypatch.setattr(
        resource, "getrlimit", lambda which: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )
    monkeypatch.setattr(resource, "setrlimit", lambda which, limits: calls.append(limits))
    monkeypatch.setattr(references, "address_space_in_use", lambda: None)
    references._limit_address_space(READER_ADDRESS_SPACE_BYTES)
    assert calls == [(READER_ADDRESS_SPACE_BYTES, resource.RLIM_INFINITY)]
    calls.clear()
    fits = READER_ADDRESS_SPACE_BYTES - READER_MEMORY_BYTES
    monkeypatch.setattr(references, "address_space_in_use", lambda: fits)
    references._limit_address_space(READER_ADDRESS_SPACE_BYTES)
    assert calls == [(READER_ADDRESS_SPACE_BYTES, resource.RLIM_INFINITY)]
    calls.clear()
    monkeypatch.setattr(references, "address_space_in_use", lambda: fits + 1)
    references._limit_address_space(READER_ADDRESS_SPACE_BYTES)
    assert calls == []
    assert "the watchdog is the memory bound" in references_log.text


def test_address_space_in_use_reads_vmsize_from_the_process_status(tmp_path):
    status = tmp_path / "status"
    status.write_text(
        "Name:\tpython\nVmPeak:\t  900000 kB\nVmSize:\t  123456 kB\nVmRSS:\t 1000 kB\n"
    )
    assert references.address_space_in_use(status) == 123456 * 1024
    assert references.address_space_in_use(tmp_path / "missing") is None  # macOS: no /proc
    (tmp_path / "bare").write_text("Name:\tpython\n")
    assert references.address_space_in_use(tmp_path / "bare") is None


# --- the memory watchdog: the bound RLIMIT_AS cannot give on macOS ------------


def _reader_that_inflates(sender, path, max_pages, cpu_seconds):
    """The real reader with pypdf's part replaced by what a FlateDecode bomb
    does to it: 900 MB resident, and KEPT resident — rewritten page by page,
    as a reader works over inflated data; an idle buffer is compressed away
    under memory pressure on macOS, and no peak would be seen — until the
    watchdog ends the process. The address-space cap is lifted so the
    allocation succeeds on Linux too and the watchdog is what stops it."""
    references.READER_ADDRESS_SPACE_BYTES = 8 << 30

    def inflate(handle, max_pages):
        view = memoryview(bytearray(b"\xa5") * (900 * 2**20))
        while True:
            for offset in range(0, len(view), 16384):
                view[offset] ^= 1

    references._last_pages = inflate
    references._read_in_child(sender, path, max_pages, cpu_seconds)


def _reader_that_hits_the_address_space_cap(sender, path, max_pages, cpu_seconds):
    """The real reader where pypdf's allocation fails: RLIMIT_AS on Linux."""

    def refuse(handle, max_pages):
        raise MemoryError

    references._last_pages = refuse
    references._read_in_child(sender, path, max_pages, cpu_seconds)


def test_a_reader_that_inflates_past_the_memory_bound_is_stopped_by_the_watchdog(
    monkeypatch, web_log
):
    monkeypatch.setattr(references, "_read_in_child", _reader_that_inflates)
    before = set(multiprocessing.active_children())
    started = time.perf_counter()
    with pytest.raises(ValueError, match="too costly to read") as caught:
        read_pages(io.BytesIO(pdf_with_pages(["x"])), timeout=30)
    assert time.perf_counter() - started < 15  # the watchdog, not the deadline
    assert "the reader needed too much memory" in str(caught.value)
    assert f"exited without a result (exit code {READER_MEMORY_EXIT_CODE})" in web_log.text
    assert set(multiprocessing.active_children()) == before


def test_a_reader_that_hits_the_address_space_cap_reads_as_too_costly(monkeypatch, web_log):
    monkeypatch.setattr(references, "_read_in_child", _reader_that_hits_the_address_space_cap)
    with pytest.raises(ValueError, match="too costly to read") as caught:
        read_pages(io.BytesIO(pdf_with_pages(["x"])), timeout=30)
    assert "the reader needed too much memory" in str(caught.value)
    assert f"exited without a result (exit code {READER_MEMORY_EXIT_CODE})" in web_log.text


# --- the reader's result is bounded on both sides of the pipe ------------------

# The parent's growth on receiving a hostile file's pages, pinned generously:
# the child sends at most two pages of PAGE_TEXT_MAX_CHARS below (~260 KB),
# and the parent's own work on them is small. Before the caps it grew by
# ~440 MiB (the 240 MB pickle, the unpickled list and the join at once).
_PARENT_GROWTH_BOUND = 64 * 2**20


def test_a_page_denser_than_the_page_cap_reaches_the_parent_cut():
    """The security review's amplifier, deterministic: 240 MB of text from a
    1.5 KB file — two pages of 120 M characters each, which peak at ~510
    MiB in the child, under its watchdog — comes back as two pages of
    PAGE_TEXT_MAX_CHARS. The child cuts each page before it is sent, so the
    parent never holds the expansion: in seconds, with its own memory
    untouched."""
    pdf = _amplifier_pdf(30_000, 4_000, pages=2)
    assert len(pdf) < 2_000
    resident = _resident_bytes()
    started = time.perf_counter()
    pages = read_pages(io.BytesIO(pdf), timeout=60)
    assert time.perf_counter() - started < 15
    assert [len(page) for page in pages] == [PAGE_TEXT_MAX_CHARS] * 2
    assert set(pages[0]) == set(pages[1]) == {"A"}
    assert _resident_bytes() - resident < _PARENT_GROWTH_BOUND


def test_the_reviews_amplifier_is_bounded_whichever_bound_it_meets():
    """The review's exact shape: 30,000 × 8,000, 240 MB on ONE page, which
    pypdf builds at 620-880 MiB in the child — either side of the 768 MiB
    watchdog from one run to the next. Whichever bound it meets, the parent
    sees the capped page or the memory bound's refusal (the 400 path),
    never a 500, in seconds, with its own memory untouched."""
    pdf = _amplifier_pdf(30_000, 8_000)
    assert len(pdf) < 1_500
    resident = _resident_bytes()
    started = time.perf_counter()
    try:
        (page,) = read_pages(io.BytesIO(pdf), timeout=60)
    except ValueError as exc:
        assert "too costly to read (the reader needed too much memory)" in str(exc)
    else:
        assert len(page) == PAGE_TEXT_MAX_CHARS and set(page) == {"A"}
    assert time.perf_counter() - started < 15
    assert _resident_bytes() - resident < _PARENT_GROWTH_BOUND


def test_the_reader_keeps_the_last_pages_that_fit_the_total_cap(monkeypatch):
    """Past READER_MAX_TEXT_CHARS the reader stops and says so: the pages
    kept are the LAST ones — what reference_text wants — read from the end
    backwards, each whole, and a page that would not fit ends the walk.
    Under the cap nothing is cut."""
    pdf = pdf_with_pages(["a" * 40, "b" * 40, "c" * 40, "d" * 40])
    pages, cut = references._last_pages(io.BytesIO(pdf), 600)
    assert (cut, [page[0] for page in pages]) == (False, ["a", "b", "c", "d"])
    lengths = [len(page) for page in pages]
    monkeypatch.setattr(references, "READER_MAX_TEXT_CHARS", sum(lengths[2:]) + lengths[1] // 2)
    pages, cut = references._last_pages(io.BytesIO(pdf), 600)
    assert (cut, [page[0] for page in pages]) == (True, ["c", "d"])
    assert [len(page) for page in pages] == lengths[2:]
    monkeypatch.setattr(references, "PAGE_TEXT_MAX_CHARS", 5)
    monkeypatch.setattr(references, "READER_MAX_TEXT_CHARS", 12)
    pages, cut = references._last_pages(io.BytesIO(pdf), 600)
    assert (pages, cut) == (["ccccc", "ddddd"], True)


def _reader_with_a_small_total_cap(sender, path, max_pages, cpu_seconds):
    """The real reader under a total cap of 8 characters — one short page
    with its line break — set in the child's own module, since the parent's
    monkeypatch does not reach a spawned process."""
    references.READER_MAX_TEXT_CHARS = 8
    references._read_in_child(sender, path, max_pages, cpu_seconds)


def test_a_read_cut_at_the_total_cap_hands_the_parent_the_last_pages_with_a_warning(
    monkeypatch, references_log
):
    monkeypatch.setattr(references, "_read_in_child", _reader_with_a_small_total_cap)
    pdf = pdf_with_pages(["one", "two", "three", "four"])
    pages = read_pages(io.BytesIO(pdf), timeout=30)
    assert [page.strip() for page in pages] == ["four"]
    assert "WARNING" in references_log.text
    assert "read in part" in references_log.text


def _reader_that_reports_too_much(sender, path, *_args):
    """A reader whose result is larger than the parent will receive."""
    from authorai.web import report_outcome

    report_outcome(sender, lambda: "x" * (2 * 2**20), lambda exc: "other")


def test_a_result_the_parent_refuses_on_its_size_reads_as_too_costly(monkeypatch, web_log):
    """The parent's side of the bound: a frame past RESULT_MAX_BYTES is
    refused on its length header, unread, and the refusal is a bound of the
    reader's — the 400, never the 500 a reader that dies mid-report is."""
    import authorai.web as web_mod

    monkeypatch.setattr(web_mod, "RESULT_MAX_BYTES", 2**20)
    monkeypatch.setattr(references, "_read_in_child", _reader_that_reports_too_much)
    before = set(multiprocessing.active_children())
    with pytest.raises(ValueError, match="too costly to read") as caught:
        read_pages(io.BytesIO(pdf_with_pages(["x"])), timeout=30)
    assert "the reader's result was too large" in str(caught.value)
    assert f"reported a result larger than {2**20} bytes" in web_log.text
    assert "exited without a result" not in web_log.text
    assert set(multiprocessing.active_children()) == before


def test_the_watchdog_ends_the_process_only_once_the_peak_passes_the_bound(monkeypatch):
    """In this process, with the measurement and the exit replaced: no exit
    at the bound, the exit with the reader's own code past it, and a daemon
    thread — one that never holds the child open after a clean read."""
    stop = threading.Event()
    peak = {"bytes": READER_MEMORY_BYTES}
    exits: list[int] = []
    monkeypatch.setattr(references, "peak_resident_bytes", lambda: peak["bytes"])
    monkeypatch.setattr(references.os, "_exit", lambda code: (exits.append(code), stop.set()))
    thread = references._watch_memory(interval=0.01, stop=stop)
    try:
        assert thread.daemon
        time.sleep(0.1)
        assert exits == []
        peak["bytes"] = READER_MEMORY_BYTES + 1
        thread.join(2)
        assert not thread.is_alive()
        assert exits == [READER_MEMORY_EXIT_CODE]
    finally:
        stop.set()
        thread.join(2)


def test_peak_resident_bytes_reads_ru_maxrss_in_each_platforms_unit(monkeypatch):
    monkeypatch.setattr(resource, "getrusage", lambda who: types.SimpleNamespace(ru_maxrss=1024))
    monkeypatch.setattr(sys, "platform", "darwin")
    assert references.peak_resident_bytes() == 1024  # bytes
    monkeypatch.setattr(sys, "platform", "linux")
    assert references.peak_resident_bytes() == 1024 * 1024  # kilobytes


def _reader_that_finishes_with_the_watchdog_running(sender, path, *_args):
    """A reader whose read is done: it returns with the watchdog still up."""
    references._watch_memory()


def test_the_watchdog_does_not_hold_the_reader_open_after_a_clean_finish(web_log):
    """The child's interpreter waits for its non-daemon threads at exit: a
    watchdog that were one would hold the process open — forever, it loops
    — until the deadline stopped it. The parent sees the child's own exit
    at once, as EOF on the result pipe, well inside the budget."""
    payload = handover_file(b"unused", "the report PDF")
    started = time.perf_counter()
    with pytest.raises(RuntimeError) as caught:
        in_bounded_child(
            _reader_that_finishes_with_the_watchdog_running,
            (),
            payload_path=payload,
            url="the report PDF",
            timeout=10,
        )
    assert not isinstance(caught.value, ExtractionTimeoutError)
    assert time.perf_counter() - started < 5
    assert caught.value.exitcode == 0  # its own exit, not the parent's SIGTERM
    assert "exited without a result (exit code 0)" in web_log.text


# --- the models and the call ----------------------------------------------------


def _list(*entries: str) -> ReferenceList:
    return ReferenceList(references=[Reference(entry=entry) for entry in entries])


def _cited(*names: str) -> list[Reference]:
    """References that each print a DOI, told apart by name: entry `n`, DOI
    `10.1000/n` — the spelling _FakeUnpaywall.by_doi reads the name back
    from."""
    return [Reference(entry=name, doi=f"10.1000/{name}") for name in names]


def _lines(count: int) -> str:
    """A reference list of `count` one-line entries, numbered so a test can
    tell which lines a chunk carried; 340 lines is ~30,000 characters."""
    return "\n".join(f"Ref {i:04d}. {ENTRY}" for i in range(count))


_LINE_CHARS = len(_lines(1))


def _prose(count: int) -> str:
    """`count` lines of _lines's length, none shaped like the start of an
    entry (no year after a period or paren, no surname-comma-initial, no
    leading number): a test about how the parts are read and joined stays
    clear of the completeness guard, which would ask again about a part
    answered with two entries where the text shows a hundred."""
    filler = "prose that no rule reads as the start of an entry " * 2
    return "\n".join(f"Line {i:04d} {filler}"[:_LINE_CHARS] for i in range(count))


# Six entries in the shapes the example reports print (pypdf text, cited by
# file): an author-year line (2024 GHI), a paren-dated one (Drought), an
# organisation with the year after a paren (the water article), a wrapped
# author list whose year opens the NEXT line (2024 GHI, Black et al.), a
# bracket-numbered one (disruptions) and a dot-numbered one — around a
# heading, a letter header, a continuation line and that wrapped year line,
# which are not starts. entry_starts reads 6 here.
SIX_ENTRIES = "\n".join(
    [
        "References",
        "A",
        "Agarwal, B. 2019. “Does Group Farming Empower Rural Women? Lessons from India’s ",
        "Experiments.” Journal of Peasant Studies 47 (4): 841–872. https://doi.org/10.1080/03066",
        "Addis Standard. (2024, January 10). News: Four million Ethiopians on the brink.",
        "Australian Standards/New Zealand Standards (AS/NZS) 2016 Water Efficient Products.",
        "Black, R. E., C. G. Victora, S. P. Walker, Z. A. Bhutta, P. Christian, et al. ",
        "2013. “Maternal and Child Undernutrition and Overweight.” Lancet  832 (9890): 427–451.",
        "[5]Z.B. Anis, H.U.U. Rahman, N. Khalid, Effect of food quality, Sustainability 14 (2022).",
        "1. Smith J, Jones K. Title of the paper. Journal. 2020;12:1-9.",
    ]
)


def _answer_by_part(answers: dict[int, ReferenceList]):
    """A FakeLLM answer keyed on the part number the prompt names — the same
    answer for a chunk whatever order the pool runs the chunks in."""

    def answer(prompt: str) -> ReferenceList:
        match = re.search(r"part (\d+) of (\d+)", prompt)
        assert match, prompt[:80]
        return answers[int(match.group(1))]

    return answer


def test_a_short_list_is_one_call_under_the_references_contract():
    wanted = ReferenceList(
        references=[
            Reference(
                entry=ENTRY, title="Water stress and cities", authors=["Smith, J."], year=2020
            )
        ]
    )
    llm = FakeLLM({ReferenceList: wanted})
    result = extract_references(llm, "claude-haiku-4-5", "References\n" + ENTRY)
    assert result.references == wanted.references
    assert result.dropped == 0
    assert len(llm.parse_calls) == 1
    call = llm.parse_calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["system"] == REFERENCES_SYSTEM
    assert call["prompt"] == "CLOSING PAGES:\n\nReferences\n" + ENTRY
    assert "part" not in call["prompt"]
    assert call["output_type"] is ReferenceList
    assert call["images"] is None


def test_the_references_call_is_sampled_at_temperature_zero():
    """Six identical live calls on one 37-entry list answered 37 / 1 / 37 /
    37 / 1 / 1 (MISTAKES 2026-09-23): the answer's SHAPE varied between
    calls. The reading is a transcription, not a judgment that needs
    variety, so the call asks for temperature 0 — a parameter Haiku 4.5, the
    pinned references model, accepts (the claude-api skill: Sonnet 5 and
    Opus 4.7+ refuse it with a 400, which is why the LLM protocol sends one
    only when a caller asks for it)."""
    llm = FakeLLM({ReferenceList: _list(ENTRY)})
    extract_references(llm, "claude-haiku-4-5", "References\n" + ENTRY)
    assert llm.parse_calls[0]["temperature"] == 0.0


def test_split_reference_text_keeps_a_short_list_whole():
    assert split_reference_text("References\n" + ENTRY) == ["References\n" + ENTRY]
    assert split_reference_text("x" * REFERENCE_CHUNK_CHARS) == ["x" * REFERENCE_CHUNK_CHARS]


def test_split_reference_text_cuts_a_long_list_at_line_breaks_in_order():
    """A 30,000-character list is three chunks of at most REFERENCE_CHUNK_CHARS,
    none cutting inside a line: every line reaches exactly one chunk, whole
    and in order, and no chunk is a sliver."""
    text = _lines(340)
    assert 29_000 <= len(text) <= 31_000
    chunks = split_reference_text(text)
    assert len(chunks) == 3
    assert all(len(chunk) <= REFERENCE_CHUNK_CHARS for chunk in chunks)
    assert [line for chunk in chunks for line in chunk.split("\n")] == text.split("\n")
    assert all(len(chunk) > REFERENCE_CHUNK_CHARS // 2 for chunk in chunks[:-1])


def test_split_reference_text_prefers_a_blank_line_between_entries():
    """Entries set apart by blank lines are cut between entries, so a cut
    splits an entry only when a chunk-sized stretch has no blank line."""
    entries = [f"Entry {i}. {ENTRY}\nsecond line of {i}" for i in range(200)]
    text = "\n\n".join(entries)
    chunks = split_reference_text(text)
    assert len(chunks) == 2
    assert [entry for chunk in chunks for entry in chunk.split("\n\n")] == entries


def test_a_cut_keeps_the_next_lines_indentation():
    """Bibliographies print continuation lines with a hanging indent. Only
    the line break(s) at a cut are dropped, so the line that opens the next
    chunk keeps its leading spaces and every line reaches its chunk intact
    — the docstring's word. At a space cut the one space still goes, and a
    hard cut drops nothing (the other test's join checks hold both)."""
    text = "\n".join("   " + line for line in _lines(340).split("\n"))
    chunks = split_reference_text(text)
    assert len(chunks) == 3
    assert all(chunk.startswith("   Ref ") for chunk in chunks)
    assert [line for chunk in chunks for line in chunk.split("\n")] == text.split("\n")


def test_a_blank_line_early_in_the_window_does_not_make_a_sliver():
    """The blank-line cut is taken only in the second half of the window. A
    blank line 200 characters into a 12,000-character window, with no later
    one, must not end the chunk there: the cut falls back to the window's
    last line break, so the first chunk is more than half a window and the
    blank line stays inside it. A blank line IN the second half is taken,
    so entries stay whole where the layout marks their boundaries."""
    early = "x" * 198 + "\n\n" + _lines(340)
    chunks = split_reference_text(early)
    assert chunks[0] != "x" * 198
    assert len(chunks[0]) > REFERENCE_CHUNK_CHARS // 2
    assert chunks[0].startswith("x" * 198 + "\n\nRef 0000.")
    assert [line for chunk in chunks for line in chunk.split("\n")] == early.split("\n")
    late = _lines(80) + "\n\n" + _lines(340)
    assert REFERENCE_CHUNK_CHARS // 2 < len(_lines(80)) < REFERENCE_CHUNK_CHARS
    assert split_reference_text(late)[0] == _lines(80)


def test_split_reference_text_cuts_a_line_longer_than_a_chunk_at_a_space():
    """A PDF whose text lost its line breaks: no line break to cut at, so the
    last space serves and, with none, the cap itself — the output budget
    holds either way."""
    words = ("word " * 5_000).strip()
    chunks = split_reference_text(words)
    assert len(chunks) == 3
    assert all(len(chunk) <= REFERENCE_CHUNK_CHARS for chunk in chunks)
    assert " ".join(chunks) == words
    solid = "x" * 25_000
    chunks = split_reference_text(solid)
    assert [len(chunk) for chunk in chunks] == [12_000, 12_000, 1_000]
    assert "".join(chunks) == solid


def test_a_long_list_is_read_in_parts_and_concatenated_in_part_order():
    """A 30,000-character list is three calls, each part's prompt naming the
    part and carrying whole lines only, under the one system prompt; the
    answer is the parts' answers in part order, whatever order the pool
    finished them in."""
    text = _prose(340)
    answers = {1: _list("a1", "a2"), 2: _list("b1"), 3: _list("c1", "c2", "c3")}
    llm = FakeLLM({ReferenceList: _answer_by_part(answers)})
    result = extract_references(llm, "m", text)
    assert [r.entry for r in result.references] == ["a1", "a2", "b1", "c1", "c2", "c3"]
    assert len(llm.parse_calls) == 3
    assert all(call["system"] == REFERENCES_SYSTEM for call in llm.parse_calls)
    assert all(call["model"] == "m" for call in llm.parse_calls)
    prompts = sorted(call["prompt"] for call in llm.parse_calls)
    for number, prompt in enumerate(prompts, start=1):
        assert prompt.startswith(f"CLOSING PAGES, part {number} of 3")
        assert "may begin or end in the middle of an entry" in prompt
        assert "never completed" in prompt
    bodies = [prompt.split("\n\n", 1)[1] for prompt in prompts]
    assert bodies == split_reference_text(text)
    assert [line for body in bodies for line in body.split("\n")] == text.split("\n")


def test_a_boundary_inside_an_entry_yields_two_fragments_never_a_merged_entry():
    """An entry a cut splits across two parts comes back as its two printed
    fragments, each listed by the part that saw it. The code joins the
    parts' answers and never welds fragments: it cannot know two entries
    were one, and a wrong weld would be an entry the page does not print."""
    text = _prose(340)
    head = "Long, A. (2021). A title that continues"
    tail = "on the next line. Journal, 3(1), 1-2."
    answers = {1: _list("Ref 0138.", head), 2: _list(tail, "Ref 0140."), 3: _list("Ref 0339.")}
    llm = FakeLLM({ReferenceList: _answer_by_part(answers)})
    result = extract_references(llm, "m", text)
    entries = [r.entry for r in result.references]
    assert entries == ["Ref 0138.", head, tail, "Ref 0140.", "Ref 0339."]


def test_a_failing_part_fails_the_whole_extraction():
    """No partial bibliography: a part whose call fails raises, as the single
    call did, so the endpoint answers 500 rather than a list with a silent
    hole in it."""
    text = _lines(340)

    def answer(prompt: str) -> ReferenceList:
        if "part 2 of 3" in prompt:
            raise RuntimeError("LLM call produced no parseable ReferenceList")
        return _list("ok")

    with pytest.raises(RuntimeError, match="no parseable ReferenceList"):
        extract_references(FakeLLM({ReferenceList: answer}), "m", text)


def test_a_long_lists_parts_are_read_four_at_a_time():
    """Four chunks, each call held at a Barrier(LOOKUP_WORKERS) until every
    one of the four is inside the model call at the same moment: the
    barrier releases only when all parties have arrived, so a pool of fewer
    workers, or a serial loop, leaves the first call waiting alone until
    the barrier times out and breaks — the extraction then raises
    BrokenBarrierError instead of hanging."""
    text = _prose(480)
    assert len(split_reference_text(text)) == references.LOOKUP_WORKERS == 4
    barrier = threading.Barrier(references.LOOKUP_WORKERS, timeout=5)

    def answer(prompt: str) -> ReferenceList:
        barrier.wait()
        return _list("ok")

    llm = FakeLLM({ReferenceList: answer})
    result = extract_references(llm, "m", text)
    assert not barrier.broken
    assert len(llm.parse_calls) == 4
    assert [r.entry for r in result.references] == ["ok"] * 4


# --- the completeness guard --------------------------------------------------


def test_entry_starts_counts_the_lines_that_open_an_entry():
    """The yardstick: lines in the shapes the example reports print (see
    SIX_ENTRIES) count; a heading, a letter header, a continuation line, a
    wrapped DOI, a section heading, prose with a year, a dated access note
    and a footnote-numbered entry (the Hunger Hotspots report — a known
    limit, which can only ever mean no retry) do not."""
    assert entry_starts(SIX_ENTRIES) == 6
    assert entry_starts("") == 0
    for line in [
        "References",
        "A",
        "Experiments.” Journal of Peasant Studies 47 (4): 841–872. https://doi.org/10.1080/03066",
        "150.2019.1628020.",
        "2013. “Maternal and Child Undernutrition and Overweight.” Lancet  832 (9890): 427–451.",
        "1. Introduction",
        "2. Hunger in 2025",
        "6. Indicators at a Glance (Real and Fabricated)",
        "In 2023, water stress rose across the basin.",
        "Global Water Stress and Hunger Assessment 2025",
        "Accessed July 16, 2025. https://acleddata.com/conflict-index/.",
        "199  International Crisis Group. 2024. Crisis Watch Sudan.",
    ]:
        assert entry_starts(line) == 0, line
    for line in [
        "   Agarwal, B. 2019. “Does Group Farming Empower Rural Women?",  # indented
        "FAO. 1997a. Irrigation in the near east region in figures. Rome: FAO.",
        "WHO (2020) Guidelines on drinking water.",
        "[1]L. Abuabara, K. Werner-Masters, A. Paucar-Caceres, Daily food planning",
        "3) World Health Organization. Global report. 2021.",
        "Bezner Kerr, R., S. Madsen, M. Stüber, J. Liebert, S. Enloe, N. Borghino,",
    ]:
        assert entry_starts(line) == 1, line


def test_a_line_continuing_an_author_list_is_not_a_second_start():
    """The Drought list wraps long author lists in a narrow column, so
    'Segadlo, N. (2019).' opens a line of its own: one entry, counted once
    — a line after one ending in a comma, an ampersand or 'and' continues
    the list. Alone, the same line is a start."""
    first = "Adaawen, S., Rademacher-Schulz, C., Schraven, B., &"
    second = "Segadlo, N. (2019). Drought, migration, and conflict"
    assert entry_starts(f"{first}\n{second}") == 1
    assert entry_starts(second) == 1
    assert entry_starts("Bezner Kerr, R., S. Madsen, M. Stüber, J. Liebert, and\n" + second) == 1


def test_a_short_answer_is_retried_once_and_the_full_retry_kept(references_log):
    """The live '1 reference' shape: the text shows six entry starts and the
    answer holds one. The part is asked again with the same prompt, the
    retry's six are kept, and the scan is not flagged."""
    llm = FakeLLM({ReferenceList: [_list("one"), _list(*"abcdef")]})
    result = extract_references(llm, "m", SIX_ENTRIES)
    assert [r.entry for r in result.references] == list("abcdef")
    assert result.possibly_incomplete is False
    assert len(llm.parse_calls) == 2
    assert llm.parse_calls[0]["prompt"] == llm.parse_calls[1]["prompt"]
    assert "1 references for about 6 entry starts" in references_log.text
    assert "possibly incomplete" not in references_log.text


@pytest.mark.parametrize("answers", [["one"], ["a", "b"]], ids=["first-smaller", "first-larger"])
def test_an_answer_short_twice_keeps_the_larger_and_flags_the_scan(answers, references_log):
    """Both answers fall under half the six starts: the one with more
    references is kept whichever came first, and the scan says it may be
    incomplete — a WARNING with both counts, and the flag the dialog shows."""
    first, second = (_list(*answers), _list(*(["one"] if answers != ["one"] else ["a", "b"])))
    llm = FakeLLM({ReferenceList: [first, second]})
    result = extract_references(llm, "m", SIX_ENTRIES)
    assert [r.entry for r in result.references] == ["a", "b"]
    assert result.possibly_incomplete is True
    assert len(llm.parse_calls) == 2
    assert "WARNING" in references_log.text
    assert "still looks incomplete after a retry" in references_log.text
    assert "2 references for about 6 entry starts" in references_log.text


def test_a_merged_answer_is_retried_whatever_the_text_shows(references_log):
    """The live '5,117 tokens, 1 reference' shape: one object whose entry text
    holds the whole list. Its RAW entry, before the prefix cut, runs past
    three times ENTRY_PREFIX_CHARS, which reads as merged even where the
    text shows fewer than four entry starts; a full retry is kept, and a
    second merged answer flags the scan."""
    merged = _list(("Smith, J. (2020). A title. " * 20).strip())
    assert len(merged.references[0].entry) > 3 * ENTRY_PREFIX_CHARS
    llm = FakeLLM({ReferenceList: [merged, _list("a", "b")]})
    result = extract_references(llm, "m", "References\n" + ENTRY)
    assert [r.entry for r in result.references] == ["a", "b"]
    assert result.possibly_incomplete is False
    assert len(llm.parse_calls) == 2
    assert "entries merged into one object" in references_log.text

    llm = FakeLLM({ReferenceList: [merged, merged]})
    result = extract_references(llm, "m", "References\n" + ENTRY)
    assert len(result.references) == 1
    assert len(result.references[0].entry) == ENTRY_PREFIX_CHARS
    assert result.possibly_incomplete is True
    assert len(llm.parse_calls) == 2


def test_a_good_answer_is_neither_retried_nor_flagged(references_log):
    llm = FakeLLM({ReferenceList: _list(*"abcdef")})
    result = extract_references(llm, "m", SIX_ENTRIES)
    assert result.possibly_incomplete is False
    assert len(llm.parse_calls) == 1
    assert references_log.text == ""


def test_a_text_with_few_entry_starts_never_triggers_the_guard(references_log):
    """Under ENTRY_STARTS_FLOOR starts — the fake reports show one or two in
    their prose — an empty answer is the right answer for a document with
    no list, and half of three is not a count to hold the model to: no
    retry, no flag. At the floor the ratio applies."""
    assert ENTRY_STARTS_FLOOR == 4
    three = "\n".join(SIX_ENTRIES.split("\n")[:6])
    assert entry_starts(three) == 3
    llm = FakeLLM({ReferenceList: _list()})
    result = extract_references(llm, "m", three)
    assert (result.references, result.possibly_incomplete) == ([], False)
    assert len(llm.parse_calls) == 1
    assert references_log.text == ""
    four = three + "\n" + "[5]Z.B. Anis, H.U.U. Rahman, N. Khalid, Effect of food quality (2022)."
    assert entry_starts(four) == 4
    llm = FakeLLM({ReferenceList: [_list("one"), _list("one")]})
    assert extract_references(llm, "m", four).possibly_incomplete is True
    assert len(llm.parse_calls) == 2


def test_one_short_part_flags_the_whole_scan_after_its_own_retry(references_log):
    """Three parts of ~113 entry starts each; part 2 answers one entry both
    times, the others 120. Only part 2 is asked again (four calls), the
    parts' answers are joined as ever, and the scan is flagged once."""
    text = _lines(340)
    answers = {
        1: _list(*(f"p1-{i}" for i in range(120))),
        2: _list("one"),
        3: _list(*(f"p3-{i}" for i in range(120))),
    }
    llm = FakeLLM({ReferenceList: _answer_by_part(answers)})
    result = extract_references(llm, "m", text)
    assert len(result.references) == 241
    assert result.references[120].entry == "one"
    assert result.possibly_incomplete is True
    assert len(llm.parse_calls) == 4
    assert sum("part 2 of 3" in call["prompt"] for call in llm.parse_calls) == 2
    assert references_log.text.count("still looks incomplete") == 1


def test_entries_are_cut_to_the_prefix_length_verbatim(references_log):
    """The prompt asks for the first ENTRY_PREFIX_CHARS characters; a model
    that returns more is cut to exactly that prefix in code, without a
    warning (the prefix is the contract), and a shorter entry is untouched."""
    printed = "Long, A., Longer, B., & Longest, C. (2020). " + "A title that runs on. " * 12
    assert len(printed) > ENTRY_PREFIX_CHARS
    result = extract_references(FakeLLM({ReferenceList: _list(printed, "short")}), "m", "R")
    assert result.references[0].entry == printed[:ENTRY_PREFIX_CHARS]
    assert len(result.references[0].entry) == ENTRY_PREFIX_CHARS
    assert result.references[1].entry == "short"
    assert references_log.text == ""
    assert f"first {ENTRY_PREFIX_CHARS} characters" in REFERENCES_SYSTEM


def test_titles_and_authors_are_cut_in_code_after_parsing_like_the_entry():
    """Bounds are code, not schema (see the constants): a title or an author
    name is model output with no length the schema may enforce, and it is
    shown in the dialog and matched against the user's sources. The entry's
    prefix rule already existed; title and authors get the same treatment,
    and a name list is cut to MAX_AUTHORS."""
    from authorai.references import AUTHOR_MAX_CHARS, MAX_AUTHORS, TITLE_MAX_CHARS

    answer = ReferenceList(
        references=[
            Reference(
                entry="e",
                title="T" * (TITLE_MAX_CHARS + 100),
                authors=[f"A{i}" + "a" * AUTHOR_MAX_CHARS for i in range(MAX_AUTHORS + 10)],
            ),
            Reference(entry="short", title="Short", authors=["One, A.", "Two, B."]),
        ]
    )
    result = extract_references(FakeLLM({ReferenceList: answer}), "m", "References\n" + ENTRY)
    long, short = result.references
    assert long.title == "T" * TITLE_MAX_CHARS
    assert len(long.authors) == MAX_AUTHORS
    assert all(len(name) == AUTHOR_MAX_CHARS for name in long.authors)
    assert long.authors[0].startswith("A0")
    assert (short.title, short.authors) == ("Short", ["One, A.", "Two, B."])
    assert (TITLE_MAX_CHARS, MAX_AUTHORS, AUTHOR_MAX_CHARS) == (500, 50, 200)


def test_a_printed_address_and_doi_are_cut_in_code_like_the_other_fields():
    """The two fields _bounded did not cut: a row that has only one of them
    is still actionable, and the dialog labels it by that value and shows
    it as the row's tooltip, so a model-written 60,000-character address
    reached the DOM whole. An address longer than fetch.MAX_URL_LENGTH can
    never become a link (offerable_url refuses it, so it is never
    suggested), and clean_doi refuses a DOI over 256 characters, so the
    cuts change nothing real."""
    long = "x" * 60_000
    answer = ReferenceList(
        references=[
            Reference(entry="", url="https://x.org/" + long),
            Reference(entry="", doi="10.1000/" + long),
        ]
    )
    result = extract_references(FakeLLM({ReferenceList: answer}), "m", "References")
    by_url, by_doi = result.references
    assert len(by_url.url) == URL_FIELD_MAX_CHARS == MAX_URL_LENGTH == 2048
    assert by_url.url.startswith("https://x.org/")
    assert len(by_doi.doi) == DOI_FIELD_MAX_CHARS == 300
    assert by_doi.doi.startswith("10.1000/")
    assert result.dropped == 0  # a cut row is still the row


def test_extract_references_caps_the_list_in_code_with_a_warning(references_log):
    """The cap is NOT in the schema (structured outputs may not honour
    max_length, and a client-side validation failure would 500 the scan): the
    model may return more, and code keeps the first MAX_REFERENCES, loudly."""
    too_many = ReferenceList(
        references=[Reference(entry=f"entry {i}") for i in range(MAX_REFERENCES + 5)]
    )
    result = extract_references(FakeLLM({ReferenceList: too_many}), "m", "References")
    assert len(result.references) == MAX_REFERENCES
    assert result.references[-1].entry == f"entry {MAX_REFERENCES - 1}"
    assert f"{MAX_REFERENCES + 5} references" in references_log.text
    assert "WARNING" in references_log.text


def test_the_cap_applies_to_the_parts_concatenated(references_log):
    """Three parts of 120 entries each are 360, over MAX_REFERENCES: the
    first 300 in part order are kept, and the warning counts the parts."""
    text = _lines(340)
    answers = {n: _list(*(f"p{n}-{i}" for i in range(120))) for n in (1, 2, 3)}
    result = extract_references(FakeLLM({ReferenceList: _answer_by_part(answers)}), "m", text)
    entries = [r.entry for r in result.references]
    assert len(entries) == MAX_REFERENCES == 300
    assert entries[0] == "p1-0"
    assert entries[119] == "p1-119"
    assert entries[120] == "p2-0"
    assert entries[-1] == "p3-59"
    assert "360 references over 3 parts" in references_log.text


def test_extract_references_returns_a_short_list_untouched(references_log):
    two = ReferenceList(references=[Reference(entry="a"), Reference(entry="b")])
    result = extract_references(FakeLLM({ReferenceList: two}), "m", "References")
    assert (result.references, result.dropped, result.possibly_incomplete) == (
        two.references,
        0,
        False,
    )
    assert references_log.text == ""


def test_extract_references_counts_every_row_it_dropped(monkeypatch, references_log):
    """Blank rows and the rows past MAX_REFERENCES alike: the count the scan
    reports as `references_dropped`."""
    monkeypatch.setattr(references, "MAX_REFERENCES", 2)
    answer = ReferenceList(
        references=[Reference(entry=""), *[Reference(entry=f"r{i}") for i in range(5)]]
    )
    result = extract_references(FakeLLM({ReferenceList: answer}), "m", "References")
    assert [r.entry for r in result.references] == ["r0", "r1"]
    assert result.dropped == 1 + 3


def test_blank_optional_fields_read_as_absent():
    """The model sometimes prints '' for a field it was told to leave null;
    downstream code (the DOI lookup, the printed-URL rule) keys on absence.
    `entry` is a str, not optional: a blank one stays "" (never None, which
    the field's own type rejects on re-validation) and is judged by the
    actionable rule instead."""
    reference = Reference(entry="  x  ", title="  ", doi="", url=" \n", authors=["A", "  ", "B"])
    assert reference.entry == "x"
    assert reference.title is None
    assert reference.doi is None
    assert reference.url is None
    assert reference.authors == ["A", "B"]
    assert Reference(entry="").entry == ""
    assert Reference(entry=" \n\t ").entry == ""


def test_a_blank_entry_is_dropped_loudly_unless_the_row_names_the_work(references_log):
    """A model slip: `entry` printed as "" or whitespace where the prompt
    asked for the entry's first characters. The rule, stated once in
    `actionable`: with no title, DOI or address either, the row is not a
    reference the user can act on and is dropped — counted in a WARNING,
    never silently; with any of those it is kept, its entry "", since the
    dialog can still show the title and check the work. Authors and a year
    alone name nothing the dialog can show, so that row is dropped too."""
    answer = ReferenceList(
        references=[
            Reference(entry=""),
            Reference(entry="   ", title="Titled work"),
            Reference(entry="\n\t", doi="10.1000/x"),
            Reference(entry="", url="https://x.org/p"),
            Reference(entry="  ", authors=["Only, A."], year=2020),
            Reference(entry="kept as printed"),
        ]
    )
    result = extract_references(FakeLLM({ReferenceList: answer}), "m", "References")
    assert [(r.entry, r.title, r.doi, r.url) for r in result.references] == [
        ("", "Titled work", None, None),
        ("", None, "10.1000/x", None),
        ("", None, None, "https://x.org/p"),
        ("kept as printed", None, None, None),
    ]
    assert all(isinstance(r.entry, str) for r in result.references)
    assert "WARNING" in references_log.text
    assert "dropping 2 of 6 references" in references_log.text


def test_a_null_entry_is_a_parse_failure_not_a_blank_row():
    """`entry` is a required string in the schema the model decodes under,
    so the constrained answer cannot carry a null there; a hand-built one
    fails validation at parse time — the model-failure 500, like any
    unparseable answer — and never reaches the actionable rule."""
    from anthropic import transform_schema
    from pydantic import ValidationError

    schema = transform_schema(ReferenceList)["$defs"]["Reference"]
    assert "entry" in schema["required"]
    assert schema["properties"]["entry"]["type"] == "string"
    with pytest.raises(ValidationError):
        Reference(entry=None)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ReferenceList.model_validate({"references": [{"entry": None}]})


def test_the_schema_sent_to_the_api_carries_no_constraint_it_might_reject():
    """Decision: bounds live in code. Structured outputs do not support string
    or numeric constraints; the SDK would strip them and validate client-side,
    turning an over-long model answer into a 500."""
    from anthropic import transform_schema

    dumped = json.dumps(transform_schema(ReferenceList))
    for keyword in (
        "maxLength",
        "minLength",
        "pattern",
        "maxItems",
        "minItems",
        "minimum",
        "maximum",
    ):
        assert keyword not in dumped, keyword
    assert references.MAX_REFERENCES == 300
    assert references.REFERENCE_MAX_CHARS == 65_000
    assert references.REFERENCE_CHUNK_CHARS == 12_000
    assert references.ENTRY_PREFIX_CHARS == 160
    assert references.HEADING_RUN_PAGES == 2
    assert references.REFERENCE_MAX_PAGES == 600


def test_a_chunks_worst_case_output_fits_the_parse_budget_with_margin():
    """The output budget is the design constraint (MISTAKES 2026-09-23): a
    dense list prints an entry every ~120 characters, and one entry with a
    160-character `entry`, title, authors, year, DOI and JSON keys is ~100
    output tokens, so a chunk's worst case must sit well under
    PARSE_MAX_TOKENS. Raising REFERENCE_CHUNK_CHARS re-runs this arithmetic."""
    entries_per_chunk = REFERENCE_CHUNK_CHARS // 120
    tokens_per_entry = ENTRY_PREFIX_CHARS // 4 + 60
    assert entries_per_chunk * tokens_per_entry <= PARSE_MAX_TOKENS * 2 // 3


def test_the_prompt_asks_for_every_entry_of_a_long_list():
    """The cap admits 300 entries, so the model must not stop early or
    summarize a long list, and a list read in parts must list a fragment as
    printed: the contract says so in words."""
    contract = " ".join(REFERENCES_SYSTEM.split())  # the prompt wraps at 80 columns
    assert "may be long" in contract
    assert "every entry" in contract
    assert "arrives in parts" in contract
    assert "never completed" in contract


def test_the_prompt_opens_with_the_one_object_per_entry_rule():
    """Measured live (MISTAKES 2026-09-23): six identical calls on a 37-entry
    list answered 37 / 1 / 37 / 37 / 1 / 1 — the model either closed the
    list after one short object or put the whole list into one object's
    text. The structural rule is therefore the FIRST thing the prompt says,
    in one unmistakable sentence pair, before any field is described."""
    contract = " ".join(REFERENCES_SYSTEM.split())  # the prompt wraps at 80 columns
    rule = (
        "Return ONE object per entry. A reference list of N entries yields exactly N "
        "objects — never combine entries into one object, never stop before the last "
        "entry of the text you were given."
    )
    assert contract.startswith(rule)


# --- Unpaywall: the one registry-GET policy, on a second registry --------------

MAILTO = "checker@example.org"


def _unpaywall() -> UnpaywallClient:
    return UnpaywallClient(MAILTO)


def _record(is_oa=True, url_for_pdf=None, url_for_landing_page=None, url=None) -> dict:
    """An Unpaywall v2 record in the shape the live API answers with: a
    closed work has `best_oa_location: null`; an open one names its best copy
    with `url_for_pdf` NULL for many genuinely open works (the heliyon case)
    and `url` the landing page."""
    if not is_oa:
        return {"is_oa": False, "best_oa_location": None}
    return {
        "is_oa": True,
        "best_oa_location": {
            "url_for_pdf": url_for_pdf,
            "url_for_landing_page": url_for_landing_page,
            "url": url,
        },
    }


def _no_sleep(monkeypatch):
    # The retry policy is credibility's; its backoff sleeps through that module.
    monkeypatch.setattr("authorai.credibility.time.sleep", lambda seconds: None)


@respx.mock
def test_unpaywall_timeout_retries_then_raises_loudly(monkeypatch):
    """An unreachable Unpaywall is an outage, not 'paywalled' — silently
    listing every cited work as unretrievable would hide the failure."""
    _no_sleep(monkeypatch)
    route = respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/slow")
    route.side_effect = httpx.ConnectTimeout("slow")
    with pytest.raises(RuntimeError, match="no answer after 3 attempts"):
        _unpaywall().by_doi("10.1000/slow")
    assert route.call_count == 3  # initial + 2 retries


@respx.mock
def test_unpaywall_throttling_retries_then_succeeds(monkeypatch):
    _no_sleep(monkeypatch)
    route = respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/busy")
    route.side_effect = [
        httpx.Response(429),
        httpx.Response(200, json=_record(url_for_pdf="https://x.org/busy.pdf")),
    ]
    assert _unpaywall().by_doi("10.1000/busy") == _record(url_for_pdf="https://x.org/busy.pdf")
    assert route.call_count == 2


@respx.mock
def test_unpaywall_server_errors_raise_after_retries(monkeypatch):
    _no_sleep(monkeypatch)
    route = respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/down").mock(return_value=httpx.Response(503))
    with pytest.raises(RuntimeError, match="HTTP 503"):
        _unpaywall().by_doi("10.1000/down")
    assert route.call_count == 3


@respx.mock
def test_unpaywall_malformed_200_body_raises_instead_of_reading_as_not_found():
    respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/xyz").mock(return_value=httpx.Response(200, json=[]))
    with pytest.raises(RuntimeError, match="non-object body"):
        _unpaywall().by_doi("10.1000/xyz")


@respx.mock
def test_a_200_whose_body_is_not_json_raises_the_registry_failure_not_a_decode_error():
    """A captive portal or CDN error page answers 200 with HTML. Like the
    non-object case it is a malfunction, and it must surface as the same
    RuntimeError the pooled lookup turns into "unavailable" — a decode error
    escaping here would fail the whole scan with a 500."""
    respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/html").mock(
        return_value=httpx.Response(
            200,
            text="<html><body>Service degraded</body></html>",
            headers={"content-type": "text/html"},
        )
    )
    with pytest.raises(RuntimeError, match="not JSON") as caught:
        _unpaywall().by_doi("10.1000/html")
    assert not isinstance(caught.value, ValueError)
    assert "Unpaywall" in str(caught.value)
    assert "/v2/10.1000/html" in str(caught.value)


@respx.mock
def test_an_unknown_doi_is_an_html_404_and_reads_as_not_found():
    """Live: Unpaywall answers an unknown DOI with HTTP 404 and an HTML body.
    That is an answer — None — and the body is never parsed as JSON (which
    would raise on the HTML)."""
    respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/nope").mock(
        return_value=httpx.Response(
            404,
            text="<html><body><h1>Not Found</h1></body></html>",
            headers={"content-type": "text/html"},
        )
    )
    assert _unpaywall().by_doi("10.1000/nope") is None


@respx.mock
def test_the_request_names_the_operator_and_cleans_the_doi():
    route = respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/abc").mock(
        return_value=httpx.Response(200, json=_record())
    )
    _unpaywall().by_doi("https://doi.org/10.1000/abc")  # the URL prefix is stripped first
    request = route.calls.last.request
    assert request.url.params["email"] == MAILTO
    assert request.headers["User-Agent"] == f"AuthorAI/2.0 (mailto:{MAILTO})"


@respx.mock
def test_a_malformed_doi_makes_no_request():
    # No route mocked: any HTTP call would make respx raise.
    for bad in ("not-a-doi", "10.1/x", "https://doi.org/nope", "10.1000/with space"):
        assert _unpaywall().by_doi(bad) is None


@respx.mock
def test_the_request_path_is_exactly_the_doi_under_v2():
    """The DOI is the whole of the path after /v2/, dots inside a segment
    included (the heliyon DOI is a real, open one)."""
    doi = "10.1016/j.heliyon.2024.e34730"
    route = respx.get(f"{UNPAYWALL_BASE}/v2/{doi}").mock(
        return_value=httpx.Response(200, json=_record())
    )
    _unpaywall().by_doi(doi)
    assert str(route.calls.last.request.url) == (
        f"{UNPAYWALL_BASE}/v2/{doi}?email=checker%40example.org"
    )


@respx.mock
def test_a_doi_that_would_steer_the_request_path_makes_no_request():
    """The DOI is model-extracted text. httpx resolves "." and ".." segments
    even after quoting (RFC 3986 dot-segment removal), so "10.1000/../../admin"
    would GET /admin on the registry. Such a DOI is malformed: no request."""
    route = respx.route(host="api.unpaywall.org").mock(return_value=httpx.Response(404))
    for bad in (
        "10.1000/../../admin",
        "10.1000/x/../../../etc",
        "10.1000/./x",
        "10.1000/x/.",
        "10.1000/..",
        "10.1000/a\\b",
        "10.1000/..;/x",  # a servlet container drops ";..." before normalizing
        "10.1000/x/.;y/z",
        "10.1000/..%3B/x",  # the same, with the ";" percent-encoded
        "10.1000/%2e%2e/x",  # a server decodes before it normalizes
        "10.1000/%252e%252e/x",  # ... and a decoding proxy in front of it decodes once more
        "10.1000/x%2F..%2F..%2Fadmin",  # encoded slashes become segments on such a server
        "10.1000/a%5Cb",  # an encoded backslash
    ):
        assert _unpaywall().by_doi(bad) is None, bad
    assert route.call_count == 0, [str(c.request.url) for c in route.calls]


@respx.mock
def test_a_semicolon_that_hides_no_dot_segment_is_requested_exactly():
    """A ";" after ordinary text is DOI punctuation, not a hidden dot
    segment: the lookup goes out, quoted, as the DOI under /v2/. Encoded in
    the DOI itself it is still ordinary text — that DOI goes out exactly as
    given, its "%" quoted once more, never decoded into something else."""
    route = respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/a%3Bb/c").mock(
        return_value=httpx.Response(200, json=_record())
    )
    _unpaywall().by_doi("10.1000/a;b/c")
    assert str(route.calls.last.request.url) == (
        f"{UNPAYWALL_BASE}/v2/10.1000/a%3Bb/c?email=checker%40example.org"
    )
    encoded = respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/a%253Bb/c").mock(
        return_value=httpx.Response(200, json=_record())
    )
    _unpaywall().by_doi("10.1000/a%3Bb/c")
    assert str(encoded.calls.last.request.url) == (
        f"{UNPAYWALL_BASE}/v2/10.1000/a%253Bb/c?email=checker%40example.org"
    )


@respx.mock
def test_an_empty_mailto_is_refused_before_any_request():
    """Unpaywall answers HTTP 422 without an email; refusing at construction
    is the loud version, and no route is mocked so a request would fail."""
    for empty in ("", "   "):
        with pytest.raises(ValueError, match="contact email"):
            UnpaywallClient(empty)


# --- retrievability: what the record lets us offer -------------------------------


def test_no_record_is_unknown():
    assert retrievability(None) == ("unknown", None)


def test_a_closed_work_is_paywalled():
    assert retrievability(_record(is_oa=False)) == ("paywalled", None)


def test_a_record_that_does_not_say_is_unknown_not_paywalled():
    assert retrievability({}) == ("unknown", None)
    assert retrievability({"is_oa": None}) == ("unknown", None)


def test_a_free_pdf_wins_over_the_landing_page():
    record = _record(url_for_pdf="https://x.org/a.pdf", url_for_landing_page="https://x.org/a")
    assert retrievability(record) == ("pdf", "https://x.org/a.pdf")


def test_the_heliyon_shape_offers_the_landing_page():
    """Live: 10.1016/j.heliyon.2024.e34730 is open access with url_for_pdf
    null and `url` the doi.org link — a landing page, still a free copy."""
    record = _record(url="https://doi.org/10.1016/j.heliyon.2024.e34730")
    assert retrievability(record) == ("landing", "https://doi.org/10.1016/j.heliyon.2024.e34730")


def test_an_open_work_with_no_address_is_unknown():
    assert retrievability(_record()) == ("unknown", None)


@pytest.mark.parametrize(
    "record",
    [
        {"is_oa": True, "best_oa_location": "https://x.org/a.pdf"},
        {"is_oa": True, "best_oa_location": [{"url_for_pdf": "https://x.org/a.pdf"}]},
        {"is_oa": True, "best_oa_location": {"url_for_pdf": 123}},
        {"is_oa": True, "best_oa_location": {"url_for_pdf": True}},
        {"is_oa": True, "best_oa_location": {"url_for_pdf": ["https://x.org/a.pdf"]}},
        {"is_oa": True, "best_oa_location": {"url_for_pdf": {"url": "https://x.org/a.pdf"}}},
    ],
    ids=["location-str", "location-list", "url-int", "url-bool", "url-list", "url-dict"],
)
def test_a_record_of_the_wrong_shape_is_unknown_never_an_exception(record, references_log):
    """The registry's record is trusted for its VALUES, never its types: a
    best_oa_location that is not an object, or an address that is not a
    string, reads as no address — with a warning — rather than raising in
    the caller's thread, where an exception is the scan's 500."""
    assert retrievability(record) == ("unknown", None)
    assert "WARNING" in references_log.text


def test_a_wrong_shaped_address_does_not_hide_the_next_one():
    record = {
        "is_oa": True,
        "best_oa_location": {"url_for_pdf": 123, "url_for_landing_page": "https://x.org/a"},
    }
    assert retrievability(record) == ("landing", "https://x.org/a")


def test_an_unusable_address_is_dropped_and_the_next_one_tried(references_log):
    """Every suggested address passes the same syntax gate a pasted link does;
    one the gate refuses is never offered, and the record's other address is."""
    assert retrievability(_record(url_for_pdf="ftp://x.org/a.pdf")) == ("unknown", None)
    assert "ftp://x.org/a.pdf" in references_log.text
    record = _record(url_for_pdf="javascript:alert(1)", url_for_landing_page="https://x.org/a")
    assert retrievability(record) == ("landing", "https://x.org/a")


@pytest.mark.parametrize(
    "bad",
    [
        "http://127.0.0.1/x",  # loopback
        "http://10.0.0.1/x",  # private
        "http://[::1]/x",  # IPv6 loopback
        "http://localhost/x",
        "http://Localhost./x",  # case and a trailing dot do not make another host
        "http://api.localhost/x",  # anything under .localhost is the local machine
        "http://169.254.169.254/latest/meta-data",  # link-local: the cloud metadata service
        "http://100.64.0.1/x",  # carrier-grade NAT
        "http://224.0.0.1/x",  # multicast
        "http://[::ffff:127.0.0.1]/x",  # loopback wrapped in IPv6
    ],
)
def test_an_address_the_fetcher_would_refuse_is_never_offered(bad, references_log):
    """The syntax gate passes these; the fetcher refuses them at ingest. A
    link the app will not read is not offered: the record's next address is
    tried instead, and a printed one is not offered at all."""
    record = _record(url_for_pdf=bad, url_for_landing_page="https://x.org/a")
    assert retrievability(record) == ("landing", "https://x.org/a"), bad
    assert "WARNING" in references_log.text
    assert "dropping the pdf address" in references_log.text
    assert printed_url(Reference(entry="e", url=bad)) is None, bad


@pytest.mark.parametrize(
    "video",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
    ],
)
def test_a_youtube_address_is_never_offered_because_the_dialog_would_refuse_it(
    video, references_log
):
    """The offer gate applies every refusal the dialog's own link check
    applies (lib/links.ts checkLink), YouTube included: a suggested link the
    dialog then refuses is a row the user can tick but never add. A record's
    next address is tried; a printed one is not offered."""
    record = _record(url_for_pdf=video, url_for_landing_page="https://x.org/a")
    assert retrievability(record) == ("landing", "https://x.org/a"), video
    assert "YouTube" in references_log.text
    assert printed_url(Reference(entry="e", url=video)) is None, video


def test_a_host_that_merely_contains_youtube_is_offered():
    for url in ("https://notyoutube.com/x", "https://youtube.com.example.org/x"):
        assert retrievability(_record(url_for_pdf=url)) == ("pdf", url), url


def test_a_public_literal_address_or_a_name_is_offered():
    """Only the address forms that can be judged without a network are
    judged: a public literal passes, and a NAME is offered as it is — a name
    that resolves to private space is refused at ingest by the fetch gate,
    which is where the DNS lookup belongs."""
    for url in (
        "https://x.org/a.pdf",
        "http://93.184.216.34/a.pdf",
        "http://[2606:4700::1111]/a.pdf",
        "https://internal.example.org/a.pdf",
    ):
        assert retrievability(_record(url_for_pdf=url)) == ("pdf", url), url
        assert printed_url(Reference(entry="e", url=url)) == url, url


def test_a_suggested_address_is_the_normalized_form():
    record = _record(url_for_pdf="HTTPS://X.org/a.pdf#page=3")
    assert retrievability(record) == ("pdf", "https://x.org/a.pdf")


def test_printed_url_is_offered_only_when_there_is_no_doi_to_check():
    """A reference with no DOI but a printed address is addable but UNCHECKED
    (the frontend labels it so); with a DOI the lookup's verdict rules."""
    assert printed_url(Reference(entry="e", url="https://x.org/r#top")) == "https://x.org/r"
    assert printed_url(Reference(entry="e", url="https://x.org/r", doi="10.1000/x")) is None
    assert printed_url(Reference(entry="e")) is None
    assert printed_url(Reference(entry="e", url="not a url")) is None


# --- the pooled lookup ----------------------------------------------------------


@respx.mock
def test_lookups_run_only_for_dois_and_stay_aligned_with_the_references():
    respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/open").mock(
        return_value=httpx.Response(200, json=_record(url_for_pdf="https://x.org/open.pdf"))
    )
    respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/closed").mock(
        return_value=httpx.Response(200, json=_record(is_oa=False))
    )
    refs = [
        Reference(entry="closed", doi="10.1000/closed"),
        Reference(entry="no doi", url="https://x.org/p"),
        Reference(entry="open", doi="10.1000/open"),
        Reference(entry="bad doi", doi="nope"),
    ]
    client = _unpaywall()
    try:
        resolved = lookup_retrievability(client, refs)
    finally:
        client.close()
    assert resolved == [
        ("paywalled", None),
        ("unknown", None),
        ("pdf", "https://x.org/open.pdf"),
        ("unknown", None),
    ]


@respx.mock
def test_the_same_doi_printed_by_several_entries_is_looked_up_once():
    """A report cites one work in several chapters: one request, and the
    verdict reaches every row that prints the DOI."""
    twice = respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/twice").mock(
        return_value=httpx.Response(200, json=_record(url_for_pdf="https://x.org/twice.pdf"))
    )
    other = respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/other").mock(
        return_value=httpx.Response(200, json=_record(is_oa=False))
    )
    refs = [
        Reference(entry="a", doi="10.1000/twice"),
        Reference(entry="b", doi="10.1000/other"),
        Reference(entry="c", doi="10.1000/twice"),
        Reference(entry="d"),
        Reference(entry="e", doi="10.1000/twice"),
    ]
    client = _unpaywall()
    try:
        resolved = lookup_retrievability(client, refs)
    finally:
        client.close()
    assert twice.call_count == 1
    assert other.call_count == 1
    assert resolved == [
        ("pdf", "https://x.org/twice.pdf"),
        ("paywalled", None),
        ("pdf", "https://x.org/twice.pdf"),
        ("unknown", None),
        ("pdf", "https://x.org/twice.pdf"),
    ]


def test_a_registry_that_answers_too_slowly_hits_the_lookup_deadline(monkeypatch):
    """Cancel-on-failure was the only exit: a registry answering 200 slowly
    (under its per-request timeout) never fails, and 300 DOIs four at a time
    at 9 s each is eleven minutes with the dialog waiting. The lookup phase
    has a deadline: at it, queued lookups are cancelled, in-flight ones are
    abandoned (their answers discarded; the caller is not held), and the
    scan answers "unavailable" with what WAS resolved — the same partial
    contract as a failure."""
    monkeypatch.setattr(references, "LOOKUP_DEADLINE_SECONDS", 0.3)
    fake = _FakeUnpaywall(blocking={"slow"})
    names = ["fast", "slow1", "slow2", "slow3", "slow4", "queued"]
    refs = _cited(*names)
    started = time.perf_counter()
    try:
        with pytest.raises(RegistryUnavailable, match="timed out after 0.3 seconds") as caught:
            lookup_retrievability(fake, refs)
        elapsed = time.perf_counter() - started
    finally:
        fake.release.set()
    assert elapsed < 2.0, elapsed  # the deadline, not the slow lookups
    assert caught.value.resolved == [("pdf", "https://x.org/fast.pdf"), *[("unknown", None)] * 5]
    threading.Event().wait(0.3)  # the released workers finish, and try the queue
    assert sorted(fake.calls) == sorted(f"10.1000/{n}" for n in names[:5])  # queued: never


def test_a_registry_failure_waits_for_in_flight_lookups_only_to_the_deadline(monkeypatch):
    """The failure branch used to shutdown(wait=True): a failure at 44 s
    followed by in-flight lookups against a registry gone silent held the
    dialog for their whole retry schedule (~33 s each) past the deadline.
    Now a failure gives the in-flight lookups what is LEFT of the deadline
    — here half a second, which one lookup never finishes in — then
    abandons them as the deadline does, with the failure's own message and
    what was reached. The queued lookup is still never requested."""
    monkeypatch.setattr(references, "LOOKUP_DEADLINE_SECONDS", 0.5)
    monkeypatch.setattr(references, "LOOKUP_WORKERS", 2)
    fake = _FakeUnpaywall(raising={"fail": RuntimeError("Unpaywall HTTP 503")}, blocking={"slow"})
    refs = _cited("slow1", "fail", "queued")
    started = time.perf_counter()
    try:
        with pytest.raises(RegistryUnavailable, match="HTTP 503") as caught:
            lookup_retrievability(fake, refs)
        elapsed = time.perf_counter() - started
    finally:
        fake.release.set()
    assert 0.3 <= elapsed < 2.0, elapsed  # what was left of the deadline, not the lookup
    assert caught.value.resolved == [("unknown", None)] * 3
    threading.Event().wait(0.3)
    assert sorted(fake.calls) == ["10.1000/fail", "10.1000/slow1"]


@respx.mock
def test_the_lookup_is_capped_at_max_references(monkeypatch):
    monkeypatch.setattr(references, "MAX_REFERENCES", 2)
    for name in ("one", "two"):
        respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/{name}").mock(
            return_value=httpx.Response(200, json=_record(is_oa=False))
        )
    # No route for the third: a request for it would make respx raise.
    refs = _cited("one", "two", "three")
    assert lookup_retrievability(_unpaywall(), refs) == [
        ("paywalled", None),
        ("paywalled", None),
        ("unknown", None),
    ]


@respx.mock
def test_the_first_outage_cancels_the_rest_and_keeps_what_was_resolved(monkeypatch):
    """One DOI's registry failure ends the lookup: lookups not yet started
    are never requested, in-flight ones are allowed to finish, and the error
    carries every verdict that was reached so the caller can still show the
    list — flagged, never silently 'unknown'.

    The failing lookup is held until every other worker has a lookup in
    flight, so the failure lands with the pool full and two lookups queued.
    """
    _no_sleep(monkeypatch)
    in_flight = references.LOOKUP_WORKERS - 1
    started = 0
    lock = threading.Lock()
    all_started = threading.Event()

    def slow(request):
        nonlocal started
        with lock:
            started += 1
            if started == in_flight:
                all_started.set()
        # Hold the worker while the failure lands; not time.sleep, which the
        # retry backoff patch has replaced.
        threading.Event().wait(0.5)
        return httpx.Response(200, json=_record(url_for_pdf="https://x.org/slow.pdf"))

    def fail(request):
        assert all_started.wait(2), "the slow lookups never started"
        return httpx.Response(503)

    down = respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/down").mock(side_effect=fail)
    slow_route = respx.get(url__regex=rf"{UNPAYWALL_BASE}/v2/10\.1000/slow\d").mock(
        side_effect=slow
    )
    late = respx.get(url__regex=rf"{UNPAYWALL_BASE}/v2/10\.1000/late\d").mock(
        return_value=httpx.Response(200, json=_record(is_oa=False))
    )
    names = ["down", *(f"slow{i}" for i in range(in_flight)), "late1", "late2"]
    refs = _cited(*names)
    client = _unpaywall()
    try:
        with pytest.raises(RegistryUnavailable, match="HTTP 503") as caught:
            lookup_retrievability(client, refs)
    finally:
        client.close()
    assert down.call_count == 3  # initial + 2 retries, then the outage
    assert slow_route.call_count == in_flight
    assert late.call_count == 0  # queued behind the failure: never requested
    assert caught.value.resolved == [
        ("unknown", None),
        *[("pdf", "https://x.org/slow.pdf")] * in_flight,
        ("unknown", None),
        ("unknown", None),
    ]


@respx.mock
def test_the_failing_worker_itself_never_requests_the_next_queued_lookup(monkeypatch):
    """The `failed` flag exists for one race: the failing lookup's own worker
    returns to the queue and dequeues a pending lookup BEFORE the caller's
    thread reaches cancel_futures. The outage test above usually sees the
    caller win that race, so removing the flag survives it most runs.

    Here the race is decided, not timed: one worker, so the queued lookups
    sit behind the failing one; and as_completed is wrapped so the caller
    does not receive the failed future until the worker has finished with
    the NEXT queued lookup — skipped by the flag, or requested without it.
    Only then may the caller cancel. Either way the outcome is observable:
    with the flag, the late lookup's future is done with no request made.
    """
    _no_sleep(monkeypatch)
    monkeypatch.setattr(references, "LOOKUP_WORKERS", 1)
    real_as_completed = references.as_completed

    def as_completed_after_the_worker_moved_on(futures, timeout=None):
        by_index = {rows[0]: future for future, rows in futures.items()}
        for future in real_as_completed(futures, timeout=timeout):
            if future is by_index[0]:
                # The caller is late: the worker has dequeued late1 already.
                done, _ = wait([by_index[1]], timeout=5)
                assert done, "the worker never reached the queued lookup"
            yield future

    monkeypatch.setattr(references, "as_completed", as_completed_after_the_worker_moved_on)
    down = respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/down").mock(return_value=httpx.Response(503))
    late = respx.get(url__regex=rf"{UNPAYWALL_BASE}/v2/10\.1000/late\d").mock(
        return_value=httpx.Response(200, json=_record(is_oa=False))
    )
    refs = _cited("down", "late1", "late2")
    client = _unpaywall()
    try:
        with pytest.raises(RegistryUnavailable, match="HTTP 503") as caught:
            lookup_retrievability(client, refs)
    finally:
        client.close()
    assert down.call_count == 3
    assert late.call_count == 0, "a queued lookup was requested after the registry had failed"
    assert caught.value.resolved == [("unknown", None)] * 3


class _FakeUnpaywall:
    """A client whose `by_doi` answers, raises, or blocks per DOI, and records
    every call — the pooled lookup's contract without a socket."""

    def __init__(
        self, raising: dict[str, Exception] | None = None, blocking: set[str] = frozenset()
    ):
        self.raising = raising or {}
        self.blocking = blocking
        self.release = threading.Event()
        self.in_flight = threading.Event()  # set once a blocking lookup has begun
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def by_doi(self, doi: str) -> dict | None:
        with self._lock:
            self.calls.append(doi)
        name = doi.rsplit("/", 1)[-1]
        if name in self.raising:
            raise self.raising[name]
        if any(name.startswith(prefix) for prefix in self.blocking):
            self.in_flight.set()
            assert self.release.wait(10), f"{doi} was never released"
        return _record(url_for_pdf=f"https://x.org/{name}.pdf")

    def close(self) -> None:
        pass


@respx.mock
def test_a_decoding_error_from_the_registry_is_the_registry_failure_not_a_500(monkeypatch):
    """A response whose body does not match its Content-Encoding (a CDN error
    page behind a gzip header) raises httpx.DecodingError — a RequestError,
    not the TransportError the retry policy catches nor the RuntimeError the
    pool shielded. It is a registry failure like any other: the lookup ends,
    queued lookups are never requested, and the scan answers "unavailable"."""
    monkeypatch.setattr(references, "LOOKUP_WORKERS", 1)
    lie = respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/lie").mock(
        return_value=httpx.Response(
            503, headers={"content-encoding": "gzip"}, stream=httpx.ByteStream(b"{not gzip}")
        )
    )
    late = respx.get(url__regex=rf"{UNPAYWALL_BASE}/v2/10\.1000/late\d").mock(
        return_value=httpx.Response(200, json=_record(is_oa=False))
    )
    refs = _cited("lie", "late1", "late2")
    client = _unpaywall()
    try:
        with pytest.raises(RegistryUnavailable, match="DecodingError") as caught:
            lookup_retrievability(client, refs)
    finally:
        client.close()
    assert lie.call_count == 1
    assert late.call_count == 0
    assert caught.value.resolved == [("unknown", None)] * 3


@pytest.mark.parametrize(
    "failure",
    [httpx.InvalidURL("URL too long"), ValueError("no JSON object could be decoded")],
    ids=["InvalidURL", "ValueError"],
)
def test_any_lookup_failure_ends_the_lookup_like_an_outage(monkeypatch, failure):
    """httpx.InvalidURL (a request URL past httpx's own limit) and ValueError
    are neither TransportError nor RuntimeError; each is wrapped as the one
    registry failure, sets the flag the workers check, and cancels the queue."""
    monkeypatch.setattr(references, "LOOKUP_WORKERS", 1)
    fake = _FakeUnpaywall(raising={"down": failure})
    refs = _cited("down", "late1", "late2")
    with pytest.raises(RegistryUnavailable, match=type(failure).__name__) as caught:
        lookup_retrievability(fake, refs)
    assert fake.calls == ["10.1000/down"]
    assert str(failure) in str(caught.value)
    assert caught.value.resolved == [("unknown", None)] * 3


def test_an_escape_from_the_caller_thread_cancels_the_queued_lookups(monkeypatch):
    """A failure raised while READING a result (not in a worker) is not a
    registry failure and propagates — but it must not leave every queued
    lookup to run: the pool's teardown cancels the queue and the workers'
    flag stops the one dequeued in the meantime. One worker: `bad` answers,
    `late1` is in flight (blocked) when the caller fails, `late2` is queued."""
    monkeypatch.setattr(references, "LOOKUP_WORKERS", 1)
    fake = _FakeUnpaywall(blocking={"late"})
    real = references.retrievability

    def failing(record):
        if record and "bad.pdf" in (record["best_oa_location"]["url_for_pdf"] or ""):
            raise AttributeError("'list' object has no attribute 'get'")
        return real(record)

    monkeypatch.setattr(references, "retrievability", failing)
    real_as_completed = references.as_completed

    def as_completed_once_late1_is_in_flight(futures, timeout=None):
        # The race, decided: the caller reads `bad` only once the worker
        # has dequeued late1 and is inside its request.
        for future in real_as_completed(futures, timeout=timeout):
            assert fake.in_flight.wait(5), "the worker never reached late1"
            yield future

    monkeypatch.setattr(references, "as_completed", as_completed_once_late1_is_in_flight)
    refs = _cited("bad", "late1", "late2")
    # Released on a timer, never by the test body: a teardown that WAITS for
    # the in-flight lookup would otherwise hang the suite instead of failing.
    threading.Timer(0.5, fake.release.set).start()
    started = time.perf_counter()
    with pytest.raises(AttributeError):
        lookup_retrievability(fake, refs)
    assert time.perf_counter() - started < 0.5  # the caller is not held for the in-flight one
    assert fake.release.wait(5)
    threading.Event().wait(0.3)  # the released worker finishes late1 and tries the queue
    assert fake.calls == ["10.1000/bad", "10.1000/late1"]  # late2: cancelled, never requested


def test_registry_unavailable_is_a_runtime_error_carrying_the_partial_result():
    exc = RegistryUnavailable("Unpaywall gave no answer", [("unknown", None)])
    assert isinstance(exc, RuntimeError)
    assert str(exc) == "Unpaywall gave no answer"
    assert exc.resolved == [("unknown", None)]
