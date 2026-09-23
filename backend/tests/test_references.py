"""Bibliography scan (references.py): closing-page text, the reference models,
the Haiku call, the Unpaywall client and retrievability — all offline (respx
for HTTP, FakeLLM for the model, a hand-built PDF for pypdf)."""

import io
import json
import re
import threading
from concurrent.futures import wait

import httpx
import pytest
import respx

from authorai import references
from authorai.llm import PARSE_MAX_TOKENS
from authorai.references import (
    ENTRY_PREFIX_CHARS,
    HEADING_RUN_PAGES,
    MAX_REFERENCES,
    REFERENCE_CHUNK_CHARS,
    REFERENCE_MAX_CHARS,
    REFERENCES_SYSTEM,
    UNPAYWALL_BASE,
    Reference,
    ReferenceList,
    RegistryUnavailable,
    UnpaywallClient,
    extract_references,
    lookup_retrievability,
    printed_url,
    read_pages,
    reference_text,
    retrievability,
    split_reference_text,
)
from tests.conftest import FakeLLM, pdf_with_pages

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
    text, source = reference_text(pages, max_chars=100)
    assert source == "heading"
    assert text == "References\n" + ENTRY


def test_a_running_header_does_not_cut_a_multi_page_bibliography_short():
    """Each page of a long bibliography carries 'References' as a running
    header. The LAST match is on the last page; slicing from it alone would
    lose every earlier entry. Headings at most HEADING_RUN_PAGES pages apart
    are one section, so the slice starts at the earliest of them."""
    first = "References\n" + ENTRY
    second = "45\nReferences\n" + SECOND_ENTRY
    text, source = reference_text([first, second])
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
    text, source = reference_text(pages)
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
    text, source = reference_text([chapter_list, sparse_page, sparse_page, sparse_page, final_list])
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
    text, source = reference_text(pages)
    assert source == "heading"
    assert text == "\n".join(pages)


def test_headings_one_page_more_than_a_run_apart_are_two_sections():
    """One page past the run and the earlier heading is a separate section:
    the slice starts at the last heading."""
    pages = _pages_with_headings_apart(HEADING_RUN_PAGES + 1)
    text, source = reference_text(pages)
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
    text, source = reference_text(pages)
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
    text, source = reference_text(pages)
    assert source == "heading"
    assert text == "\n".join(pages[10:])


def test_a_contents_page_mention_far_earlier_stays_excluded():
    """A table of contents prints 'References' on a line of its own, many
    pages before the bibliography: more than HEADING_RUN_PAGES pages back
    from the last heading, so the slice still starts at the bibliography."""
    pages = ["Contents\nReferences\n45", *[BODY] * 10, "References\n" + ENTRY]
    text, source = reference_text(pages)
    assert source == "heading"
    assert text == "References\n" + ENTRY


def test_the_slice_and_its_cap_start_at_the_heading_word():
    """Blank lines and indentation before the heading are matched by the
    pattern but are not bibliography: they do not count against the cap."""
    text, source = reference_text(["Body", "\n\n   References\n" + "x" * 100], max_chars=20)
    assert source == "heading"
    assert text == ("References\n" + "x" * 100)[:20]


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
    ],
)
def test_every_heading_form_is_recognised(heading):
    text, source = reference_text(["Body text.", heading + "\n" + ENTRY])
    assert source == "heading"
    assert text.startswith(heading.strip())
    assert text.endswith(ENTRY)


def test_a_mid_line_mention_is_not_a_heading():
    """Only a line that IS the heading counts — 'see References' in prose is
    not where the bibliography starts."""
    text, source = reference_text(["Intro. See References for details.", ENTRY])
    assert source == "tail"


def test_no_heading_falls_back_to_the_document_tail():
    text, source = reference_text(["A" * 100, "B" * 100], max_chars=50)
    assert source == "tail"
    assert text == "B" * 50


def test_the_heading_slice_is_capped_from_the_heading_forward():
    text, source = reference_text(["References\n" + "x" * 100], max_chars=20)
    assert source == "heading"
    assert text == ("References\n" + "x" * 100)[:20]


def test_the_default_cap_is_the_code_constant():
    text, source = reference_text(["y" * (REFERENCE_MAX_CHARS + 500)])
    assert source == "tail"
    assert len(text) == REFERENCE_MAX_CHARS


def test_no_text_at_all_means_no_model_call():
    """A scanned PDF yields no text; the answer is 'none', never an invented
    bibliography — the endpoint skips the model on this value."""
    assert reference_text([]) == ("", "none")
    assert reference_text(["", "  \n\t"]) == ("", "none")


# --- read_pages: pypdf over the spooled upload ---------------------------------


def test_read_pages_extracts_the_text_of_each_page():
    pdf = pdf_with_pages(["Page one", "Page two", "References\n" + ENTRY])
    pages = read_pages(io.BytesIO(pdf))
    assert [page.strip() for page in pages] == ["Page one", "Page two", "References\n" + ENTRY]


def test_read_pages_reads_only_the_last_max_pages():
    pdf = pdf_with_pages(["one", "two", "three"])
    assert [p.strip() for p in read_pages(io.BytesIO(pdf), max_pages=2)] == ["two", "three"]


def test_read_pages_keeps_a_textless_page_as_an_empty_string():
    assert [p.strip() for p in read_pages(io.BytesIO(pdf_with_pages([""])))] == [""]


@pytest.mark.parametrize(
    "junk",
    [b"", b"not a pdf", b"%PDF-1.4 fake pdf content", b"%PDF-1.7\n1 0 obj\n<< >>\nendobj\n"],
)
def test_read_pages_turns_any_pypdf_failure_into_a_value_error(junk):
    """The endpoint answers 400 for what it cannot read; pypdf's own exception
    family is not its contract."""
    with pytest.raises(ValueError, match="could not read"):
        read_pages(io.BytesIO(junk))


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
    assert [p.strip() for p in read_pages(encrypted)] == ["Restricted references"]


# --- the models and the call ----------------------------------------------------


def _list(*entries: str) -> ReferenceList:
    return ReferenceList(references=[Reference(entry=entry) for entry in entries])


def _lines(count: int) -> str:
    """A reference list of `count` one-line entries, numbered so a test can
    tell which lines a chunk carried; 340 lines is ~30,000 characters."""
    return "\n".join(f"Ref {i:04d}. {ENTRY}" for i in range(count))


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
    assert result == wanted
    assert len(llm.parse_calls) == 1
    call = llm.parse_calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["system"] == REFERENCES_SYSTEM
    assert call["prompt"] == "CLOSING PAGES:\n\nReferences\n" + ENTRY
    assert "part" not in call["prompt"]
    assert call["output_type"] is ReferenceList
    assert call["images"] is None


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
    text = _lines(340)
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
    text = _lines(340)
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
    text = _lines(480)
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
    assert extract_references(FakeLLM({ReferenceList: two}), "m", "References") == two
    assert references_log.text == ""


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
def test_the_lookup_is_capped_at_max_references(monkeypatch):
    monkeypatch.setattr(references, "MAX_REFERENCES", 2)
    for name in ("one", "two"):
        respx.get(f"{UNPAYWALL_BASE}/v2/10.1000/{name}").mock(
            return_value=httpx.Response(200, json=_record(is_oa=False))
        )
    # No route for the third: a request for it would make respx raise.
    refs = [Reference(entry=n, doi=f"10.1000/{n}") for n in ("one", "two", "three")]
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
    refs = [Reference(entry=n, doi=f"10.1000/{n}") for n in names]
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

    def as_completed_after_the_worker_moved_on(futures):
        by_index = {index: future for future, index in futures.items()}
        for future in real_as_completed(futures):
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
    refs = [Reference(entry=n, doi=f"10.1000/{n}") for n in ("down", "late1", "late2")]
    client = _unpaywall()
    try:
        with pytest.raises(RegistryUnavailable, match="HTTP 503") as caught:
            lookup_retrievability(client, refs)
    finally:
        client.close()
    assert down.call_count == 3
    assert late.call_count == 0, "a queued lookup was requested after the registry had failed"
    assert caught.value.resolved == [("unknown", None)] * 3


def test_registry_unavailable_is_a_runtime_error_carrying_the_partial_result():
    exc = RegistryUnavailable("Unpaywall gave no answer", [("unknown", None)])
    assert isinstance(exc, RuntimeError)
    assert str(exc) == "Unpaywall gave no answer"
    assert exc.resolved == [("unknown", None)]
