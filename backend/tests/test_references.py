"""Bibliography scan (references.py): closing-page text, the reference models,
the Haiku call, the Unpaywall client and retrievability — all offline (respx
for HTTP, FakeLLM for the model, a hand-built PDF for pypdf)."""

import io
import json

import pytest

from authorai import references
from authorai.references import (
    MAX_REFERENCES,
    REFERENCE_MAX_CHARS,
    REFERENCES_SYSTEM,
    Reference,
    ReferenceList,
    extract_references,
    read_pages,
    reference_text,
)
from tests.conftest import FakeLLM, pdf_with_pages

ENTRY = "Smith, J. (2020). Water stress and cities. Journal of Hydrology, 12(3), 1-9."


# --- reference_text: where the bibliography is ---------------------------------


def test_slices_from_the_last_references_heading():
    """A report can print 'References' more than once (a chapter's own list, a
    running header); the bibliography is the LAST one."""
    pages = [
        "Intro. See References for details.",
        "References\nChapter 1's own short list.",
        "Body text continues.",
        "References\n" + ENTRY,
    ]
    text, source = reference_text(pages)
    assert source == "heading"
    assert text == "References\n" + ENTRY


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


def test_extract_references_sends_the_closing_text_under_the_references_contract():
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
    call = llm.parse_calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["system"] == REFERENCES_SYSTEM
    assert ENTRY in call["prompt"]
    assert call["output_type"] is ReferenceList
    assert call["images"] is None


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


def test_extract_references_returns_a_short_list_untouched(references_log):
    two = ReferenceList(references=[Reference(entry="a"), Reference(entry="b")])
    assert extract_references(FakeLLM({ReferenceList: two}), "m", "References") == two
    assert references_log.text == ""


def test_blank_optional_fields_read_as_absent():
    """The model sometimes prints '' for a field it was told to leave null;
    downstream code (the DOI lookup, the printed-URL rule) keys on absence."""
    reference = Reference(entry="  x  ", title="  ", doi="", url=" \n", authors=["A", "  ", "B"])
    assert reference.entry == "x"
    assert reference.title is None
    assert reference.doi is None
    assert reference.url is None
    assert reference.authors == ["A", "B"]


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
    assert references.MAX_REFERENCES == 80
    assert references.REFERENCE_MAX_PAGES == 600
