"""Bibliography scan (references.py): closing-page text, the reference models,
the Haiku call, the Unpaywall client and retrievability — all offline (respx
for HTTP, FakeLLM for the model, a hand-built PDF for pypdf)."""

import io
import json
import threading

import httpx
import pytest
import respx

from authorai import references
from authorai.references import (
    MAX_REFERENCES,
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
    ):
        assert _unpaywall().by_doi(bad) is None, bad
    assert route.call_count == 0, [str(c.request.url) for c in route.calls]


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


def test_registry_unavailable_is_a_runtime_error_carrying_the_partial_result():
    exc = RegistryUnavailable("Unpaywall gave no answer", [("unknown", None)])
    assert isinstance(exc, RuntimeError)
    assert str(exc) == "Unpaywall gave no answer"
    assert exc.resolved == [("unknown", None)]
