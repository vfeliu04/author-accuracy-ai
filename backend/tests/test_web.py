"""Web-page extraction: trafilatura body -> sections, the page's own markup -> metadata.

The fixtures under fixtures/web/ are shaped like real pages — site chrome,
consent banners, analytics scripts, CMS markup, entity-encoded attributes,
malformed JSON-LD — because surviving that noise is the extractor's whole job.
"""

import json
import logging
import time
from pathlib import Path

import pytest
import trafilatura

import authorai.web as web_mod
from authorai.web import (
    MIN_BODY_CHARS,
    PageMetadata,
    ThinPageError,
    _normalize_date,
    _page_metadata,
    _split_sections,
    _strip_emphasis,
    extract_web,
)

FIXTURES = Path(__file__).parent / "fixtures" / "web"

NEWS_URL = "https://www.globalwatermonitor.org/fact-sheets/detail/drought-and-water-scarcity"
SCHOLARLY_URL = "https://journals.hydrosynth.org/jhs/article/view/2024-0173"
REPORT_URL = "https://www.riverbasininstitute.org/reports/transboundary-water-2025/"
SPA_URL = "https://app.reservoirwatch.io/basins/upper-kessa"
DIARIO_URL = "https://www.diariodelagua.com/salud/desnutricion-cronica-infantil-america-latina"
BLOG_URL = "https://who-cares.com/posts/drought-myths/"

# The body call extract_web makes, minus the formatting switch under test.
BODY_ARGS = dict(
    output_format="markdown",
    include_tables=True,
    include_comments=False,
    include_images=False,
    include_links=False,
)


def _page(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _body(document) -> str:
    return "\n\n".join(section.text for section in document.sections)


def _assert_no_chrome(document, chrome: tuple[str, ...]) -> None:
    for section in document.sections:
        for text in chrome:
            assert text not in section.text, (section.title, text)


@pytest.fixture()
def web_log(caplog):
    """authorai loggers do not propagate to the root logger (log.setup_logger),
    so caplog's handler is attached to the module logger itself."""
    web_mod.logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        web_mod.logger.removeHandler(caplog.handler)


# --- (a) news / fact-sheet page ----------------------------------------------


def test_news_factsheet_keeps_the_article_and_drops_site_chrome():
    document, metadata = extract_web(_page("news_factsheet.html"), url=NEWS_URL)

    # The <h1> is followed directly by the first <h2> once the timestamp and
    # share links are dropped — its empty section is dropped with them.
    assert [s.title for s in document.sections] == [
        "Key facts",
        "Overview",
        "Who is affected",
        "Response",
    ]
    assert all(section.page is None for section in document.sections)
    assert document.tables == [] and document.figures == []
    _assert_no_chrome(
        document,
        (
            "Skip to main content",  # skip link
            "Health topics",  # primary nav
            "Data portal",
            "About us",
            "Fact sheets",  # breadcrumb nav
            "Share on social media",
            "Subscribe to our newsletter",  # footer
            "Privacy policy",
            "All rights reserved",
            "cookies",  # consent banner
            "Manage preferences",
        ),
    )
    assert metadata == PageMetadata(
        title="Drought & water scarcity: key facts",  # og:title, entity-decoded, collapsed
        authors=["Maria Gonzalez"],
        publisher="Global Water Monitor",
        publication_date="2025-03-14",
    )
    assert document.title == metadata.title


def test_emphasis_markers_are_stripped_so_quotes_match_verbatim():
    document, _ = extract_web(_page("news_factsheet.html"), url=NEWS_URL)
    sections = {section.title: section.text for section in document.sections}

    # Copied from the source paragraph carrying <em>, <strong> and <b>: chunk
    # text is later quoted as evidence and substring-checked, so a marker left
    # between these words would make a faithful quote fail.
    assert (
        "Drought is a prolonged period of abnormally low rainfall that leads to a shortage "
        "of water, and it affected 55 million people in 2024. Its impacts compound over "
        "time: crop failure, loss of livestock and the spread of water-borne disease."
    ) in sections["Overview"]
    assert "- An estimated 2.3 billion people live in countries" in sections["Key facts"]
    assert "classified as high or extremely high water stress" in sections["Who is affected"]
    for text in sections.values():
        assert "*" not in text and "_" not in text

    # The pipe table stays inline in its section, rows intact.
    lines = [line.rstrip() for line in sections["Who is affected"].splitlines()]
    start = lines.index("| Region | People in high water stress | Share of population |")
    assert lines[start + 1 : start + 5] == [
        "|---|---|---|",
        "| Sub-Saharan Africa | 430 million | 36% |",
        "| South Asia | 1.1 billion | 57% |",
        "| Middle East and North Africa | 390 million | 83% |",
    ]


def test_formatting_switch_cannot_replace_emphasis_stripping():
    """The path considered first: trafilatura 2.2.0's include_formatting=False
    does drop the emphasis markers, but it also drops the heading hashes and
    the paragraph breaks that sections are split on — so markers are stripped
    in post-processing instead. Pinned so an upgrade that changes either
    behavior is noticed."""
    page = _page("news_factsheet.html")

    plain = trafilatura.extract(page, url=NEWS_URL, include_formatting=False, **BODY_ARGS)
    assert "**" not in plain
    assert "| Sub-Saharan Africa | 430 million | 36% |" in plain
    assert not [line for line in plain.splitlines() if line.startswith("#")]
    assert "\n\n" not in plain

    formatted = trafilatura.extract(page, url=NEWS_URL, **BODY_ARGS)
    assert "## Who is affected" in formatted.splitlines()
    assert "**55 million people**" in formatted  # what extract_web must remove


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("affected **2.3 billion people** in 2024", "affected 2.3 billion people in 2024"),
        (
            "an *abnormally low* rate, __underlined__, _also italic_",
            "an abnormally low rate, underlined, also italic",
        ),
        (
            "***Bold italic lead.*** Then **bold *nested* text**",
            "Bold italic lead. Then bold nested text",
        ),
        ("rose **45**% to $**5** (*a*, *b*)", "rose 45% to $5 (a, b)"),
        ("*spans a\nsoft line break* here", "spans a\nsoft line break here"),
        ("- **Key finding:** rainfall fell", "- Key finding: rainfall fell"),
        # Literal characters inside words or numbers, or loose, are text.
        ("snake_case_name and file_name.py", "snake_case_name and file_name.py"),
        ("2*3*4 and 5 * 3 = 15 and Data* here", "2*3*4 and 5 * 3 = 15 and Data* here"),
        ("un**believ**able", "un**believ**able"),
        # Emphasis never spans a paragraph break or a table-cell boundary.
        ("*first paragraph\n\nsecond paragraph*", "*first paragraph\n\nsecond paragraph*"),
        ("| *note | x* |", "| *note | x* |"),
        (
            "| **Region** | Value |\n|---|---|\n| *Africa* | **650** |\n| a*b | c_d |",
            "| Region | Value |\n|---|---|\n| Africa | 650 |\n| a*b | c_d |",
        ),
        ("## Heading ##", "## Heading ##"),
        # A * never pairs with a _, and unequal runs pair only what they can.
        ("mixed _open and close* runs", "mixed _open and close* runs"),
        ("*lopsided** run", "lopsided* run"),
        # A run glued to a word, or loose after a space, closes nothing.
        ("call _internal_name now", "call _internal_name now"),
        ("*Estimated figure: 5 * 3 = 15", "*Estimated figure: 5 * 3 = 15"),
    ],
)
def test_strip_emphasis_removes_markers_without_touching_literal_text(markdown, expected):
    assert _strip_emphasis(markdown) == expected


@pytest.mark.parametrize("unit", ["*a ", " _a", "**a "])
def test_strip_emphasis_is_linear_on_unmatched_openers(unit):
    # The first implementation (a regex with lazy spans) re-scanned to the end
    # of the paragraph from every unmatched opener: 90 KB of "*a " took minutes.
    text = unit * 30_000
    started = time.perf_counter()
    assert _strip_emphasis(text) == text
    assert time.perf_counter() - started < 2.0


def test_large_jsonld_graph_is_ordered_in_linear_time():
    # Putting article nodes ahead of page nodes once used list-membership
    # tests: quadratic in dict comparisons (~2.5 s at 10,000 nodes, ~9x that here).
    graph = {
        "@graph": [
            {"@type": ["WebPage", "Article"], "@id": f"#node-{i}", "headline": f"Item {i}"}
            for i in range(30_000)
        ]
    }
    markup = _markup(_ld(graph))
    started = time.perf_counter()
    metadata = _page_metadata(markup, url="https://example.org/catalog")
    assert time.perf_counter() - started < 2.0
    assert metadata.title == "Item 0"


def test_sections_split_on_headings_but_never_inside_a_fenced_code_block():
    markdown = (
        "Intro before any heading, with **bold**.\n"
        "\n"
        "# Top ##\n"
        "\n"
        "## Empty\n"
        "\n"
        "## **Code** sample\n"
        "\n"
        "````\n"
        "```\n"
        "## still inside the fence\n"
        "**kwargs stays**\n"
        "````\n"
        "## After the fence, in C#\n"
        "Body with C# and #hashtag.\n"
        "\n"
        "####### seven hashes is text\n"
    )
    sections, first_heading = _split_sections(markdown)
    assert [(s.title, s.page, s.text) for s in sections] == [
        ("", None, "Intro before any heading, with bold."),
        ("Code sample", None, "````\n```\n## still inside the fence\n**kwargs stays**\n````"),
        (
            "After the fence, in C#",  # a hash glued to a word is not a closing sequence
            None,
            "Body with C# and #hashtag.\n\n####### seven hashes is text",
        ),
    ]
    # "Top" and "Empty" had no body and were dropped — the first heading is
    # still the document's first heading.
    assert first_heading == "Top"


# --- (b) scholarly article landing page --------------------------------------


def test_scholarly_landing_page_prefers_citation_tags_over_every_other_source():
    document, metadata = extract_web(_page("scholarly_article.html"), url=SCHOLARLY_URL)

    assert metadata == PageMetadata(
        title="Groundwater depletion under recurrent drought: evidence from 42 aquifers",
        authors=["Okafor, Chidi", "Lindqvist, Anna"],  # as printed, in order
        publisher="Hydrology Research Society",  # not JSON-LD's or og:site_name's
        publication_date="2024-06-03",  # Google Scholar's slash format, normalized
        doi="10.5555/jhs.2024.0173",  # the https://doi.org/ form, cleaned
        scholarly=True,
    )
    assert document.title == metadata.title
    assert [s.title for s in document.sections] == [
        "DOI:",
        "Abstract",
        "1. Introduction",
        "2. Data and methods",
        "3. Results",
        "References",
    ]
    body = _body(document)
    # The reference list's DOIs ARE in the body — and never become the page's.
    assert "https://doi.org/10.5555/wrr.2018.0456" in body
    assert "doi:10.5555/jge.2019.124001" in body
    assert "Recovery was incomplete in 31 of the 42 aquifers" in body
    _assert_no_chrome(
        document,
        (
            "Skip to main content",
            "Current Issue",
            "Submit an Article",
            "Editorial Board",
            "Make a Submission",
            "For Librarians",
            "Privacy Statement",
            "Platform and workflow",
            "cookies",
            "Accept All Cookies",
        ),
    )


def test_a_page_without_its_own_doi_never_borrows_one():
    page = _page("scholarly_article.html")
    tag = '<meta name="citation_doi" content="https://doi.org/10.5555/jhs.2024.0173"/>'
    assert tag in page
    _, metadata = extract_web(page.replace(tag, ""), url=SCHOLARLY_URL)
    # DOIs remain in the body, in citation_reference tags and in DC.Identifier —
    # none of which is a precedence source for the page's own DOI.
    assert metadata.doi is None
    assert metadata.scholarly


# --- (c) JSON-LD @graph + a malformed block ----------------------------------


def test_jsonld_graph_supplies_metadata_and_a_malformed_block_is_skipped_loudly(web_log):
    document, metadata = extract_web(_page("report_jsonld_graph.html"), url=REPORT_URL)

    assert metadata == PageMetadata(
        # The Report node's headline beats the WebPage node listed before it,
        # og:title and <title>.
        title="Transboundary Water Cooperation Report 2025",
        # Entity-decoded, deduplicated, and the Organization author is not a person.
        authors=["Leila Haddad", "Tomás Ruiz"],
        publisher="River Basin Institute",  # beats og:site_name "RBI"
        publication_date="2025-01-22",  # beats WebPage's date and article:published_time
        doi="10.5555/rbi.2025.014",  # the DOI PropertyValue, "doi:" prefix cleaned
    )
    warnings = [record for record in web_log.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1  # the BreadcrumbList's trailing comma, nothing else
    assert REPORT_URL in warnings[0].getMessage()

    # trafilatura drops the entry header's <h1>, so the lead paragraph comes
    # before any heading and carries the empty title.
    assert [s.title for s in document.sections] == [
        "",
        "Key findings",
        "Basin-by-basin results",
        "Recommendations",
    ]
    assert document.sections[0].text.startswith("Shared rivers supply water")
    assert document.title == "Transboundary Water Cooperation Report 2025"
    assert "| Upper Kessa | 3 | 7.5 |" in document.sections[2].text
    _assert_no_chrome(
        document,
        (
            "Skip to content",
            "Independent research on shared waters",
            "Our research",
            "Donate",
            "You may also like",
            "Sign up for our monthly briefing",
            "Privacy Policy",
            "Registered charity",
            "Manage Cookie Consent",
            "cookies",
        ),
    )


def _markup(*head: str) -> str:
    return "<!DOCTYPE html><html><head>" + "\n".join(head) + "</head><body><p>x</p></body></html>"


def _ld(data) -> str:
    return f'<script type="application/ld+json">{json.dumps(data)}</script>'


def test_jsonld_top_level_list_with_plain_string_fields():
    markup = _markup(
        _ld(
            [
                {"@type": "WebSite", "name": "Daily Ledger"},  # not an article or page type
                {
                    "@type": "NewsArticle",
                    "headline": "Rivers run low",
                    "author": "Jane Roe",
                    "publisher": "Daily Ledger",
                    "datePublished": "2024-3-5",
                },
            ]
        )
    )
    assert _page_metadata(markup, url="https://example.org/n") == PageMetadata(
        title="Rivers run low",
        authors=["Jane Roe"],
        publisher="Daily Ledger",
        publication_date="2024-03-05",
    )


def test_jsonld_id_references_resolve_within_the_graph():
    graph = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "Article",
                "headline": "Aquifers under pressure",
                "author": {"@id": "#person-1"},
                "publisher": {"@id": "#org"},
                "sameAs": ["https://x.com/ledger", "https://doi.org/10.5555/aq.7"],
            },
            {"@type": "Person", "@id": "#person-1", "name": "Ann Lee"},
            {"@type": "Organization", "@id": "#org", "name": "Ledger Media"},
        ],
    }
    metadata = _page_metadata(_markup(_ld(graph)), url="https://example.org/a")
    assert (metadata.authors, metadata.publisher, metadata.doi) == (
        ["Ann Lee"],
        "Ledger Media",
        "10.5555/aq.7",
    )


def test_first_valid_doi_wins_and_a_malformed_one_becomes_none():
    malformed = '<meta name="citation_doi" content="doi: not-a-doi">'
    jsonld = _ld(
        {"@type": "ScholarlyArticle", "headline": "X", "identifier": "https://doi.org/10.5555/x.9"}
    )
    assert _page_metadata(_markup(malformed, jsonld), url="https://example.org/d").doi == (
        "10.5555/x.9"
    )
    assert _page_metadata(_markup(malformed), url="https://example.org/d").doi is None


def test_empty_values_fall_through_and_meta_author_is_one_string():
    metadata = _page_metadata(
        _markup(
            '<meta name="citation_title" content="   ">',
            '<meta property="og:site_name" content="">',
            '<meta name="author" content="First Author">',
            '<meta name="author" content="Second Author">',
            '<meta name="citation_date" content="2019">',
            "<title>Rivers &amp;  Lakes\n</title>",
        ),
        url="https://example.org/e",
    )
    assert metadata == PageMetadata(
        title="Rivers & Lakes",
        authors=["First Author"],
        publication_date="2019",
        scholarly=True,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024", "2024"),
        ("2024-3", "2024-03"),
        ("2024/06/03", "2024-06-03"),
        ("2025-03-14T09:30:00+01:00", "2025-03-14"),
        ("2025-01-22 07:15:00 +0000", "2025-01-22"),
        ("March 14, 2025", "March 14, 2025"),  # not ISO-shaped: kept as printed
        ("2024-02-30", "2024-02-30"),  # not a real date: kept, never guessed
        ("2024-13", "2024-13"),
    ],
)
def test_publication_dates_normalize_only_when_they_parse(raw, expected):
    assert _normalize_date(raw) == expected


# --- (d) JavaScript-only shell -------------------------------------------------


def test_javascript_shell_raises_thin_page_error_naming_the_url():
    with pytest.raises(ThinPageError) as excinfo:
        extract_web(_page("spa_shell.html"), url=SPA_URL)
    message = str(excinfo.value)
    assert SPA_URL in message
    assert "no readable article text" in message
    assert "JavaScript-only pages are not supported" in message
    assert isinstance(excinfo.value, ValueError)


def test_thin_page_floor_is_measured_on_section_text(monkeypatch):
    def serve(markdown):
        monkeypatch.setattr(web_mod.trafilatura, "extract", lambda *args, **kwargs: markdown)

    serve("# Title only\n\n" + "a" * MIN_BODY_CHARS)
    document, _ = extract_web("<html></html>", url="https://example.org/enough")
    assert [len(section.text) for section in document.sections] == [MIN_BODY_CHARS]

    thin_bodies = (
        "# Title only\n\n" + "a" * (MIN_BODY_CHARS - 1),  # headings do not count
        "**" + "a" * (MIN_BODY_CHARS - 2) + "**",  # neither do stripped markers
        "# A heading\n\n## Another heading\n",
        "",
        None,  # trafilatura found nothing at all
    )
    for markdown in thin_bodies:
        serve(markdown)
        with pytest.raises(ThinPageError, match="https://example.org/thin"):
            extract_web("<html></html>", url="https://example.org/thin")


# --- (e) encodings -------------------------------------------------------------


def test_utf8_bytes_page_keeps_non_ascii_text_intact():
    raw = (FIXTURES / "diario_agua_es.html").read_bytes()
    assert b'<meta charset="utf-8">' in raw

    document, metadata = extract_web(raw, url=DIARIO_URL)

    assert [s.title for s in document.sections] == [
        "Desnutrición crónica infantil baja en América Latina, según la OMS",
        "El Corredor Seco, en alerta",
    ]
    body = _body(document)
    assert "La Organización Mundial de la Salud informó que la desnutrición crónica" in body
    assert "según el informe regional publicado este viernes en São Paulo." in body
    assert metadata == PageMetadata(
        # og:title carries &oacute; entities; the result must equal the UTF-8 text.
        title="Desnutrición crónica infantil baja en América Latina, según la OMS",
        authors=["Lucía Fernández Ibáñez"],
        publisher="Diario del Agua",
        publication_date="2025-02-07",
    )
    _assert_no_chrome(document, ("Portada", "Suscríbete", "cookies", "derechos reservados"))
    assert (document, metadata) == extract_web(raw.decode("utf-8"), url=DIARIO_URL)


def test_declared_legacy_charset_decodes_like_a_browser():
    # Servers still label windows-1252 pages "iso-8859-1"; browsers decode that
    # label as windows-1252, where 0x93/0x94 are curly quotes, not C1 controls.
    page = _page("diario_agua_es.html").replace(
        '<meta charset="utf-8">',
        '<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">',
    )
    raw = page.encode("cp1252")
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")

    document, metadata = extract_web(raw, url=DIARIO_URL)

    assert "“La reducción es real, pero frágil”" in _body(document)
    assert metadata.authors == ["Lucía Fernández Ibáñez"]


def test_undeclared_non_utf8_bytes_fail_loudly_naming_the_url():
    page = _page("diario_agua_es.html").replace('<meta charset="utf-8">', "")
    with pytest.raises(ValueError) as excinfo:
        extract_web(page.encode("cp1252"), url=DIARIO_URL)
    assert DIARIO_URL in str(excinfo.value)
    assert not isinstance(excinfo.value, ThinPageError)


def test_unknown_declared_charset_fails_loudly_naming_the_url():
    page = _page("diario_agua_es.html").replace(
        '<meta charset="utf-8">', '<meta charset="x-legacy-klingon">'
    )
    with pytest.raises(ValueError, match="x-legacy-klingon") as excinfo:
        extract_web(page.encode("cp1252"), url=DIARIO_URL)
    assert DIARIO_URL in str(excinfo.value)


def test_declared_utf8_with_a_stray_invalid_byte_decodes_with_a_warning(web_log):
    raw = (FIXTURES / "diario_agua_es.html").read_bytes()
    damaged = raw.replace("São Paulo".encode(), b"S\xe3o Paulo")
    assert damaged != raw

    document, _ = extract_web(damaged, url=DIARIO_URL)

    assert "S�o Paulo" in _body(document)
    assert [record for record in web_log.records if DIARIO_URL in record.getMessage()]


# --- (f) hostname is not a publisher -------------------------------------------


def test_hostname_never_becomes_the_publisher():
    document, metadata = extract_web(_page("personal_blog.html"), url=BLOG_URL)

    # "who-cares.com" is the hostname, the header, the footer and the <title>
    # suffix, and the body names the World Health Organization — none of it is
    # publisher metadata. The printed "February 3, 2025" is not a date tag.
    assert metadata == PageMetadata(
        title="Five myths about drought I keep hearing",
        authors=["Sam Patel"],
    )
    assert [s.title for s in document.sections] == [
        "Five myths about drought I keep hearing",
        "Myth 1: drought only matters for farmers",
        "Myth 2: one wet winter ends a drought",
    ]
    _assert_no_chrome(document, ("About me", "Built with Hugo", "RSS", "4 min read"))


# --- cross-cutting ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "url"),
    [("news_factsheet.html", NEWS_URL), ("report_jsonld_graph.html", REPORT_URL)],
)
def test_extraction_is_deterministic(name, url):
    page = _page(name)
    assert extract_web(page, url=url) == extract_web(page, url=url)


def test_document_title_falls_back_to_first_heading_then_none(monkeypatch):
    body = "a" * MIN_BODY_CHARS
    monkeypatch.setattr(
        web_mod.trafilatura,
        "extract",
        lambda *args, **kwargs: f"# **First** heading\n\n## Second\n\n{body}",
    )
    document, metadata = extract_web("<html><head></head></html>", url="https://example.org/a")
    assert metadata.title is None
    assert document.title == "First heading"

    monkeypatch.setattr(web_mod.trafilatura, "extract", lambda *args, **kwargs: body)
    document, _ = extract_web("<html></html>", url="https://example.org/b")
    assert document.title is None
