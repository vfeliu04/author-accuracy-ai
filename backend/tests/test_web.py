"""Web-page extraction: trafilatura body -> sections, the page's own markup -> metadata.

The fixtures under fixtures/web/ are shaped like real pages — site chrome,
consent banners, paywalls, analytics scripts, CMS markup, entity-encoded
attributes, malformed JSON-LD, nested tables, footnote markers — because
surviving that noise is the extractor's whole job.
"""

import dataclasses
import json
import logging
import os
import subprocess
import sys
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
    _prepare_tree,
    extract_web,
)

FIXTURES = Path(__file__).parent / "fixtures" / "web"

NEWS_URL = "https://www.globalwatermonitor.org/fact-sheets/detail/drought-and-water-scarcity"
SCHOLARLY_URL = "https://journals.hydrosynth.org/jhs/article/view/2024-0173"
REPORT_URL = "https://www.riverbasininstitute.org/reports/transboundary-water-2025/"
SPA_URL = "https://app.reservoirwatch.io/basins/upper-kessa"
DIARIO_URL = "https://www.diariodelagua.com/salud/desnutricion-cronica-infantil-america-latina"
BLOG_URL = "https://who-cares.com/posts/drought-myths/"
BULLETIN_URL = "https://www.hydrology.example.gov/bulletins/groundwater-drought-2025-03"
PAYWALL_URL = "https://www.valleycourier.example/news/2025/04/11/aquifer-collapse"
ZH_URL = "https://www.shuili.example.cn/guoji/2025-03/20/c_1130.htm"
STATS_URL = "https://www.drought-observatory.example.org/statistics/2023-2024"

# The body options extract_web passes to trafilatura, minus the formatting switch.
BODY_ARGS = dict(
    output_format="markdown",
    include_tables=True,
    include_comments=False,
    include_images=False,
    include_links=False,
)

PROSE = (
    "Rainfall in the upper basin fell by a third in 2024 compared with the long-term "
    "average, and reservoir operators cut releases to the lower valley twice. "
)


def _page(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _body(document) -> str:
    return "\n\n".join(section.text for section in document.sections)


def _sections(document) -> dict[str, str]:
    return {section.title: section.text for section in document.sections}


def _assert_no_chrome(document, chrome: tuple[str, ...]) -> None:
    for section in document.sections:
        for text in chrome:
            assert text not in section.text, (section.title, text)


def _prose(length: int) -> str:
    """Real sentences cut to exactly `length` characters, never ending in a space."""
    text = (PROSE * (length // len(PROSE) + 1))[:length]
    return text if not text.endswith(" ") else text[:-1] + "."


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
    sections = _sections(document)

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


def test_emphasis_is_removed_from_the_page_tree_not_by_either_rejected_path():
    """Three ways to keep emphasis markers out of quoted text were tried.

    1. trafilatura's include_formatting=False drops the markers, but also the
       blank lines between paragraphs (which chunking splits on) — rejected.
    2. Stripping markers from the markdown afterwards cannot work: the markdown
       does not escape literal asterisks, so a page's own "(*)" and a marker
       glued to a number ("US$**2.3**bn") are the same characters — rejected
       after it both left glued markers in and ate literal ones.
    3. Unwrapping <b>/<strong>/<i>/<em>/<u> in the parsed page before extraction
       — what extract_web does. Pinned so an upgrade that changes 1 or 2 is noticed.
    """
    news = _page("news_factsheet.html")
    plain = trafilatura.extract(news, url=NEWS_URL, include_formatting=False, **BODY_ARGS)
    assert "**" not in plain
    assert "| Sub-Saharan Africa | 430 million | 36% |" in plain
    assert "\n\n" not in plain

    bulletin = _page("hydrology_bulletin.html")
    markdown = trafilatura.extract(bulletin, url=BULLETIN_URL, **BODY_ARGS)
    assert "US$**2.3**bn" in markdown  # a marker glued to digits...
    assert "cases (*) and suspected cases (**)" in markdown  # ...and the page's own asterisks

    document, _ = extract_web(bulletin, url=BULLETIN_URL)
    body = _body(document)
    assert "The programme cost US$2.3bn and cut losses 10x faster than planned." in body
    assert "Confirmed cases (*) and suspected cases (**) of water-borne disease" in body
    assert "\n\n" in _sections(document)["Household survey"]


def test_large_jsonld_graph_is_ordered_in_linear_time():
    # Choosing the work node once used list-membership tests: quadratic in
    # dict comparisons (~2.5 s at 10,000 nodes, ~9x that here).
    graph = {
        "@graph": [
            {"@type": ["WebPage", "Article"], "@id": f"#node-{i}", "headline": f"Item {i}"}
            for i in range(30_000)
        ]
    }
    graph["@graph"][0]["url"] = "https://example.org/catalog"  # the one node naming the page
    markup = _markup(_ld(graph))
    started = time.perf_counter()
    metadata = _page_metadata(markup, url="https://example.org/catalog")
    assert time.perf_counter() - started < 2.0
    assert metadata.title == "Item 0"


# --- body fidelity: a technical bulletin with every awkward construct ---------


def test_bulletin_sections_follow_the_page_headings_and_drop_its_chrome():
    document, metadata = extract_web(_page("hydrology_bulletin.html"), url=BULLETIN_URL)

    assert [s.title for s in document.sections] == [
        "Groundwater and drought bulletin: March 2025",
        "Household survey",
        "Regional figures",
        "Energy and emissions",
        "Data access",
        "Results for 31 aquifers",  # the heading's <em> is unwrapped too
    ]
    _assert_no_chrome(
        document,
        (
            "Skip to main content",
            "About the service",
            "Close menu",  # the menu button's inline <svg><title>
            "Open Government Licence",
            "Accessibility statement",
            "analytics cookies",  # consent notice outside <main>
        ),
    )
    # A <meta> carrying both name= and property= counts under both keys.
    assert metadata == PageMetadata(
        title="Groundwater and drought bulletin: March 2025",  # og:title via property=
        publisher="National Hydrology Service",
        publication_date="2025-04-02",  # article:published_time via property=
    )


def test_a_paragraph_starting_with_a_hash_is_text_not_a_heading():
    document, _ = extract_web(_page("hydrology_bulletin.html"), url=BULLETIN_URL)
    survey = _sections(document)["Household survey"]
    assert survey.startswith(
        "# of households without piped water: 1,240 in 2024, up from 860 in 2020.\n\n"
        "The survey covered 18 districts."
    )


def test_a_backtick_paragraph_and_a_code_block_never_swallow_headings():
    document, _ = extract_web(_page("hydrology_bulletin.html"), url=BULLETIN_URL)
    sections = _sections(document)

    data_access = sections["Data access"]
    assert "```yaml is how the portal expects the data dictionary to start." in data_access
    assert "## not a heading\nloader = Loader(**kwargs)" in data_access  # code kept verbatim
    assert "## not a heading" not in [s.title for s in document.sections]
    # The unclosed-looking ``` paragraph did not open a fence over the rest of the page.
    assert sections["Results for 31 aquifers"].startswith(
        "Recovery was incomplete in 31 of 42 aquifers after the most recent drought, and "
        "median storage loss reached 0.41 metres per year"
    )


def test_emphasis_glued_to_letters_digits_and_punctuation_is_removed():
    document, _ = extract_web(_page("hydrology_bulletin.html"), url=BULLETIN_URL)
    body = _body(document)
    assert "The programme cost US$2.3bn and cut losses 10x faster than planned." in body
    assert "Status: Emergency(provisional) declared in 14 districts." in body


def test_literal_asterisks_underscores_and_code_spans_survive():
    document, _ = extract_web(_page("hydrology_bulletin.html"), url=BULLETIN_URL)
    body = _body(document)
    assert "Confirmed cases (*) and suspected cases (**) of water-borne disease" in body
    assert (
        "Configure the loader through the `__init__` method; column names use snake_case "
        "such as well_id, and the grid is 2*3 cells."
    ) in body


def test_nested_and_rowspan_tables_keep_every_value_in_its_row():
    document, _ = extract_web(_page("hydrology_bulletin.html"), url=BULLETIN_URL)
    lines = [line.rstrip() for line in _sections(document)["Regional figures"].splitlines()]

    rowspan = lines.index("| Region | 2023 | 2024 |")
    assert lines[rowspan + 1 : rowspan + 4] == [
        "|---|---|---|",
        "| East Africa | 20.3 | 23.0 |",
        "|  | 21.0 | 24.5 |",  # the spanned cell keeps its column
    ]
    # trafilatura cut a nested table's row loose from its cell ("| Sahel |  |",
    # then "| Niger | 4.4 million |" after the Horn row). The inner table is
    # flattened into its cell first, so every value stays in its own row.
    nested = lines.index("| Sahel | Niger, 4.4 million |")
    assert lines[nested + 1] == "| Horn | 23 million |"
    assert "| Indicator | Value |" in lines


def test_a_table_heavy_page_keeps_every_table_and_rowspan_cell():
    # With more tables than paragraphs, trafilatura's default cascade swaps its
    # own extraction for readability's, which dropped the rowspan placeholder
    # (21.0 slid under "Region") and, on pages with less prose, whole tables.
    document, metadata = extract_web(_page("drought_statistics.html"), url=STATS_URL)

    lines = document.sections[0].text.splitlines()
    assert [line.rstrip() for line in lines if line.startswith("|")] == [
        "| Region | 2023 | 2024 |",
        "|---|---|---|",
        "| East Africa | 20.3 | 23.0 |",
        "|  | 21.0 | 24.5 |",
        "| Sahel | Niger, 4.4 million |",
        "| Horn | 23 million |",
        "| Indicator | Value |",
        "|---|---|",
        "| Wells tested (pass \\| fail) | 312 \\| 48 |",
    ]
    assert all(line == line.lstrip() for line in lines)  # no re-indented rows
    assert "Source: national hydrological services" in document.sections[0].text
    _assert_no_chrome(document, ("Methodology", "Terms of use", "Privacy"))
    assert metadata.title == "Drought impact statistics, 2023–2024"


def test_a_layout_only_the_full_cascade_reads_is_not_thin():
    # Fast mode runs first (it keeps tables intact); its own extractor reads
    # little of this legacy <center><font> page, so the full cascade runs.
    links = "".join(
        f'<a href="/{slug}.html">{label}</a> | '
        for slug, label in (
            ("index", "Home"),
            ("reports", "Reports"),
            ("data", "Data"),
            ("staff", "Staff"),
            ("links", "Links"),
        )
    )
    page = (
        "<html><head><title>Upper basin rainfall, 2024</title></head>"
        '<body bgcolor="#FFFFFF"><center><font face="Arial" size="2">'
        + (PROSE + "<p>") * 4
        + f"</font></center><hr><center><font size=1>{links}</font></center></body></html>"
    )
    document, _ = extract_web(page, url="https://www.basin-authority.example/rain2024.html")
    assert _body(document).count("Rainfall in the upper basin fell by a third in 2024") >= 3


def test_a_pipe_inside_a_table_cell_stays_escaped_so_the_row_keeps_its_columns():
    # Policy: a literal "|" in a cell stays as trafilatura's GFM escape "\|".
    # Unescaped, "pass | fail" would read as two cells and shift "312 | 48"
    # into columns that do not exist; a value's column is part of its meaning.
    document, _ = extract_web(_page("hydrology_bulletin.html"), url=BULLETIN_URL)
    lines = [line.rstrip() for line in _sections(document)["Regional figures"].splitlines()]
    assert "| Wells tested (pass \\| fail) | 312 \\| 48 |" in lines


def test_superscripts_subscripts_and_footnote_markers():
    # Policy: digits and signs are written in plain text, a superscript as ^6, so
    # 10<sup>6</sup> stays a million instead of reading "106" and a quote, a claim
    # or a keyword search spelled "CO2" matches (the verdict quote check does not
    # fold Unicode sub/superscripts); a superscript that holds a link is a footnote
    # marker and is dropped; anything else is kept as plain text.
    document, _ = extract_web(_page("hydrology_bulletin.html"), url=BULLETIN_URL)
    assert (
        "Pumping raised CO2 emissions 5% over an irrigated area of 4.2 km^2, and about "
        "10^6 m^3 of water was lifted each day."
    ) in _sections(document)["Energy and emissions"]
    assert "<sub>" not in _body(document) and "<sup>" not in _body(document)


# Numeric references to code points XML forbids, in the spellings pages use.
CONTROL_REFERENCES = ["&#12;", "&#11;", "&#x0B;", "&#8", "&#0000031;", "&#x1B;", "&#xFFFE;"]


@pytest.mark.parametrize("reference", CONTROL_REFERENCES)
def test_a_control_character_reference_in_article_or_chrome_does_not_fail_the_page(reference):
    # The parser keeps &#12; or Word's &#11; as that character, and lxml refuses
    # to set text holding it: the tree edits re-set the text of every element
    # that had an emphasis child — chrome trafilatura discards later included —
    # so the page failed with an error naming no link.
    page = (
        f"<html><head><title>Basin{reference}report</title>"
        f'<meta property="og:site_name" content="Daily{reference}Water"></head>'
        "<body><article><h1>Basin</h1>"
        + "".join(f"<p>Gauge {n}: {PROSE}</p>" for n in range(3))
        + f"<p>Page{reference}break <a href='/x'>link</a> then <b>bold</b> text.</p>"
        + f"</article><footer><p>Contact{reference} us <a href='/c'>here</a> or <b>call</b></p>"
        + "</footer></body></html>"
    )
    document, metadata = extract_web(page, url=NEWS_URL)
    body = _body(document)
    assert "Page break link then bold text." in body
    assert body.count(PROSE.strip()) == 3
    assert not any(character in body for character in "\x08\x0b\x0c\x1b\x1f\ufffe")
    # Only the body's reading changes: the metadata is still the stdlib parser's.
    assert metadata == _page_metadata(page, url=NEWS_URL)
    assert metadata.publisher.replace(" ", "") == "DailyWater"


@pytest.mark.parametrize("reference", ["&#11;", "&#12;", "&#x1B;", "&#xFFFE;"])
@pytest.mark.parametrize(
    "paragraph",
    [
        "<p>{prose} <b>held</b> releases <a href='/n'>note</a>{ref} {prose}</p>",
        "<p><span>{prose}</span>{ref}<span>{prose}</span><b>.</b></p>",
        "<p>{prose} 10<sup>6</sup> <a href='/n'>note</a>{ref} {prose}</p>",
        "<p><b>Note</b> {prose}<br>{ref}{prose}</p>",
        "<p>{prose} <b>held{ref}</b> {prose}</p>",
        "<p>{prose}{ref} {prose}</p>",
    ],
    ids=["link-tail", "span-tail", "sup-sibling", "br-tail", "inside-bold", "plain-text"],
)
def test_a_control_character_reference_anywhere_in_a_paragraph_keeps_the_page(reference, paragraph):
    # Beside emphasis the tree edits raised over it; in plain text trafilatura
    # swallowed the same refusal and discarded the page as thin.
    page = (
        "<html><body><article><h1>Basin</h1><p>"
        + PROSE
        + "</p>"
        + paragraph.format(prose=PROSE, ref=reference)
        + "<p>"
        + PROSE
        + "</p></article></body></html>"
    )
    document, _ = extract_web(page, url=REPORT_URL)
    body = _body(document)
    assert body.count("reservoir operators cut releases") >= 3
    assert not any(character in body for character in "\x0b\x0c\x1b\ufffe")


def test_inline_svg_title_is_never_the_page_title():
    page = _page("hydrology_bulletin.html")
    for tag in (
        "<title>Groundwater and drought bulletin: March 2025 - National Hydrology Service</title>",
        '<meta name="twitter:title" property="og:title" '
        'content="Groundwater and drought bulletin: March 2025">',
    ):
        assert tag in page
        page = page.replace(tag, "")

    document, metadata = extract_web(page, url=BULLETIN_URL)

    assert metadata.title is None  # not the menu icon's "Close menu"
    assert document.title == "Groundwater and drought bulletin: March 2025"

    # An SVG sprite sheet ahead of the real <title> (or on a page without a
    # <body> tag, which is read whole) is skipped too.
    sprite = (
        '<svg xmlns="http://www.w3.org/2000/svg" style="display:none"><symbol id="i-search">'
        "<title>Search</title><path d='M1 1'/></symbol></svg><title>Rivers run low</title>"
    )
    assert _page_metadata(_markup(sprite), url="https://example.org/s").title == "Rivers run low"


def test_cjk_text_keeps_its_characters_when_emphasis_is_removed():
    raw = (FIXTURES / "shuili_news_zh.html").read_bytes()
    document, metadata = extract_web(raw, url=ZH_URL)

    body = _body(document)
    assert "世界卫生组织报告显示，2024年全球有5500万人受到干旱影响" in body
    assert "安全饮用水的获取仍然是最紧迫的问题" in body
    assert "*" not in body
    _assert_no_chrome(document, ("客户端下载", "版权所有", "评论"))
    assert metadata == PageMetadata(
        title="世界卫生组织：2024年全球5500万人受干旱影响",
        authors=["李明"],
        publisher="水利日报",
        publication_date="2025-03-20",
    )


def test_a_page_of_many_small_inline_elements_is_read_in_linear_time():
    # Three ways this went quadratic in the number of inline elements under one
    # parent, all reachable from a page inside the 10 MB fetch cap: libxml2
    # merges an XPath union ("//b | //i") by scanning one branch for every node
    # of the other; lxml.html's drop_tag/drop_tree re-copy the parent's growing
    # text for every child they remove; and libxml2's XPath is quadratic again
    # over the adjacent text nodes a C-level strip leaves behind (trafilatura
    # runs such XPaths). This page (0.6 MB, 64,000 inline elements) took about
    # 12 s; the cap allows 16x more elements, on the only worker thread, with
    # no cancel and no time limit.
    page = (
        "<!DOCTYPE html><html><head><title>Heavy</title></head><body><main><article>"
        f"<h1>Formatting</h1><p>{_prose(400)}</p><p>"
        + "<b>x</b><i>y</i>" * 16_000
        + "<sup>1</sup><sub>2</sub>" * 16_000
        + "</p></article></main></body></html>"
    )
    started = time.perf_counter()
    document, _ = extract_web(page, url="https://example.org/heavy")
    assert time.perf_counter() - started < 2.0
    assert "xy" * 16_000 + "^12" * 16_000 in _body(document)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("<a href='/start'>start</a> ", ""),  # the run lands in a kept child's tail
        ("", " <a href='/end'>end</a>"),  # the run lands in the parent's text
    ],
    ids=["run-in-child-tail", "run-in-parent-text"],
)
def test_inline_runs_beside_a_kept_child_are_read_in_linear_time(before, after):
    # The page above leaves its paragraph with no element at all. One element
    # that stays (a link, a <br>) splits the joining of the text nodes the strip
    # leaves into the parent's text and each kept child's tail; a run left
    # unjoined in either makes trafilatura's XPaths quadratic (over 10 s here).
    page = (
        "<!DOCTYPE html><html><head><title>Heavy</title></head><body><main><article>"
        f"<h1>Formatting</h1><p>{_prose(400)}</p><p>{before}"
        + "<b>x</b><i>y</i>" * 8_000
        + "<sup>1</sup><sub>2</sub>" * 8_000
        + f"{after}</p></article></main></body></html>"
    )
    started = time.perf_counter()
    document, _ = extract_web(page, url="https://example.org/heavy")
    assert time.perf_counter() - started < 2.0
    assert "xy" * 8_000 + "^12" * 8_000 in _body(document)


def _paragraph_of_bold(lead: str, count: int) -> str:
    return (
        "<!DOCTYPE html><html><head><title>Heavy</title></head><body><main><article>"
        f"<h1>Formatting</h1><p>{lead}" + "<b>x</b>" * count + "</p></article></main></body></html>"
    )


def _prepare_seconds(page: str):
    """CPU seconds _prepare_tree spends on the page, best of two, and the tree."""
    best, tree = float("inf"), None
    for _ in range(2):
        tree = trafilatura.load_html(page)
        started = time.process_time()
        _prepare_tree(tree)
        best = min(best, time.process_time() - started)
    return best, tree


def test_joining_a_kept_childs_text_runs_is_itself_linear():
    # Joining must not read a run after the strip: lxml's .text/.tail getter
    # concatenates a run of adjacent text nodes one node at a time, quadratic in
    # its length (with one <br>, a 2.4 MB paragraph took four to five times as
    # long as without it, and a 10 MB one about a minute). Timed relative to
    # the same page without the <br>, so machine speed cancels out.
    count = 300_000
    plain, _ = _prepare_seconds(_paragraph_of_bold("", count))
    kept, tree = _prepare_seconds(_paragraph_of_bold("<br>", count))
    assert kept < 2 * plain + 0.25, (kept, plain)
    br = tree.find(".//br")
    # One text node, counted in XPath: reading an unjoined run (below) is the
    # quadratic getter itself, and listing its nodes in Python is slower still.
    assert br.xpath("count(following-sibling::text())") == 1
    assert br.tail == "x" * count


def test_nested_scripts_emphasis_and_tables_read_the_same_after_the_linear_rewrite():
    # Pins the reading the element-by-element edits produced — a script is
    # classified after the scripts inside it were rewritten (its footnote link
    # gone, its digits already plain), emphasis nests, a Yoast FAQ question
    # stays a <strong>, sibling tables nested in one cell flatten in order —
    # so the rename-then-strip implementation cannot drift from it.
    page = (
        "<!DOCTYPE html><html><head><title>Nested</title></head><body><main><article>"
        f"<h1>Nested</h1><p>{_prose(300)}</p>"
        "<p>A<sup>1<sub>2</sub></sup> B<sup>3<sup><a href='#f'>4</a></sup></sup> "
        "C<sub>x<sup>5</sup></sub> D<b>bold <i>italic <em>deep</em></i></b> "
        "E<b>x<strong class='schema-faq-question'>Q?</strong></b>x F<u>u</u><sup>n</sup>tail</p>"
        "<table><tr><td>Cell"
        "<table><tr><td>a</td><td>b</td></tr></table>"
        "<table><tr><td>c</td></tr></table>"
        "</td><td>Other</td></tr></table>"
        "</article></main></body></html>"
    )
    document, _ = extract_web(page, url="https://example.org/nested")
    body = _body(document)
    assert "A^12 B^3 Cx^5 Dbold italic deep Ex\n## Q?\n\nx Funtail" in body
    assert "| Cell a, b c | Other |" in body


def test_a_nested_table_keeps_the_cells_and_caption_an_inline_tag_wraps():
    # The parser keeps malformed <tr><em><td> and <table><b><caption> as
    # written. The nested table is flattened while those wrappers are still in
    # place, and their cells and caption were lost; this is the reading the
    # element-by-element edits gave.
    page = (
        "<!DOCTYPE html><html><head><title>Wrapped</title></head><body><main><article>"
        f"<h1>Wrapped</h1><p>{_prose(300)}</p>"
        "<table><tr><td>Outer</td><td><table><b><caption>Cap</caption></b>"
        "<tr><td>Region</td><em><td>Deficit 41%</td></em></tr>"
        "<tr><sup><td>North</td></sup><strong><th>12</th></strong></tr>"
        "<tr><td>South</td><i><u><td>7</td></u></i></tr></table>"
        "</td></tr></table>"
        "</article></main></body></html>"
    )
    document, _ = extract_web(page, url="https://example.org/wrapped")
    assert "| Outer | Cap; Region, Deficit 41%; North, 12; South, 7 |" in _body(document)


def test_literal_asterisk_runs_survive_verbatim_and_fast():
    # The markdown post-pass this replaced went quadratic on unmatched openers
    # (90 KB of "*a " took minutes) and deleted literal asterisks.
    literal = "*a " * 30_000
    page = (
        "<!DOCTYPE html><html><head><title>Notes</title></head><body><main><article>"
        f"<h1>Field notes</h1><p>{_prose(400)}</p><p>{literal.strip()}</p>"
        "</article></main></body></html>"
    )
    started = time.perf_counter()
    document, _ = extract_web(page, url="https://example.org/notes")
    assert time.perf_counter() - started < 10.0
    assert literal.strip() in _body(document)


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
        # Entity-decoded, deduplicated PERSONAL names: the Organization author is
        # not a person (SourceMetadata.authors), and this page declares a publisher.
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


def _markup(*head: str, body: str = "<p>x</p>") -> str:
    return f"<!DOCTYPE html><html><head>{''.join(head)}</head><body>{body}</body></html>"


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


def test_a_website_node_is_not_a_description_of_the_page():
    markup = _markup(
        '<meta property="og:title" content="Rivers run low">',
        _ld({"@type": "WebSite", "name": "Daily Ledger", "publisher": "Ledger Group"}),
    )
    assert _page_metadata(markup, url="https://example.org/w") == PageMetadata(
        title="Rivers run low"
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


def test_every_jsonld_field_comes_from_one_work_node():
    # A news story that embeds the study it reports on: the study's authors
    # and DOI must never be attributed to the news page (credibility would
    # verify the page against someone else's record — Crossref resolves the
    # study's DOI and the page inherits its tier) — whichever node the page
    # lists first, and however the pasted link differs from the canonical URL.
    canonical = "https://news.example/aquifers"
    own = {
        "@type": "NewsArticle",
        "headline": "Study finds aquifers shrinking",
        "datePublished": "2025-02-01",
        "mainEntityOfPage": canonical,
    }
    study = {
        "@type": "ScholarlyArticle",
        "headline": "Global groundwater decline",
        "author": [{"@type": "Person", "name": "A. Researcher"}],
        "publisher": {"@type": "Organization", "name": "Nature Portfolio"},
        "datePublished": "2019-05-01",
        "identifier": "https://doi.org/10.1038/s41586-019-0000-0",
    }
    expected = PageMetadata(title="Study finds aquifers shrinking", publication_date="2025-02-01")
    for blocks in ((own, study), (study, own)):
        for url in (
            canonical,
            canonical + "?utm_source=twitter&utm_medium=social",
            "http://www.news.example/aquifers/",
        ):
            assert _page_metadata(_markup(*map(_ld, blocks)), url=url) == expected, (blocks, url)

    # When no article node names the page, or a cited node's "@id" is a
    # fragment of the page's URL so two do, document order decides nothing:
    # the JSON-LD contributes no field and the <meta> tags stand alone.
    tags = (
        '<meta property="og:title" content="Study finds aquifers shrinking">',
        '<meta property="article:published_time" content="2025-02-01T09:00:00Z">',
    )
    unnamed = {key: value for key, value in own.items() if key != "mainEntityOfPage"}
    fragment = {**study, "@id": canonical + "#cited-study"}
    for blocks in ((study, unnamed), (unnamed, study), (fragment, own), (own, fragment)):
        markup = _markup(*tags, *map(_ld, blocks))
        assert _page_metadata(markup, url=canonical) == expected, blocks
    assert _page_metadata(_markup(*map(_ld, (study, unnamed))), url=canonical) == PageMetadata()


def test_nodes_declaring_the_same_work_are_one_work_not_an_ambiguity():
    # WordPress: Yoast's @graph Article plus a theme's own BlogPosting, both
    # naming the page (or, with a share parameter on the link, neither). Every
    # JSON-LD field was dropped as ambiguous, the author with them.
    url = "https://greenhydro.example/2025/03/aquifer-recharge/"
    yoast = {
        "@graph": [
            {
                "@type": "Article",
                "@id": url + "#article",
                "headline": "Aquifer recharge",
                "datePublished": "2025-03-14T08:00:00+00:00",
                "mainEntityOfPage": {"@id": url},
                "author": {"@id": "https://greenhydro.example/#/schema/person/abc"},
                "publisher": {"@id": "https://greenhydro.example/#organization"},
            },
            {
                "@type": "Organization",
                "@id": "https://greenhydro.example/#organization",
                "name": "Green Hydro Media",
            },
            {
                "@type": "Person",
                "@id": "https://greenhydro.example/#/schema/person/abc",
                "name": "Maria Lopez",
            },
        ]
    }
    theme = {
        "@type": "BlogPosting",
        "mainEntityOfPage": {"@type": "WebPage", "@id": url},
        "headline": "Aquifer recharge",
        "datePublished": "2025-03-14",
        "author": {"@type": "Person", "name": "Maria Lopez"},
        "publisher": {"@type": "Organization", "name": "Green Hydro Media"},
    }
    expected = PageMetadata(
        title="Aquifer recharge",
        authors=["Maria Lopez"],
        publisher="Green Hydro Media",
        publication_date="2025-03-14",
    )
    for blocks in ((yoast, theme), (theme, yoast), (theme, theme)):
        for pasted in (url, url + "?ref=newsletter"):  # named by both, or by neither
            markup = _markup(*map(_ld, blocks))
            assert _page_metadata(markup, url=pasted) == expected, (blocks, pasted)

    # Nodes that describe different works stay ambiguous, named or not.
    cited = {**theme, "@id": url + "#cited-study", "headline": "Global groundwater decline"}
    other_author = {**theme, "author": {"@type": "Person", "name": "Ana Ruiz"}}
    for blocks in ((cited, theme), (theme, cited), (theme, other_author)):
        for pasted in (url, url + "?ref=newsletter"):
            metadata = _page_metadata(_markup(*map(_ld, blocks)), url=pasted)
            assert (metadata.title, metadata.authors) == (None, []), (blocks, pasted)


def test_the_work_node_that_names_this_url_is_the_page():
    url = "https://news.example/2025/aquifers"
    cited = {"@type": "ScholarlyArticle", "headline": "Global groundwater decline"}
    own = {
        "@type": "NewsArticle",
        "headline": "Study finds aquifers shrinking",
        "mainEntityOfPage": {"@type": "WebPage", "@id": url + "/"},
        "author": {"@type": "Person", "name": "Kim Osei"},
    }
    metadata = _page_metadata(_markup(_ld({"@graph": [cited, own]})), url=url + "#top")
    assert (metadata.title, metadata.authors) == ("Study finds aquifers shrinking", ["Kim Osei"])

    # What a pasted link adds to the canonical address does not hide the match:
    # scheme, a leading www., host case, a trailing slash, tracking parameters.
    for pasted in (
        "http://www.news.example/2025/aquifers/",
        "https://News.Example/2025/aquifers?utm_source=x&fbclid=1&mc_cid=2&gclid=3",
    ):
        metadata = _page_metadata(_markup(_ld({"@graph": [cited, own]})), url=pasted)
        assert metadata.authors == ["Kim Osei"], pasted
    # Any other query is part of the address: ?id=2 is not the page ?id=1 describes.
    listed = {**own, "mainEntityOfPage": url + "?id=1"}
    markup = _markup(_ld({"@graph": [cited, listed]}))
    assert _page_metadata(markup, url=url + "?id=2").authors == []
    assert _page_metadata(markup, url=url + "?utm_term=y&id=1").authors == ["Kim Osei"]


def test_the_page_node_match_ignores_how_the_address_is_encoded():
    # The fetched URL is spelled the way the link check spells it: an
    # international host in punycode, a typed non-ASCII path as upper-case %XX,
    # an escape pasted in lower case kept as pasted. A CMS writes its own url
    # decoded, or with lower-case escapes (WordPress slugs). With a cited work
    # also listed, the page's own node must still be found, in either order.
    cited = {
        "@type": "ScholarlyArticle",
        "headline": "Groundwater in Iberia",
        "author": {"@type": "Person", "name": "A. Researcher"},
    }
    for fetched, declared in (
        ("https://diario.example/agua-en-espa%C3%B1a/", "https://diario.example/agua-en-españa/"),
        (
            "https://diario.example/agua-en-espa%C3%B1a/",
            "https://diario.example/agua-en-espa%c3%b1a",
        ),
        ("https://diario.example/agua-en-espa%c3%b1a/", "https://diario.example/agua-en-españa/"),
        ("https://xn--bcher-kva.example/agua", "https://www.Bücher.example/agua/"),
        ("https://xn--strae-oqa.example/agua", "https://straße.example/agua"),
    ):
        own = {
            "@type": "NewsArticle",
            "headline": "Agua en España",
            "url": declared,
            "author": {"@type": "Person", "name": "Lucía Pérez"},
        }
        for blocks in ((own, cited), (cited, own)):
            metadata = _page_metadata(_markup(*map(_ld, blocks)), url=fetched)
            assert (metadata.title, metadata.authors) == ("Agua en España", ["Lucía Pérez"]), (
                fetched,
                declared,
            )

    # A different address stays different: "espana" is not "españa", and an
    # escaped slash is not a path separator.
    for fetched, declared in (
        ("https://diario.example/agua-en-espa%C3%B1a/", "https://diario.example/agua-en-espana/"),
        ("https://diario.example/agua/en", "https://diario.example/agua%2Fen"),
    ):
        own = {"@type": "NewsArticle", "headline": "Agua", "url": declared}
        markup = _markup(_ld(own), _ld(cited))
        assert _page_metadata(markup, url=fetched).title is None, declared

    # A declared url no text encoding can spell is simply not this page's.
    broken = {"@type": "NewsArticle", "headline": "A", "url": "https://a.example/x?q=\ud800"}
    markup = _markup(_ld(broken), _ld(cited))
    assert _page_metadata(markup, url="https://a.example/x").title is None


def test_organization_authors_are_not_personal_authors_but_can_name_the_publisher():
    """Authors are personal names, as the PDF path extracts them, so an institution
    credited as author earns no author points. It names the publisher only when the
    page declares none."""
    org_only = _markup(
        _ld(
            {
                "@type": "MedicalWebPage",
                "name": "Drought and health",
                "author": {"@type": "Organization", "name": "World Health Organization"},
            }
        )
    )
    metadata = _page_metadata(org_only, url="https://example.org/o")
    assert (metadata.authors, metadata.publisher) == ([], "World Health Organization")

    declared = _markup(
        _ld(
            {
                "@type": "Article",
                "headline": "Reservoirs at record lows",
                "author": [
                    {"@type": "Person", "name": "Ana Pérez"},
                    {"@type": "GovernmentOrganization", "name": "Water Ministry"},
                ],
                "publisher": {"@type": "NewsMediaOrganization", "name": "Daily Water"},
            }
        )
    )
    metadata = _page_metadata(declared, url="https://example.org/p")
    assert (metadata.authors, metadata.publisher) == (["Ana Pérez"], "Daily Water")


def test_a_meta_author_naming_an_organization_the_page_declares_is_not_a_personal_author():
    # The untyped <meta name=author> fallback carries no Person/Organization
    # type. A name the page itself declares as an organization elsewhere — the
    # JSON-LD author or publisher, citation_publisher, og:site_name — cannot be
    # a personal author (it would earn author points no PDF could).
    typed = _markup(
        '<meta name="author" content="World Health Organization">',
        _ld(
            {
                "@type": "MedicalWebPage",
                "name": "Drought and health",
                "author": {"@type": "Organization", "name": "World Health Organization"},
            }
        ),
    )
    metadata = _page_metadata(typed, url="https://example.org/o")
    assert (metadata.authors, metadata.publisher) == ([], "World Health Organization")

    site = _markup(
        '<meta name="author" content="world health organization">',  # compared casefolded
        '<meta property="og:site_name" content="World Health Organization">',
    )
    metadata = _page_metadata(site, url="https://example.org/s")
    assert (metadata.authors, metadata.publisher) == ([], "World Health Organization")

    person = _markup(
        '<meta name="author" content="Sam Patel">',
        '<meta property="og:site_name" content="Daily Water">',
    )
    metadata = _page_metadata(person, url="https://example.org/p")
    assert (metadata.authors, metadata.publisher) == (["Sam Patel"], "Daily Water")


def test_first_doi_source_wins_and_an_invalid_one_becomes_none():
    malformed = '<meta name="citation_doi" content="doi: not-a-doi">'
    jsonld = _ld(
        {"@type": "ScholarlyArticle", "headline": "X", "identifier": "https://doi.org/10.5555/x.9"}
    )
    url = "https://example.org/d"
    # A malformed citation_doi is the first non-empty source: it becomes None
    # rather than falling through to another source's DOI.
    assert _page_metadata(_markup(malformed, jsonld), url=url).doi is None
    assert _page_metadata(_markup(malformed), url=url).doi is None
    assert _page_metadata(_markup(jsonld), url=url).doi == "10.5555/x.9"
    # The same within JSON-LD: a declared DOI identifier comes before sameAs.
    declared = _ld(
        {
            "@type": "ScholarlyArticle",
            "identifier": {"@type": "PropertyValue", "propertyID": "DOI", "value": "pending"},
            "sameAs": "https://doi.org/10.5555/x.9",
        }
    )
    assert _page_metadata(_markup(declared), url=url).doi is None


def test_citation_doi_beats_a_valid_jsonld_doi():
    markup = _markup(
        '<meta name="citation_doi" content="10.5555/own.1">',
        _ld({"@type": "ScholarlyArticle", "identifier": "https://doi.org/10.5555/other.2"}),
    )
    assert _page_metadata(markup, url="https://example.org/d").doi == "10.5555/own.1"


def test_jsonld_author_names_beat_the_meta_author():
    markup = _markup(
        '<meta name="author" content="Site Editor">',
        _ld({"@type": "BlogPosting", "author": [{"name": "Ann Lee"}, {"name": "Bo Chen"}]}),
    )
    assert _page_metadata(markup, url="https://example.org/b").authors == ["Ann Lee", "Bo Chen"]


def test_citation_publication_date_beats_citation_date():
    markup = _markup(
        '<meta name="citation_date" content="2023/11/30">',
        '<meta name="citation_publication_date" content="2024/01/15">',
    )
    assert _page_metadata(markup, url="https://example.org/s").publication_date == "2024-01-15"


def test_jsonld_name_beats_og_title_when_there_is_no_headline():
    markup = _markup(
        '<meta property="og:title" content="Drought | Health topics">',
        _ld({"@type": "MedicalWebPage", "name": "Drought"}),
    )
    assert _page_metadata(markup, url="https://example.org/h").title == "Drought"


def test_author_names_dedupe_case_insensitively_keeping_the_first():
    markup = _markup(
        '<meta name="citation_author" content="Ann Lee">',
        '<meta name="citation_author" content="ANN LEE">',
        '<meta name="citation_author" content="Bo Chen">',
    )
    assert _page_metadata(markup, url="https://example.org/c").authors == ["Ann Lee", "Bo Chen"]


@pytest.mark.parametrize("declared", ["schema:Article", "https://schema.org/NewsArticle"])
def test_prefixed_schema_org_types_are_recognized(declared):
    markup = _markup(_ld({"@type": declared, "headline": "Rivers run low"}))
    assert _page_metadata(markup, url="https://example.org/p").title == "Rivers run low"


def test_jsonld_nested_past_the_recursion_limit_is_skipped_with_a_warning(web_log):
    url = "https://attacker.example/page"
    array_bomb = "[" * 1000 + "]" * 1000
    object_bomb = '{"@type":"Article","headline":"H","x":' + "[" * 3000 + "]" * 3000 + "}"
    page = _page("diario_agua_es.html").replace(
        "</head>",
        f'<script type="application/ld+json">{array_bomb}</script>'
        f'<script type="application/ld+json">{object_bomb}</script></head>',
    )

    document, metadata = extract_web(page, url=url)

    assert metadata.authors == ["Lucía Fernández Ibáñez"]  # the page's other metadata stands
    assert document.sections
    warnings = [r for r in web_log.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert all(url in record.getMessage() for record in warnings)


def test_attribute_and_title_entities_are_decoded_exactly_once():
    markup = _markup(
        '<meta property="og:title" content="What does &amp;nbsp; mean in HTML?">',
        '<meta name="citation_author" content="Ann &amp;amp; Bo">',
    )
    metadata = _page_metadata(markup, url="https://example.org/u")
    assert metadata.title == "What does &nbsp; mean in HTML?"
    assert metadata.authors == ["Ann &amp; Bo"]
    title_only = _markup("<title>Escaping &amp;lt;p&amp;gt; tags</title>")
    assert (
        _page_metadata(title_only, url="https://example.org/t").title == "Escaping &lt;p&gt; tags"
    )


def test_lone_surrogates_in_jsonld_become_replacement_characters():
    # JSON.stringify of a string cut mid-emoji emits "\ud83d"; json decodes it
    # to a lone surrogate, which the stored page's UTF-8 write cannot encode —
    # a readable article failed the whole run over half an emoji, with an error
    # naming no link. Every JSON-LD field read is a route, the DOI included.
    markup = _markup(
        _ld(
            {
                "@type": "ScholarlyArticle",
                "headline": "Drought update \ud83d",
                "author": [{"@type": "Person", "name": "Ana \udc00"}],
                "publisher": {"@type": "Organization", "name": "Water \ud83d News"},
                "datePublished": "2026-01-01\ud83d",
                "identifier": "10.5555/x.9\ud83d",
            }
        )
    )
    metadata = _page_metadata(markup, url="https://example.org/s")
    assert metadata.title == "Drought update �"
    assert metadata.authors == ["Ana �"]
    assert metadata.publisher == "Water � News"
    assert metadata.publication_date == "2026-01-01�"
    assert metadata.doi == "10.5555/x.9�"
    # A proper pair is one character and stays untouched.
    paired = _markup(_ld({"@type": "Article", "headline": "Up \U0001f600"}))
    assert _page_metadata(paired, url="https://example.org/p").title == "Up \U0001f600"


def test_meta_tags_in_the_body_are_not_page_metadata():
    markup = _markup(
        "<title>Rivers run low</title>",
        body=(
            '<div class="embed-card"><meta property="og:title" content="Another story">'
            '<meta name="citation_doi" content="10.5555/other.3"></div><p>Body text.</p>'
        ),
    )
    assert _page_metadata(markup, url="https://example.org/e") == PageMetadata(
        title="Rivers run low"
    )
    # Nor is a <title> element in the body — pasted or embedded markup.
    embedded = _markup(body="<div class='embed'><title>Widget - Maps</title></div><p>Body.</p>")
    assert _page_metadata(embedded, url="https://example.org/e").title is None


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


# --- (d) thin pages: JavaScript shells, consent banners, paywalls -------------


def test_javascript_shell_raises_thin_page_error_naming_the_url():
    with pytest.raises(ThinPageError) as excinfo:
        extract_web(_page("spa_shell.html"), url=SPA_URL)
    message = str(excinfo.value)
    assert SPA_URL in message
    assert "no readable article text" in message
    assert "JavaScript-only pages are not supported" in message
    assert isinstance(excinfo.value, ValueError)


@pytest.mark.parametrize(
    "banner",
    [
        'id="cookie-banner" class="consent consent--bottom"',  # as served
        'class="cookie-banner"',
        'class="gdpr-overlay" role="dialog"',
        'class="cmp-container" aria-modal="true"',
    ],
)
def test_a_consent_banner_is_never_the_article_text_of_a_javascript_shell(banner):
    page = _page("spa_consent_shell.html")
    served = 'id="cookie-banner" class="consent consent--bottom"'
    assert served in page
    with pytest.raises(ThinPageError, match=SPA_URL):
        extract_web(page.replace(served, banner), url=SPA_URL)


def test_a_paywall_prompt_is_not_article_text_so_a_teaser_page_is_thin():
    with pytest.raises(ThinPageError) as excinfo:
        extract_web(_page("paywall_teaser.html"), url=PAYWALL_URL)
    assert PAYWALL_URL in str(excinfo.value)


LEAKY_PARAGRAPHS = [
    "Survey crews measured subsidence at 214 benchmarks along the valley floor, and the "
    "fastest-sinking points dropped 6 cm in the last year alone, according to the "
    "regional water authority's latest monitoring report.",
    "The authority said irrigation wells drilled during the 2021 drought are the main "
    "cause. Pumping from the lower aquifer tripled between 2019 and 2023, while recharge "
    "from winter rain fell to less than half of its long-term average.",
    "Town engineers in Marlow and Ester Creek have already repaired cracked water mains "
    "twice this year, and a canal that carries drinking water to 40,000 residents has "
    "lost a fifth of its capacity where the ground beneath it subsided.",
    "State officials will vote next month on pumping limits that would cut groundwater "
    "use by 30 percent over five years, with compensation for farmers who fallow land.",
]
LEAKY_SUBSCRIBERS = (
    "The water utility says 12,000 of its subscribers in the valley have reported low "
    "pressure since January, and it has asked the state for emergency funding to replace "
    "the pumps at its two deepest wells before the summer demand peak arrives."
)


@pytest.mark.parametrize(
    "paragraphs",
    [
        LEAKY_PARAGRAPHS,  # short, not worded as an offer
        [*LEAKY_PARAGRAPHS, LEAKY_SUBSCRIBERS],  # says "subscribers", but article-length
    ],
    ids=["short", "long-mentions-subscribers"],
)
def test_article_text_inside_a_paywall_wrapper_is_kept(paragraphs):
    # Google's structured-data guidance wraps the PAYWALLED ARTICLE TEXT in
    # class="paywall"; when a server delivers that text, it is the article.
    page = _page("paywall_teaser.html")
    start = page.index('<div class="paywall')
    end = page.index("</div>", start) + len("</div>")
    leaky = '<div class="paywall">' + "".join(f"<p>{p}</p>" for p in paragraphs) + "</div>"
    assert (sum(map(len, paragraphs)) < 1000) == (paragraphs is LEAKY_PARAGRAPHS)

    document, _ = extract_web(page[:start] + leaky + page[end:], url=PAYWALL_URL)

    body = _body(document)
    assert "Engineers warned on Friday that falling groundwater" in body
    for paragraph in paragraphs:
        assert paragraph in body


def test_a_page_about_cookies_keeps_its_cookie_sections():
    # The consent pruning must not eat content that merely mentions cookies in
    # its class names. (trafilatura itself drops a <section> whose FIRST
    # attribute is a cookie class, so the section carries an id first.)
    guide = (
        "<p>First-party cookies are set by the site you visit and usually keep you signed in. "
        "Third-party cookies are set by other domains embedded in the page, such as ad "
        "networks, and can follow you from one site to the next.</p>"
    )
    page = (
        "<!DOCTYPE html><html><head><title>How cookies track you</title></head>"
        '<body class="has-cookie-banner">'
        '<article class="cookie-guide"><h1>How third-party cookies track you</h1>'
        f"<p>{_prose(300)}</p>"
        f'<section id="types" class="cookie-types"><h2>Types of cookies</h2>{guide}</section>'
        "</article>"
        '<div class="cookie-banner"><p>We use cookies to improve this site.</p>'
        "<button>OK</button></div></body></html>"
    )
    document, _ = extract_web(page, url="https://example.org/privacy/cookies")
    sections = _sections(document)
    assert "First-party cookies are set by the site you visit" in sections["Types of cookies"]
    assert "We use cookies to improve this site." not in _body(document)


def _short_article(article_text: str, heading: str = "Short note") -> str:
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>Short note</title>"
        "</head><body><header><nav><ul><li><a href='/'>Home</a></li><li><a href='/about'>"
        "About the institute</a></li></ul></nav></header><main><article>"
        f"<h1>{heading}</h1><p>{article_text}</p></article></main><footer><p>The River "
        "Basin Institute is a registered charity (no. 1187345). All rights reserved.</p>"
        "</footer></body></html>"
    )


@pytest.mark.parametrize("length", [199, 200, 201, MIN_BODY_CHARS - 1])
def test_a_short_article_is_thin_even_where_trafilatura_fuses_its_heading(length):
    # Below its own 250-character threshold trafilatura swaps its structured
    # result for a whole-text rescue that fuses the <h1> into the first word
    # ("Short noteRainfall...") and counts it as body text.
    with pytest.raises(ThinPageError, match="https://example.org/thin"):
        extract_web(_short_article(_prose(length)), url="https://example.org/thin")


def test_heading_text_does_not_count_toward_the_floor():
    heading = "Rainfall deficits in the upper Kessa basin and their effect on reservoir storage"
    with pytest.raises(ThinPageError):
        extract_web(_short_article(_prose(MIN_BODY_CHARS - 1), heading), url="https://ex.org/a")

    document, _ = extract_web(
        _short_article(_prose(MIN_BODY_CHARS), heading), url="https://ex.org/b"
    )
    assert [(s.title, len(s.text)) for s in document.sections] == [(heading, MIN_BODY_CHARS)]


def test_min_body_chars_is_trafilaturas_rescue_threshold():
    rescue = trafilatura.settings.DEFAULT_CONFIG.getint("DEFAULT", "MIN_EXTRACTED_SIZE")
    assert MIN_BODY_CHARS == rescue


def test_nothing_extractable_is_a_thin_page():
    for page in ("", "<html></html>", "<html><body><div id='app'></div></body></html>"):
        with pytest.raises(ThinPageError, match="https://example.org/empty"):
            extract_web(page, url="https://example.org/empty")


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


def test_charset_inside_another_meta_value_is_not_a_declaration(web_log):
    page = _page("diario_agua_es.html").replace(
        '<meta charset="utf-8">',
        '<meta name="description" content="How to declare charset=utf-8 on legacy pages">'
        '<meta charset="windows-1252">',
    )
    document, _ = extract_web(page.encode("cp1252"), url=DIARIO_URL)
    assert "“La reducción es real, pero frágil”" in _body(document)
    assert not web_log.records


def test_utf16_page_with_a_byte_order_mark_decodes():
    page = _page("diario_agua_es.html").replace('<meta charset="utf-8">', '<meta charset="utf-16">')
    document, metadata = extract_web(page.encode("utf-16"), url=DIARIO_URL)
    assert "La Organización Mundial de la Salud informó" in _body(document)
    assert metadata.authors == ["Lucía Fernández Ibáñez"]


def test_utf16_label_on_ascii_compatible_bytes_means_utf8(web_log):
    # WHATWG: a declaration readable as ASCII cannot be UTF-16, so the label
    # means UTF-8 — decoding these bytes as UTF-16 would yield CJK-looking noise.
    page = _page("diario_agua_es.html").replace('<meta charset="utf-8">', '<meta charset="utf-16">')
    document, _ = extract_web(page.encode("cp1252"), url=DIARIO_URL)
    body = _body(document)
    assert "Mundial de la Salud inform" in body
    assert "�" in body  # the cp1252 accents, replaced
    assert [r for r in web_log.records if r.levelno == logging.WARNING]


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


@pytest.mark.parametrize("label", ["bz2_codec", "base64", "quoted-printable", "undefined", "idna"])
def test_a_charset_label_that_is_not_a_text_encoding_fails_like_an_unknown_one(label):
    # codecs.lookup also resolves Python's byte-to-byte codecs (bz2, base64,
    # quoted-printable ...), which bytes.decode refuses with a LookupError, and
    # a few text codecs refuse any byte at all ("undefined", "idna"). Neither
    # failure was the documented ValueError naming the link, so the run showed
    # Python internals and no source row was marked.
    page = _page("diario_agua_es.html").replace(
        '<meta charset="utf-8">', f'<meta charset="{label}">'
    )
    with pytest.raises(ValueError, match="declares an unknown charset") as excinfo:
        extract_web(page.encode("cp1252"), url=DIARIO_URL)
    assert DIARIO_URL in str(excinfo.value)
    assert label in str(excinfo.value)
    assert not isinstance(excinfo.value, ThinPageError)


def test_declared_utf8_with_a_stray_invalid_byte_decodes_with_a_warning(web_log):
    raw = (FIXTURES / "diario_agua_es.html").read_bytes()
    damaged = raw.replace("São Paulo".encode(), b"S\xe3o Paulo")
    assert damaged != raw

    document, _ = extract_web(damaged, url=DIARIO_URL)

    assert "S�o Paulo" in _body(document)
    naming = [record for record in web_log.records if DIARIO_URL in record.getMessage()]
    assert naming and all(record.levelno == logging.WARNING for record in naming)


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


def test_extraction_is_deterministic_across_hash_seeds():
    # Same-process calls share one string-hash seed, so set- or hash-ordered
    # output would still compare equal there. Separate interpreters do not.
    backend = Path(web_mod.__file__).resolve().parents[1]
    script = (
        "import dataclasses, json, pathlib, sys\n"
        "import authorai.web as web\n"
        "print(pathlib.Path(web.__file__).resolve())\n"
        "page = pathlib.Path(sys.argv[1]).read_bytes()\n"
        "document, metadata = web.extract_web(page, url=sys.argv[2])\n"
        "output = [dataclasses.asdict(document), dataclasses.asdict(metadata)]\n"
        "print(json.dumps(output, sort_keys=True))\n"
    )
    outputs = []
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(backend)}
        result = subprocess.run(
            [sys.executable, "-c", script, str(FIXTURES / "report_jsonld_graph.html"), REPORT_URL],
            capture_output=True,
            text=True,
            env=env,
            cwd=backend,
            check=True,
        )
        imported, payload = result.stdout.splitlines()
        assert imported == str(Path(web_mod.__file__).resolve())  # this checkout's code ran
        outputs.append(payload)
    assert outputs[0] == outputs[1] == outputs[2]
    document, metadata = extract_web(_page("report_jsonld_graph.html"), url=REPORT_URL)
    assert json.loads(outputs[0]) == json.loads(
        json.dumps([dataclasses.asdict(document), dataclasses.asdict(metadata)])
    )


def test_document_title_falls_back_to_first_heading_then_none():
    page = (
        "<!DOCTYPE html><html><head></head><body><main><article>"
        "<h1><strong>First</strong> heading</h1><h2>Second</h2>"
        f"<p>{_prose(400)}</p></article></main></body></html>"
    )
    document, metadata = extract_web(page, url="https://example.org/a")
    assert metadata.title is None
    assert [s.title for s in document.sections] == ["Second"]
    assert document.title == "First heading"  # its own section was empty and dropped

    headless = (
        f"<!DOCTYPE html><html><body><main><article><p>{_prose(400)}</p></article></main>"
        "</body></html>"
    )
    document, _ = extract_web(headless, url="https://example.org/b")
    assert [s.title for s in document.sections] == [""]
    assert document.title is None


def test_a_jsonld_block_ending_in_a_semicolon_still_declares_the_page(web_log):
    """Shaped like a real institutional fact sheet (synthetic text): a
    BreadcrumbList, then the Article block — valid JSON followed by a stray ";" —
    holding the page's only publisher and date, then an ItemPage block."""
    article = json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": "Drinking-water",
            "datePublished": "2023-09-13T07:10:00.0000000+00:00",
            "author": {"@type": "Organization", "name": "Example Health Agency: EHA"},
            "publisher": {"@type": "Organization", "name": "Example Health Agency: EHA"},
        }
    )
    markup = _markup(
        "<title>\n\tDrinking-water</title>",
        _ld({"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": []}),
        f'<script type="application/ld+json">{article};</script>',
        _ld({"@context": "https://schema.org", "@type": "ItemPage", "name": "Detail"}),
    )
    metadata = _page_metadata(markup, url="https://www.health.example/fact-sheets/drinking-water")
    assert metadata.publisher == "Example Health Agency: EHA"
    assert metadata.publication_date == "2023-09-13"
    assert metadata.title == "Drinking-water"
    assert metadata.authors == []
    assert [r for r in web_log.records if r.levelno == logging.WARNING] == []


def test_other_content_after_a_jsonld_value_is_still_malformed(web_log):
    markup = _markup(
        '<script type="application/ld+json">{"@type": "Article", "headline": "A"} '
        '{"@type": "Article", "headline": "B"}</script>'
    )
    assert _page_metadata(markup, url="https://example.org/x").title is None
    assert len([r for r in web_log.records if r.levelno == logging.WARNING]) == 1


@pytest.mark.parametrize("page_type", ["ItemPage", "CollectionPage", "FAQPage", "AboutPage"])
def test_webpage_subtypes_describe_the_page(page_type):
    markup = _markup(
        _ld(
            {
                "@type": page_type,
                "name": "Groundwater levels",
                "publisher": {"@type": "Organization", "name": "Basin Authority"},
            }
        )
    )
    metadata = _page_metadata(markup, url="https://example.org/g")
    assert (metadata.title, metadata.publisher) == ("Groundwater levels", "Basin Authority")
