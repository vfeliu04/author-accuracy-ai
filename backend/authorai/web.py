"""Web-page sources: readable article text and self-declared metadata from HTML.

One page, two independent readings:

- BODY: the page is parsed once and edited before trafilatura reads it.
  Consent banners, modal dialogs and paywall prompts are pruned (on some page
  shapes trafilatura returns them as the article). Emphasis tags are
  unwrapped: chunk text is later quoted verbatim as evidence, and markdown
  markers cannot be stripped afterwards because trafilatura does not escape a
  page's own asterisks. Super/subscripts become Unicode or plain text, and a
  table nested in a table is flattened into its cell (trafilatura drops such
  tables, and the tables around them). Sections are split on the heading
  elements of trafilatura's extracted tree, never by parsing its markdown,
  where a paragraph that starts with "# " or "```" looks like structure.
  Each section is written with trafilatura's own markdown writer, so pipe
  tables stay inline and unchanged.
- METADATA: read from the page's OWN markup (<meta> tags and <title> in the
  head, JSON-LD anywhere) with the stdlib HTML parser. Never from
  trafilatura's metadata (its site name can be derived from the hostname and
  its date search is heuristic — guesses that would feed credibility
  scoring), never from the hostname (the authority matcher token-fuses, so
  "who-cares.com" would borrow the World Health Organization's authority),
  and never from body text (reference lists describe OTHER works).

The result is a ParsedDocument like a PDF's: everything downstream of parsing
treats both the same.
"""

import codecs
import copy
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from html import unescape
from html.parser import HTMLParser

import trafilatura

# trafilatura's markdown writer — the function its extract() uses — applied
# per section. Internal module; trafilatura is pinned exactly in pyproject.
from trafilatura.xml import xmltotxt

from authorai.credibility import clean_doi
from authorai.ingest import ParsedDocument, ParsedSection
from authorai.log import setup_logger

logger = setup_logger(__name__)

# Floor on extracted body text in characters, counted over section text only
# (headings do not count). It equals trafilatura's MIN_EXTRACTED_SIZE: below
# that size trafilatura replaces its structured extraction with a whole-text
# rescue that fuses the heading into the first sentence ("Short noteRainfall")
# and drops paragraph breaks, so any lower floor would accept exactly that
# output. 250 characters is two or three sentences, about the least prose that
# can carry one checkable claim with its context; a JavaScript shell's
# <noscript> notice is ~50. (A rescue that recovers MORE than the floor is kept:
# it is trafilatura's recall path for markup its main extractor misreads.)
MIN_BODY_CHARS = 250


class ThinPageError(ValueError):
    """The page yielded no readable article text — typically a JavaScript-rendered shell."""


@dataclass
class PageMetadata:
    """What a page's markup declares about itself. Field names mirror
    credibility.SourceMetadata so scoring can consume either."""

    title: str | None = None
    authors: list[str] = field(default_factory=list)
    publisher: str | None = None
    publication_date: str | None = None
    doi: str | None = None
    scholarly: bool = False  # True when the page carries any citation_* meta tag


def extract_web(html: bytes | str, *, url: str) -> tuple[ParsedDocument, PageMetadata]:
    """Parse a fetched page into sections (page=None; no tables or figures —
    tables stay inline as markdown) plus the metadata its markup declares.

    Raises ThinPageError when less than MIN_BODY_CHARS of article text can be
    read, and ValueError when page bytes cannot be decoded.
    """
    text = html if isinstance(html, str) else _decode(html, url)
    sections, first_heading = _body_sections(text, url)
    body_chars = sum(len(section.text) for section in sections)
    if body_chars < MIN_BODY_CHARS:
        raise ThinPageError(
            f"{url} has no readable article text ({body_chars} characters extracted, "
            f"at least {MIN_BODY_CHARS} needed) — JavaScript-only pages are not supported"
        )
    metadata = _page_metadata(text, url)
    document = ParsedDocument(
        title=metadata.title or first_heading, sections=sections, tables=[], figures=[]
    )
    return document, metadata


# --- decoding ----------------------------------------------------------------

# Where browsers look for a <meta> charset declaration (WHATWG's prescan reads
# 1024 bytes; real heads with inline scripts often run longer).
_CHARSET_SCAN_BYTES = 4096
_CONTENT_CHARSET = re.compile(r"charset\s*=\s*[\"']?\s*([A-Za-z0-9._:-]+)", re.IGNORECASE)


def _decode(raw: bytes, url: str) -> str:
    """A UTF-16 byte-order mark decides first; then UTF-8 whenever the bytes
    are valid UTF-8 (whatever the page claims); then the charset the page
    declares, decoded the way a browser does; otherwise a loud failure — a
    guessed encoding would silently corrupt text that is later quoted as
    evidence."""
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return _decode_as(raw, "utf-16", "UTF-16 byte-order mark", url)
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    label = _declared_charset(raw[:_CHARSET_SCAN_BYTES])
    if label is None:
        raise ValueError(
            f"{url} is not valid UTF-8 and has no byte-order mark or <meta> charset "
            f"declaration in its first {_CHARSET_SCAN_BYTES} bytes — decode it with the "
            "HTTP Content-Type charset and pass text instead"
        )
    try:
        codec = codecs.lookup(label).name
    except LookupError as exc:
        raise ValueError(f"{url} declares an unknown charset {label!r}") from exc
    if codec in ("iso8859-1", "ascii"):
        codec = "cp1252"  # WHATWG: browsers decode these labels as windows-1252
    elif codec.startswith("utf-16"):
        codec = "utf-8"  # WHATWG: a declaration readable as ASCII cannot mean UTF-16
    return _decode_as(raw, codec, f"declared charset {label!r}", url)


def _decode_as(raw: bytes, codec: str, source: str, url: str) -> str:
    text = raw.decode(codec, errors="replace")
    if replaced := text.count("�"):
        logger.warning("%s: %d undecodable byte sequence(s) replaced (%s)", url, replaced, source)
    return text


class _CharsetScanner(HTMLParser):
    """The two declarations WHATWG's prescan honors: <meta charset=...>, and
    <meta http-equiv="content-type" content="...; charset=...">. A "charset="
    inside any other attribute value — a description, say — declares nothing."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.label: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta" or self.label is not None:
            return
        attributes: dict[str, str] = {}
        for name, value in attrs:
            attributes.setdefault(name, value or "")  # the first of a repeated attribute
        if charset := attributes.get("charset", "").strip():
            self.label = charset
        elif attributes.get("http-equiv", "").strip().lower() == "content-type":
            if match := _CONTENT_CHARSET.search(attributes.get("content", "")):
                self.label = match.group(1)


def _declared_charset(prefix: bytes) -> str | None:
    scanner = _CharsetScanner()
    scanner.feed(prefix.decode("latin-1"))  # byte-transparent; declarations are ASCII
    return scanner.label


# --- body --------------------------------------------------------------------

# Page furniture pruned before extraction. A JavaScript-only shell whose one
# server-rendered block is a cookie banner came back with the banner as its
# only section, and a paywall prompt as a section of its own lifted a
# two-sentence teaser over the thin-page floor.
# - Dialogs (role or aria-modal), unless they hold the page's article.
# - cookie/consent in an id or class, but never on, inside, or around the
#   article or <main>: pages ABOUT cookies carry the token on real content,
#   and themes put it on wrapper classes ("has-cookie-banner").
# - paywall in an id or class, only when it reads like an offer: under 1000
#   characters AND worded as a subscription (en/es/fr/de stems). Google's
#   structured-data guidance wraps the paywalled ARTICLE TEXT in
#   class="paywall", and when a server delivers that text it is the article —
#   a short one included. An unmatched prompt stays in, as before; article
#   text is never the thing guessed away.
_PRUNE_XPATH = """
//body//*[
  (@role = 'dialog' or @role = 'alertdialog' or @aria-modal = 'true')
    and not(.//*[self::article or self::main or @role = 'main'])
  or (contains(translate(@id, 'COKIE', 'cokie'), 'cookie')
      or contains(translate(@class, 'COKIE', 'cokie'), 'cookie')
      or contains(translate(@id, 'CONSET', 'conset'), 'consent')
      or contains(translate(@class, 'CONSET', 'conset'), 'consent'))
    and not(ancestor-or-self::*[self::article or self::main or @role = 'main']
            or .//*[self::article or self::main or @role = 'main'])
  or (contains(translate(@id, 'PAYWL', 'paywl'), 'paywall')
      or contains(translate(@class, 'PAYWL', 'paywl'), 'paywall'))
    and string-length(normalize-space()) < 1000
    and (contains(translate(., 'SUBCRIEA', 'subcriea'), 'subscri')
         or contains(translate(., 'SUBCRIEA', 'subcriea'), 'suscri')
         or contains(translate(., 'ABON', 'abon'), 'abonn'))
]
"""
# Yoast FAQ questions are <strong> elements trafilatura turns into headings.
_EMPHASIS_XPATH = "//b | //strong[not(contains(@class, 'schema-faq-question'))] | //i | //em | //u"
_SCRIPT_CHARACTERS = frozenset("0123456789+-−=()")
_SUPERSCRIPT = str.maketrans("0123456789+-−=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾")
_SUBSCRIPT = str.maketrans("0123456789+-−=()", "₀₁₂₃₄₅₆₇₈₉₊₋₋₌₍₎")
_HEADING_PREFIX = re.compile(r"^#{1,6} ?")


def _body_sections(text: str, url: str) -> tuple[list[ParsedSection], str | None]:
    """trafilatura's fast mode first, its full cascade only for a thin result.

    The full cascade compares trafilatura's extraction with readability's and
    takes readability's whenever the page has more tables than paragraphs —
    a statistics page — and readability's tables lose rowspan placeholders
    (a value slides into the wrong column) and sometimes whole tables. Fast
    mode skips that comparison. It also skips readability's and jusText's
    rescue of layouts trafilatura's own extractor misreads, so a page fast
    mode finds thin gets the full cascade before it is called thin.
    """
    tree = trafilatura.load_html(text)
    if tree is None:
        return [], None
    _prepare_tree(tree)
    sections, first_heading = _extract_sections(copy.deepcopy(tree), url, fast=True)
    if sum(len(section.text) for section in sections) < MIN_BODY_CHARS:
        sections, first_heading = _extract_sections(tree, url, fast=False)
    return sections, first_heading


def _extract_sections(tree, url: str, *, fast: bool) -> tuple[list[ParsedSection], str | None]:
    document = trafilatura.bare_extraction(
        tree,  # modified in place (pruning)
        url=url,
        fast=fast,
        output_format="markdown",
        include_formatting=True,
        include_tables=True,
        include_comments=False,
        include_images=False,
        include_links=False,
        prune_xpath=_PRUNE_XPATH,
    )
    if document is None or document.body is None:
        return [], None
    return _split_sections(document.body)


def _prepare_tree(tree) -> None:
    """Edit the parsed page in place, before trafilatura reads it:

    - emphasis tags are unwrapped, keeping their text;
    - a <sup>/<sub> holding only digits and signs becomes Unicode super- or
      subscript (10<sup>6</sup> must not read "106"); a <sup> holding a link
      is a footnote or reference marker and is removed; any other
      <sup>/<sub> is unwrapped to plain text;
    - a table nested in a table becomes its text inside the outer cell —
      cells joined by ", ", rows by "; " — innermost first.
    """
    for element in tree.xpath(_EMPHASIS_XPATH):
        element.drop_tag()
    for element in reversed(tree.xpath("//sup | //sub")):
        content = element.text_content().strip()
        if element.tag == "sup" and element.find(".//a") is not None:
            element.drop_tree()
        elif content and set(content) <= _SCRIPT_CHARACTERS:
            script = _SUPERSCRIPT if element.tag == "sup" else _SUBSCRIPT
            _replace_with_text(element, content.translate(script))
        else:
            element.drop_tag()
    # Reverse document order visits a nested table before the table holding it.
    for table in reversed(tree.xpath("//table[ancestor::table]")):
        _replace_with_text(table, f" {_table_text(table)} ")


def _replace_with_text(element, text: str) -> None:
    element.tail = text + (element.tail or "")
    element.drop_tree()  # lxml joins the tail to the preceding text


def _table_text(table) -> str:
    rows = []
    caption = table.find("caption")
    if caption is not None:
        rows.append(" ".join(caption.text_content().split()))
    for row in table.iter("tr"):
        cells = (" ".join(cell.text_content().split()) for cell in row if cell.tag in ("td", "th"))
        rows.append(", ".join(cell for cell in cells if cell))
    return "; ".join(row for row in rows if row)


def _split_sections(body) -> tuple[list[ParsedSection], str | None]:
    """One section per heading element at the top level of trafilatura's
    extracted tree; content before the first heading gets the title "".
    Sections with no text are dropped. Also returns the first heading's title,
    even when its own section was dropped."""
    sections: list[ParsedSection] = []
    first_heading: str | None = None
    title = ""
    blocks = body.makeelement(body.tag, {})
    blocks.text = body.text
    for child in list(body):
        if child.tag != "head":
            blocks.append(child)  # moves it out of body
            continue
        _add_section(sections, title, blocks)
        tail, child.tail = child.tail, None  # loose text after a heading opens its section
        title = " ".join(_HEADING_PREFIX.sub("", _markdown(child)).split())
        if first_heading is None and title:
            first_heading = title
        blocks = body.makeelement(body.tag, {})
        blocks.text = tail
    _add_section(sections, title, blocks)
    return sections, first_heading


def _add_section(sections: list[ParsedSection], title: str, blocks) -> None:
    if text := _markdown(blocks):
        sections.append(ParsedSection(title=title, page=None, text=text))


def _markdown(element) -> str:
    # NFC, as trafilatura.extract normalizes its output.
    return unicodedata.normalize("NFC", xmltotxt(element, include_formatting=True)).strip()


# --- metadata ----------------------------------------------------------------

_ARTICLE_TYPES = frozenset({"Article", "NewsArticle", "BlogPosting", "Report", "ScholarlyArticle"})
_PAGE_TYPES = frozenset({"WebPage", "MedicalWebPage"})
_DOI_LIKE = re.compile(r"\s*(?:https?://(?:dx\.)?doi\.org/|doi:|10\.\d{4,9}/)", re.IGNORECASE)
_ISO_DATE = re.compile(r"(\d{4})(?:[-/](\d{1,2})(?:[-/](\d{1,2}))?)?(?:[T ]\d{1,2}:\d{2}.*)?")
# Inline SVG and MathML have <title> elements of their own (an icon's label).
_FOREIGN_CONTENT = frozenset({"svg", "math"})


class _MarkupParser(HTMLParser):
    """Collects what a page says about itself: <meta name|property> pairs and
    the first <title> from the head — everything before <body>; a page with no
    <body> tag is read whole — plus every JSON-LD block's raw text wherever it
    sits (pages often emit JSON-LD at the end of the body)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: list[tuple[str, str]] = []
        self.jsonld: list[str] = []
        self.title: str | None = None
        self._in_head = True
        self._foreign_depth = 0
        self._capturing: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "body":
            self._in_head = False
        elif tag in _FOREIGN_CONTENT:
            self._foreign_depth += 1
        elif tag == "meta" and self._in_head:
            content = attributes.get("content")
            keys = (attributes.get("name"), attributes.get("property"))
            for key in dict.fromkeys(key.strip().lower() for key in keys if key):
                if content is not None:
                    self.meta.append((key, content))
        elif tag == "script":
            kind = (attributes.get("type") or "").split(";")[0].strip().lower()
            if kind == "application/ld+json":
                self._capturing, self._parts = "script", []
        elif (
            tag == "title"
            and self._in_head
            and not self._foreign_depth
            and self.title is None
            and self._capturing is None
        ):
            self._capturing, self._parts = "title", []

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _FOREIGN_CONTENT:  # a self-closed <svg/> opens nothing
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag in _FOREIGN_CONTENT and self._foreign_depth:
            self._foreign_depth -= 1
        if tag != self._capturing:
            return
        if tag == "script":
            self.jsonld.append("".join(self._parts))
        else:
            self.title = "".join(self._parts)
        self._capturing = None

    def handle_data(self, data: str) -> None:
        if self._capturing is not None:
            self._parts.append(data)


def _page_metadata(text: str, url: str) -> PageMetadata:
    """Metadata by fixed precedence — the first non-empty source wins, and a
    winning DOI that fails validation is None, never a fall-through:

    title       citation_title > JSON-LD headline/name > og:title > <title>
    authors     citation_author (all) > JSON-LD author names > meta author (one)
    publisher   citation_publisher > JSON-LD publisher name > og:site_name
    date        citation_publication_date > citation_date > JSON-LD datePublished
                > article:published_time
    doi         citation_doi > JSON-LD identifier/sameAs DOI

    Every JSON-LD value comes from ONE node, the page's own work (_primary_work):
    a page can embed records of other works — the study a news story reports
    on — whose authors and DOI are not the page's.
    """
    parser = _MarkupParser()
    parser.feed(text)
    parser.close()

    meta: dict[str, list[str]] = {}
    for key, raw in parser.meta:
        if value := _clean(raw):
            meta.setdefault(key, []).append(value)

    def first(*keys: str) -> str | None:
        return next((meta[key][0] for key in keys if key in meta), None)

    nodes = _jsonld_nodes(parser.jsonld, url)
    by_id: dict[str, dict] = {}
    for node in nodes:
        if isinstance(node.get("@id"), str):
            by_id.setdefault(node["@id"], node)
    work = _primary_work(nodes, url) or {}

    title = (
        first("citation_title")
        or _clean_jsonld(_first_str(work.get("headline")))
        or _clean_jsonld(_first_str(work.get("name")))
        or first("og:title")
        or _clean(parser.title)
    )
    authors = (
        meta.get("citation_author")
        or _names(work.get("author"), by_id)
        or meta.get("author", [])[:1]
    )
    publisher = (
        first("citation_publisher")
        or next(iter(_names(work.get("publisher"), by_id)), None)
        or first("og:site_name")
    )
    published = (
        first("citation_publication_date", "citation_date")
        or _clean_jsonld(_first_str(work.get("datePublished")))
        or first("article:published_time")
    )
    declared_doi = first("citation_doi") or next(iter(_doi_candidates(work)), None)
    return PageMetadata(
        title=title,
        authors=_dedupe(authors),
        publisher=publisher,
        publication_date=_normalize_date(published) if published else None,
        doi=clean_doi(declared_doi) if declared_doi else None,
        scholarly=any(key.startswith("citation_") for key, _ in parser.meta),
    )


def _jsonld_nodes(blocks: list[str], url: str) -> list[dict]:
    """Every object in the page's JSON-LD — a block may hold one object, a
    list, or an {"@graph": [...]} wrapper. A malformed block is the page's
    defect, not ours: it is skipped with a warning naming the page."""
    nodes: list[dict] = []
    for number, block in enumerate(blocks, start=1):
        if not block.strip():
            continue
        try:
            # strict=False: CMSes emit raw newlines inside JSON-LD strings.
            data = json.loads(block, strict=False)
        except (ValueError, RecursionError) as exc:
            # RecursionError: nesting deeper than the interpreter's limit.
            logger.warning("%s: skipping malformed JSON-LD block %d (%s)", url, number, exc)
            continue
        for item in data if isinstance(data, list) else [data]:
            if not isinstance(item, dict):
                continue
            nodes.append(item)
            graph = item.get("@graph")
            for node in graph if isinstance(graph, list) else [graph]:
                if isinstance(node, dict):
                    nodes.append(node)
    return nodes


def _primary_work(nodes: list[dict], url: str) -> dict | None:
    """The node describing this page: among article-type nodes, the one whose
    url, @id or mainEntityOfPage names this URL, else the first of them. Only
    without any article node, the same choice among page-type nodes — a
    WebPage node describes the container ("Title - Site Name"), not the work."""
    page = _comparable_url(url)
    for kinds in (_ARTICLE_TYPES, _PAGE_TYPES):
        candidates = [node for node in nodes if _types(node) & kinds]
        if candidates:
            return next((node for node in candidates if page in _named_urls(node)), candidates[0])
    return None


def _named_urls(node: dict) -> set[str]:
    entity = node.get("mainEntityOfPage")
    values = [node.get("url"), node.get("@id"), entity]
    if isinstance(entity, dict):
        values += [entity.get("@id"), entity.get("url")]
    return {_comparable_url(value) for value in values if isinstance(value, str)}


def _comparable_url(value: str) -> str:
    return value.strip().split("#", 1)[0].rstrip("/")


def _types(node: dict) -> set[str]:
    declared = node.get("@type")
    names = declared if isinstance(declared, list) else [declared]
    # "schema:Article" and "https://schema.org/Article" name the same type.
    return {n.rsplit("/", 1)[-1].rsplit(":", 1)[-1] for n in names if isinstance(n, str)}


def _names(value: object, by_id: dict[str, dict]) -> list[str]:
    """Names from a string, an object (a Person or an Organization), or a list
    of either. An object that is only an {"@id": ...} reference resolves
    against the page's graph."""
    names: list[str] = []
    for item in value if isinstance(value, list) else [value]:
        name = item
        if isinstance(item, dict):
            node = item
            if "name" not in node and isinstance(node.get("@id"), str):
                node = by_id.get(node["@id"], node)
            name = _first_str(node.get("name"))
        if cleaned := _clean_jsonld(name):
            names.append(cleaned)
    return names


def _doi_candidates(node: dict) -> list[str]:
    """DOIs a node declares, in order: identifier (a string, a PropertyValue,
    or a list of either), then sameAs. A PropertyValue whose propertyID is DOI
    counts whatever its value looks like — so a malformed declared DOI reaches
    clean_doi and is reported — while other values count only when DOI-shaped,
    so profile links in sameAs are not taken for malformed DOIs."""
    candidates: list[str] = []
    for key in ("identifier", "sameAs"):
        value = node.get(key)
        for item in value if isinstance(value, list) else [value]:
            declared_doi = False
            if isinstance(item, dict):
                declared_doi = str(item.get("propertyID", "")).strip().lower() == "doi"
                item = item.get("value")
            if not isinstance(item, str) or not item.strip():
                continue
            if declared_doi or _DOI_LIKE.match(item):
                candidates.append(unescape(item.strip()))
    return candidates


def _normalize_date(raw: str) -> str:
    """ISO YYYY-MM-DD — or YYYY-MM / YYYY when only those are present — when
    the value parses as a date; otherwise the value as printed, never a guess."""
    match = _ISO_DATE.fullmatch(raw)
    if match is None:
        return raw
    year, month, day = match.groups()
    try:
        parsed = date(int(year), int(month or 1), int(day or 1))
    except ValueError:
        return raw
    return parsed.isoformat()[: 10 if day else 7 if month else 4]


def _clean(value: object) -> str | None:
    """Whitespace-collapsed and stripped; empty or non-string is None. Markup
    values arrive entity-decoded from the HTML parser: decoding them again
    would turn a page's literal "&amp;nbsp;" into a space."""
    if not isinstance(value, str):
        return None
    return " ".join(value.split()) or None


def _clean_jsonld(value: object) -> str | None:
    """A JSON-LD string, entity-decoded first: <script> content is not
    decoded by HTML parsers, and CMSes entity-encode it ("Tom&aacute;s")."""
    return _clean(unescape(value)) if isinstance(value, str) else None


def _first_str(value: object) -> str | None:
    if isinstance(value, list):
        value = next((item for item in value if isinstance(item, str)), None)
    return value if isinstance(value, str) else None


def _dedupe(names: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name.casefold() not in seen:
            seen.add(name.casefold())
            unique.append(name)
    return unique
