"""Web-page sources: readable article text and self-declared metadata from HTML.

One page, two independent readings:

- BODY: trafilatura extracts the main content as markdown — navigation,
  consent banners and footers are dropped — and it is split into sections on
  its headings. Emphasis markers are stripped because chunk text is later
  quoted verbatim as evidence and checked by normalized substring match; pipe
  tables stay inline in their section, unchanged.
- METADATA: read from the page's OWN markup (<meta> tags, JSON-LD, <title>)
  with the stdlib HTML parser. Never from trafilatura's metadata (its site
  name can be derived from the hostname and its date search is heuristic —
  guesses that would feed credibility scoring), never from the hostname (the
  authority matcher token-fuses, so "who-cares.com" would borrow the World
  Health Organization's authority), and never from body text (reference
  lists describe OTHER works).

The result is a ParsedDocument like a PDF's: everything downstream of parsing
treats both the same.
"""

import codecs
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from html import unescape
from html.parser import HTMLParser

import trafilatura

from authorai.credibility import clean_doi
from authorai.ingest import ParsedDocument, ParsedSection
from authorai.log import setup_logger

logger = setup_logger(__name__)

# Floor on extracted body text in characters, counted over section text only
# (headings and stripped markers do not count). ~200 characters is two or
# three sentences — about the least prose that can carry one checkable claim
# with its context. What extracts to less is not an article: a JavaScript
# shell's <noscript> notice is ~50 characters, a consent or paywall stub a few
# more, while any real article clears the floor by an order of magnitude.
MIN_BODY_CHARS = 200


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
    markdown = trafilatura.extract(
        text,
        url=url,
        output_format="markdown",
        include_tables=True,
        include_comments=False,
        include_images=False,
        include_links=False,
    )
    sections, first_heading = _split_sections(markdown or "")
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
_META_CHARSET = re.compile(rb"<meta[^>]*?charset\s*=\s*[\"']?\s*([A-Za-z0-9._:-]+)", re.IGNORECASE)


def _decode(raw: bytes, url: str) -> str:
    """UTF-8 whenever the bytes are valid UTF-8 (whatever the page claims);
    otherwise the charset the page declares, decoded the way a browser does;
    otherwise a loud failure — a guessed encoding would silently corrupt text
    that is later quoted as evidence."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    declared = _META_CHARSET.search(raw[:_CHARSET_SCAN_BYTES])
    if declared is None:
        raise ValueError(
            f"{url} is not valid UTF-8 and declares no charset in a <meta> tag — "
            "decode it with the HTTP Content-Type charset and pass text instead"
        )
    label = declared.group(1).decode("ascii")
    try:
        codec = codecs.lookup(label).name
    except LookupError as exc:
        raise ValueError(f"{url} declares an unknown charset {label!r}") from exc
    if codec in ("iso8859-1", "ascii"):
        codec = "cp1252"  # WHATWG: browsers decode these labels as windows-1252
    elif codec.startswith("utf-16"):
        codec = "utf-8"  # WHATWG: a declaration readable as ASCII cannot mean UTF-16
    text = raw.decode(codec, errors="replace")
    if replaced := text.count("�"):
        logger.warning(
            "%s: %d undecodable byte sequence(s) replaced (declared charset %r)",
            url,
            replaced,
            label,
        )
    return text


# --- body --------------------------------------------------------------------

# Emphasis markers are delimiter runs (maximal runs of * or _), removed in
# matched pairs by one linear pass modeled on CommonMark's delimiter stack. A
# regex with lazy spans re-scanned to the end of the paragraph from every
# unmatched opener — quadratic, and 90 KB of "*a " took minutes.
# A run can open when the next character is not whitespace and the previous one
# is not a letter, digit or underscore; it can close when the previous character
# is not whitespace and the next one is not a letter, digit or underscore — so
# literal snake_case, 2*3, "5 * 3" and Data* survive. Runs never pair across a
# blank line (a paragraph break) or a pipe (a table-cell boundary). The markdown
# itself cannot tell a page's literal "*phrase*" from <em>phrase</em>; that rare
# literal is stripped too — leaving real markers in would break every quote
# that spans one.
_DELIMITER_RUN = re.compile(r"\*+|_+")
_EMPHASIS_BARRIER = re.compile(r"(\n[ \t]*\n|\|)")
_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
_CLOSING_HASHES = re.compile(r"(?<!\S)#+\s*$")
_FENCE_OPEN = re.compile(r"^(`{3,})[^`]*$")


def _strip_emphasis(text: str) -> str:
    pieces = _EMPHASIS_BARRIER.split(text)  # separators land at the odd indexes
    pieces[::2] = [_strip_emphasis_segment(piece) for piece in pieces[::2]]
    return "".join(pieces)


def _strip_emphasis_segment(segment: str) -> str:
    runs: list[list] = []  # [start, end, character, count still unmatched]
    openers: dict[str, list[list]] = {"*": [], "_": []}
    for match in _DELIMITER_RUN.finditer(segment):
        start, end = match.span()
        before = segment[start - 1 : start] or " "
        after = segment[end : end + 1] or " "
        run = [start, end, segment[start], end - start]
        runs.append(run)
        stack = openers[run[2]]
        if not before.isspace() and not _is_word_char(after):  # it can close
            while run[3] and stack:
                used = min(run[3], stack[-1][3])
                run[3] -= used
                stack[-1][3] -= used
                if not stack[-1][3]:
                    stack.pop()
        if run[3] and not after.isspace() and not _is_word_char(before):  # it can open
            stack.append(run)
    kept: list[str] = []
    cursor = 0
    for start, end, character, unmatched in runs:
        kept += [segment[cursor:start], character * unmatched]
        cursor = end
    kept.append(segment[cursor:])
    return "".join(kept)


def _is_word_char(character: str) -> bool:
    return character.isalnum() or character == "_"


def _split_sections(markdown: str) -> tuple[list[ParsedSection], str | None]:
    """Sections split on ATX heading lines, never inside a fenced code block
    (fence contents are kept verbatim). Text before the first heading gets the
    title ""; sections with no text are dropped. Also returns the first
    heading's title, even when its own section was dropped."""
    sections: list[ParsedSection] = []
    first_heading: str | None = None
    title = ""
    lines: list[str] = []  # the current section, prose already de-emphasized
    prose: list[str] = []  # prose lines not yet de-emphasized (stripped as a block)
    fence: str | None = None

    def end_prose() -> None:
        if prose:
            lines.append(_strip_emphasis("\n".join(prose)))
            prose.clear()

    def end_section() -> None:
        end_prose()
        text = "\n".join(lines).strip()
        if text:
            sections.append(ParsedSection(title=title, page=None, text=text))
        lines.clear()

    for line in markdown.split("\n"):
        if fence is not None:
            lines.append(line)
            closing = line.strip()
            if closing.startswith(fence) and not closing.strip("`"):
                fence = None
        elif opening := _FENCE_OPEN.match(line):
            end_prose()
            fence = opening.group(1)
            lines.append(line)
        elif heading := _HEADING.match(line):
            end_section()
            raw_title = _CLOSING_HASHES.sub("", heading.group(1))
            title = " ".join(_strip_emphasis(raw_title).split())
            if first_heading is None and title:
                first_heading = title
        else:
            prose.append(line)
    end_section()
    return sections, first_heading


# --- metadata ----------------------------------------------------------------

_ARTICLE_TYPES = frozenset({"Article", "NewsArticle", "BlogPosting", "Report", "ScholarlyArticle"})
_PAGE_TYPES = frozenset({"WebPage", "MedicalWebPage"})
_DOI_LIKE = re.compile(r"\s*(?:https?://(?:dx\.)?doi\.org/|doi:|10\.\d{4,9}/)", re.IGNORECASE)
_ISO_DATE = re.compile(r"(\d{4})(?:[-/](\d{1,2})(?:[-/](\d{1,2}))?)?(?:[T ]\d{1,2}:\d{2}.*)?")


class _MarkupParser(HTMLParser):
    """Collects what a page says about itself: <meta name|property> pairs in
    document order, every JSON-LD block's raw text, and the first <title>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: list[tuple[str, str]] = []
        self.jsonld: list[str] = []
        self.title: str | None = None
        self._capturing: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "meta":
            key = attributes.get("name") or attributes.get("property")
            content = attributes.get("content")
            if key and content is not None:
                self.meta.append((key.strip().lower(), content))
        elif tag == "script":
            kind = (attributes.get("type") or "").split(";")[0].strip().lower()
            if kind == "application/ld+json":
                self._capturing, self._parts = "script", []
        elif tag == "title" and self.title is None and self._capturing is None:
            self._capturing, self._parts = "title", []

    def handle_endtag(self, tag: str) -> None:
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
    """Metadata by fixed precedence — the first non-empty source wins:

    title       citation_title > JSON-LD headline/name > og:title > <title>
    authors     citation_author (all) > JSON-LD author names > meta author (one)
    publisher   citation_publisher > JSON-LD publisher name > og:site_name
    date        citation_publication_date > citation_date > JSON-LD datePublished
                > article:published_time
    doi         the first VALID DOI from citation_doi, then JSON-LD identifier/sameAs
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
    # Article-like nodes describe the content itself and come first; a WebPage
    # node describes its container (Yoast lists it first, named "Title - Site
    # Name"). The sort is stable, so document order holds within each group.
    works = sorted(
        (node for node in nodes if _types(node) & (_ARTICLE_TYPES | _PAGE_TYPES)),
        key=lambda node: not _types(node) & _ARTICLE_TYPES,
    )

    def from_works(read: Callable[[dict], object]):
        return next((value for node in works if (value := read(node))), None)

    title = (
        first("citation_title")
        or from_works(lambda node: _clean(_first_str(node.get("headline"))))
        or from_works(lambda node: _clean(_first_str(node.get("name"))))
        or first("og:title")
        or _clean(parser.title)
    )
    authors = (
        meta.get("citation_author")
        or from_works(lambda node: _names(node.get("author"), by_id, people_only=True))
        or meta.get("author", [])[:1]
    )
    publisher = (
        first("citation_publisher")
        or from_works(lambda node: next(iter(_names(node.get("publisher"), by_id)), None))
        or first("og:site_name")
    )
    published = (
        first("citation_publication_date", "citation_date")
        or from_works(lambda node: _clean(_first_str(node.get("datePublished"))))
        or first("article:published_time")
    )
    doi_candidates = [
        *meta.get("citation_doi", []),
        *(candidate for node in works for candidate in _doi_candidates(node)),
    ]
    doi = next((doi for candidate in doi_candidates if (doi := clean_doi(candidate))), None)
    return PageMetadata(
        title=title,
        authors=_dedupe(authors),
        publisher=publisher,
        publication_date=_normalize_date(published) if published else None,
        doi=doi,
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
        except ValueError as exc:
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


def _types(node: dict) -> set[str]:
    declared = node.get("@type")
    names = declared if isinstance(declared, list) else [declared]
    # "schema:Article" and "https://schema.org/Article" name the same type.
    return {n.rsplit("/", 1)[-1].rsplit(":", 1)[-1] for n in names if isinstance(n, str)}


def _names(value: object, by_id: dict[str, dict], *, people_only: bool = False) -> list[str]:
    """Names from a string, an object, or a list of either. An object that is
    only an {"@id": ...} reference resolves against the page's graph."""
    names: list[str] = []
    for item in value if isinstance(value, list) else [value]:
        name = item
        if isinstance(item, dict):
            node = item
            if "name" not in node and isinstance(node.get("@id"), str):
                node = by_id.get(node["@id"], node)
            if people_only and any(kind.endswith("Organization") for kind in _types(node)):
                continue  # authors are personal names, as in SourceMetadata
            name = _first_str(node.get("name"))
        if cleaned := _clean(name):
            names.append(cleaned)
    return names


def _doi_candidates(node: dict) -> list[str]:
    """DOI-shaped strings from identifier (a string, a PropertyValue, or a list
    of either) and sameAs. Only DOI-shaped values reach clean_doi, so profile
    links in sameAs are not reported as malformed DOIs."""
    candidates: list[str] = []
    for key in ("identifier", "sameAs"):
        value = node.get(key)
        for item in value if isinstance(value, list) else [value]:
            declared_doi = False
            if isinstance(item, dict):
                declared_doi = str(item.get("propertyID", "")).strip().lower() == "doi"
                item = item.get("value")
            if isinstance(item, str) and (declared_doi or _DOI_LIKE.match(item)):
                candidates.append(item)
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
    """Entity-decoded, whitespace-collapsed and stripped; empty or non-string is None."""
    if not isinstance(value, str):
        return None
    return " ".join(unescape(value).split()) or None


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
