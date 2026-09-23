"""Bibliography scan: the works a report cites, read from its own closing pages.

A pre-upload aid. When the user picks the report PDF, the frontend asks for the
report's reference list so it can show which cited works are missing from the
user's sources and offer the free copies as links. Nothing here persists: no
run, no upload row, no file — the spooled upload is read in place.

Two judgments, split the way the rest of the pipeline splits them: the model
turns printed reference entries into fields (the language judgment), and code
decides where the bibliography is, splits it into the calls the output budget
allows, caps what the model sees and returns, and checks every address before
it is offered. The prompt is the mirror image of
credibility.METADATA_SYSTEM, which tells the model to IGNORE reference lists:
here every entry describes a work the report CITES, never the report.

Retrievability comes from Unpaywall, keyed by the DOI an entry PRINTS: a free
copy (a PDF, or a landing page — for many genuinely open works Unpaywall's
`url_for_pdf` is null and `url` is the landing page) is offered as a link the
user may add; a closed work is listed as paywalled; a work without a DOI, or
one the registry does not know, is unknown. Nothing is fetched here: an
address passes the gate a pasted link passes, plus the one refusal the
fetcher would make without a network — a literal private or local address —
so no link is offered that ingest would then refuse.
"""

import ipaddress
import os
import re
import resource
import sys
import threading
import traceback
from bisect import bisect_right
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from itertools import accumulate
from pathlib import Path
from typing import BinaryIO, Literal, NamedTuple
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader
from pypdf.generic import IndirectObject

from authorai.credibility import clean_doi, get_json_with_retries, registry_client
from authorai.fetch import (
    is_public_address,
    is_youtube_url,
    shown_text,
    url_host,
    validate_source_url,
)
from authorai.llm import LLM
from authorai.log import setup_logger
from authorai.web import (
    ExtractionTimeoutError,
    ReaderExitedError,
    end_with_parent,
    handover_file,
    in_bounded_child,
)

logger = setup_logger(__name__)

# Bounds are code, not schema: structured outputs do not honour max_length
# and a client-side validation failure would 500 the scan (see
# extract_references). Measured on the example reports: the largest real
# bibliography, the Drought report's 11-page Works Cited list, runs 61,616
# characters from its first running header to the end of its last page (an
# earlier measurement, 57,173, ran from the first running header to the
# last and left out that last page's entries), so 65,000 characters reads
# all of it with margin. The model reads that text in REFERENCE_CHUNK_CHARS
# pieces (below): at ~4 characters a token a 65,000-character list is ~6
# chunks and ~16k Haiku input tokens, and the Drought list's ~224 entries
# come back as ~25k output tokens across the pool — roughly $0.15 for the
# largest real list; a typical report's list is a page or two, under a
# cent. MAX_REFERENCES is the most the scan keeps once the chunks' answers
# are concatenated (extract_references trims a longer answer, loudly). It
# is applied AFTER every chunk has answered, not as an early stop, so the
# true worst case is a 65,000-character list of very short entries (~64
# characters, ~1,000 entries): still all ~6 calls, ~55k output tokens,
# about $0.30 at Haiku's $1/$5 per million, with ~700 entries then
# discarded. Accepted as is: an early stop would have to read the chunks
# in waves and hold the later ones until every call of the wave before
# returned — latency on every long list, the Drought report's included,
# to save a fraction of a dollar on lists the scan cuts to 300 anyway. 600
# pages from the end covers any whole report the pipeline accepts while
# bounding a hostile file's page walk.
REFERENCE_MAX_CHARS = 65_000
MAX_REFERENCES = 300
REFERENCE_MAX_PAGES = 600

# What bounds the READ, as opposed to the result: pypdf flattens the whole
# page tree before a page can be taken from its end, walking /Kids with no
# visited set and no depth guard, so a tree whose interior nodes each list
# one child twice yields 2**depth pages from a file of a few hundred bytes
# (a billion at depth 30 — an hour of CPU, then no memory left). The read
# therefore happens in a spawned child under the wall-clock budget, CPU
# limit and parent-death watcher web.in_bounded_child gives a fetched page,
# and before the child lets pypdf flatten anything it walks the tree itself:
# the root's declared /Count over this ceiling is refused at once, and so
# is a walk that reaches a node twice (a shared child, a cycle) or visits
# more nodes than this — so the flatten it then permits costs at most this
# many nodes, whatever the file declares. 10,000 is far past any report
# the pipeline accepts (the layout pass takes seconds a page) while a real
# tree of that size walks in milliseconds. The address-space cap is the
# child's bound on what pypdf's FlateDecode can inflate (zlib.decompress
# with no output limit in the pinned 3.17.4): Linux enforces RLIMIT_AS;
# macOS refuses to set it, and there the deadline and the CPU limit remain
# the reader's bounds.
PAGE_TREE_CEILING = 10_000
READER_ADDRESS_SPACE_BYTES = 1 << 30

# The memory bound where the address-space cap is not one. macOS refuses
# RLIMIT_AS, and there a FlateDecode bomb — 1.17 MB that inflates to 1.2 GB
# — was read in 43 seconds at 1.27 GB resident, inside the deadline, and
# the scan went on. So the child also runs a watchdog thread (_watch_memory,
# started before pypdf runs) that reads the process's peak resident size
# every quarter second — ru_maxrss, which macOS reports in BYTES and Linux
# in KILOBYTES; peak_resident_bytes normalises by platform — and ends the
# process with READER_MEMORY_EXIT_CODE once it passes READER_MEMORY_BYTES,
# which read_pages reports as "too costly to read". zlib releases the GIL
# while it inflates, so the thread gets its turn during the very call that
# grows the process; a quarter second bounds the overshoot (inflate runs at
# about a gigabyte a second: a few hundred MB past the line, on an 8 GB
# machine). 768 MiB is more than five times what the largest example report
# needs (139 MiB for the 66-page 2024 GHI, interpreter included: pypdf holds
# a page's streams, not the file). Where RLIMIT_AS is enforced (Linux) it
# stays the first line of defence, and the MemoryError it raises ends the
# child with the same code. The exit code is one no other ending uses:
# Python's 1 is an unhandled exception and 2 a usage error; a signal reads
# as a negative status.
READER_MEMORY_BYTES = 768 * 1024 * 1024
READER_MEMORY_EXIT_CODE = 75
READER_WATCH_INTERVAL_SECONDS = 0.25

# Every other model-written field is bounded in code after parsing too
# (extract_references), for the same reason the entry is: a title or an
# author name is shown in the dialog and matched against the user's
# sources, and the schema cannot enforce a length. Real values are far
# below — a long title is ~200 characters, an author list of 50 names is
# a consortium paper's — so a cut changes nothing real.
TITLE_MAX_CHARS = 500
MAX_AUTHORS = 50
AUTHOR_MAX_CHARS = 200

# The output budget, not the input, bounds one call: PARSE_MAX_TOKENS
# (16,000) cannot carry a whole long list — the Drought list's ~224 entries
# with a verbatim `entry` would need 20-32k output tokens, so one call would
# be cut off and 500 the scan on the very report the caps were sized for.
# The text is therefore split into chunks of at most this many characters
# (~3k tokens in, ~45 entries, ~5k tokens out with the prefix `entry`), each
# its own call, LOOKUP_WORKERS at a time. A cut falls on a line break, never
# inside a line, so an entry is split between at most two chunks, and the
# model is told to list such a fragment as printed. `entry` is a short
# verbatim PREFIX — enough to find the entry on the page, and otherwise the
# field that dominates the output — cut to ENTRY_PREFIX_CHARS in code after
# parsing (the prompt asks for that many; the cap is not in the schema).
REFERENCE_CHUNK_CHARS = 12_000
ENTRY_PREFIX_CHARS = 160

# How far apart, in PAGES, two heading matches may sit and still be one
# section: a bibliography that spans several pages repeats its heading on
# each page as a running header, and some layouts print it on alternate
# pages only (recto or verso), so a heading at most two pages before the
# next belongs to the same run, and walking back through the run reaches
# its first page. A chapter-end reference list or a contents-page mention
# several pages back stays a separate section. Pages, not characters: the
# rule is independent of how dense a page is, and of the cap, which bounds
# what the model reads, not what counts as one list.
HEADING_RUN_PAGES = 2

# httpx.Client is thread-safe and so is the Anthropic client: four calls at
# a time keep a long list's chunk calls, and its Unpaywall lookups, to
# seconds instead of a serial crawl, without leaning on either service
# (Unpaywall asks for polite use, not a rate). The lookup phase as a whole
# has a deadline: a registry that answers slowly but inside its per-request
# timeout never FAILS, and 300 DOIs four at a time at 9 s each is eleven
# minutes with the dialog waiting on a sync endpoint. 45 s is a generous
# multiple of the typical phase (a few seconds for a real list) and of the
# model calls the dialog has already waited for; at the deadline the scan
# answers with what was resolved, flagged, like a failure.
LOOKUP_WORKERS = 4
LOOKUP_DEADLINE_SECONDS = 45.0

TextSource = Literal["heading", "tail", "none"]


class ClosingText(NamedTuple):
    """What the model reads, where it came from, and whether the cap cut it:
    a cut is reported to the dialog (the scan's `limits`), not only logged,
    since a list read in part must not present the unread works as absent
    from the user's sources."""

    text: str
    source: TextSource
    truncated: bool


# A line that IS a reference-list heading, in the forms pypdf prints from
# real reports (the lines are pinned in the tests, cited by file and page):
# optionally numbered ("7. References", "39Literature Cited" — a page number
# glued before it), optionally colon-terminated, and — the running-header
# forms — with a page number after it ("BIBLIOGRAPHY    51") or a title on
# either side of a "|" ("2024 Global Hunger Index | Bibliography  51", "52
# Bibliography  | 2024 Global Hunger Index"); nothing else on the line. A
# mid-line mention ("see References"), a title glued to a sentence's end
# ("…175–179. BIBLIOGRAPHY", which such a page also carries as a footer) or
# a contents leader ("Works Cited ....... 40") never matches. `before`,
# `page` and `after` mark a running header, which names its PAGE rather
# than a position on it — see reference_text. Extend the forms only with
# evidence: a miss falls back to the document tail, which still works.
#
# Matched against ONE LINE at a time (_headings, fullmatch), never against
# the joined text: a pattern anchored with MULTILINE ^\s* re-consumed a run
# of whitespace from every line start, and a second run inside `before`
# overlapped it, so 100,000 spaces cost 33 seconds of the request thread
# after the bounded child had returned. Here only " " and "\t" are space
# (\s would run across newlines), and every repeat is possessive (*+, ++)
# so a line is read once: where the pattern must choose — the colon, the
# page number — an optional group tries its own spaces and gives them back
# whole, never a character at a time.
_HEADING = re.compile(
    r"[ \t]*+(?P<heading>"
    r"(?P<before>[^|]*+\|[ \t]*+)?"
    r"(?:\d++[. \t]*+)?"
    r"(?:references|bibliography|works cited|reference list|literature cited)"
    r"(?:[ \t]*+:)?"
    r"(?P<page>[ \t]++\d{1,4})?"
    r"[ \t]*+(?P<after>\|.*+)?"
    r")[ \t]*+",
    re.IGNORECASE,
)


class _Heading(NamedTuple):
    start: int  # where the heading word begins in the joined text
    running: bool  # a running-header form: it names its page, not a position


def _headings(text: str) -> list[_Heading]:
    """Every line of `text` that is a heading (_HEADING), in order. One
    fullmatch per line, bounded to the line's own characters, so the cost
    is the text's length whatever it holds."""
    found: list[_Heading] = []
    start = 0
    while True:
        end = text.find("\n", start)
        if end == -1:
            end = len(text)
        match = _HEADING.fullmatch(text, start, end)
        if match:
            running = any(match.group(name) for name in ("before", "page", "after"))
            found.append(_Heading(match.start("heading"), running))
        if end == len(text):
            return found
        start = end + 1


def read_pages(
    handle: BinaryIO, *, timeout: float, max_pages: int = REFERENCE_MAX_PAGES
) -> list[str]:
    """The text of the LAST `max_pages` pages, one string per page (empty for
    a page without text, such as a scan), read in a child stopped after
    `timeout` seconds of wall clock (the endpoint passes
    Settings.extract_timeout_seconds).

    pypdf, not Docling: a layout pass is seconds per page, far too slow for a
    dialog that must answer while the user is still picking sources, and the
    reference list needs no layout. `strict=False` tolerates the malformed
    xref tables real PDFs carry; a file encrypted with an empty user password
    (print/copy restrictions) is opened as readers do. Anything pypdf cannot
    read — its own exception family, or the KeyError/RecursionError a hostile
    file provokes — is one ValueError, the caller's 400; so is a file the
    child cannot finish with inside its budget — the deadline, the CPU
    limit's SIGXCPU, the memory bound (the watchdog's exit code, or the
    address-space cap's), an out-of-memory kill — which reads as "too
    costly", never as a 500. A child that ends any other way is the
    server's fault, not the file's: an unhandled exception at start-up or
    in the reader (an import that fails, a bug) or a return with nothing
    reported propagates as the ReaderExitedError it is (a 500), its exit
    status logged by in_bounded_child and the child's traceback on stderr.
    So is a failure to hand the file to the child (no temp dir), a
    RuntimeError.
    """
    handle.seek(0)  # from the start, as pypdf itself would read it
    payload_path = handover_file(handle, "the report PDF")
    try:
        kind, payload = in_bounded_child(
            _read_in_child,
            (max_pages,),
            payload_path=payload_path,
            url="the report PDF",
            timeout=timeout,
        )
    except ExtractionTimeoutError as exc:
        raise ValueError(
            "could not read the PDF: it was too costly to read "
            f"(took longer than {timeout:g} seconds)"
        ) from exc
    except ReaderExitedError as exc:
        why = _stopped_by_a_bound(exc.exitcode)
        if why is None:
            raise
        raise ValueError(f"could not read the PDF: it was too costly to read ({why})") from exc
    if kind == "result":
        return payload
    name, message, _child_traceback = payload
    if kind == "value":  # the child's own refusal, worded for the user
        raise ValueError(f"could not read the PDF: {message}")
    raise ValueError(f"could not read the PDF ({name}: {message})")


def _stopped_by_a_bound(exitcode: int | None) -> str | None:
    """Why a reader that exited without a result was stopped by one of its
    bounds, worded for the user — or None when the exit is not a bound's.
    A bound ends the child by signal (a negative status: the CPU limit's
    SIGXCPU, an out-of-memory kill) or with the memory exit code; a plain
    exit is an unhandled exception (1) or a return with nothing reported
    (0) — a bug, which the caller must see as its own."""
    if exitcode == READER_MEMORY_EXIT_CODE:
        return "the reader needed too much memory"
    if exitcode is not None and exitcode < 0:
        return "the reader was stopped"
    return None


def _read_in_child(sender, payload_path: str, max_pages: int, cpu_seconds: int) -> None:
    """The child's side of read_pages: the file opened and unlinked at once
    (the open handle keeps it; a reader killed outright never reaches its
    own removal), the pages, or the failure as a (kind, (type name, message,
    traceback)) triple — a ValueError is the child's own refusal, anything
    else is pypdf's. A MemoryError is not reported but exited on, with
    READER_MEMORY_EXIT_CODE like the watchdog: it is the memory bound
    (RLIMIT_AS, where enforced), and a process past it may not manage the
    report."""
    end_with_parent(cpu_seconds)
    _limit_address_space(READER_ADDRESS_SPACE_BYTES)
    _watch_memory()
    try:
        with open(payload_path, "rb") as pdf_file:
            Path(payload_path).unlink(missing_ok=True)
            outcome = ("result", _last_pages(pdf_file, max_pages))
    except MemoryError:  # the address-space cap, or an allocation no machine could make
        os._exit(READER_MEMORY_EXIT_CODE)
    except Exception as exc:  # noqa: BLE001 - every reader failure is "unreadable"
        kind = "value" if isinstance(exc, ValueError) else "other"
        outcome = (kind, (type(exc).__name__, str(exc), traceback.format_exc()))
    sender.send(outcome)
    sender.close()


def _limit_address_space(limit: int) -> None:
    """Cap this process's address space at `limit` bytes, never loosening a
    stricter limit it inherited. Enforced on Linux; macOS refuses the call
    (EINVAL, which the resource module raises as ValueError), and the
    refusal is not a reader failure: the deadline and the CPU limit hold."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    wanted = min(value for value in (limit, soft, hard) if value != resource.RLIM_INFINITY)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (wanted, hard))
    except (ValueError, OSError) as exc:
        logger.debug("the platform refused an address-space limit for the reader: %s", exc)


def peak_resident_bytes() -> int:
    """This process's peak resident size in bytes: ru_maxrss, which macOS
    reports in bytes and Linux in kilobytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _watch_memory(
    limit: int = READER_MEMORY_BYTES,
    interval: float = READER_WATCH_INTERVAL_SECONDS,
    stop: threading.Event | None = None,
) -> threading.Thread:
    """Start the thread that ends this process with READER_MEMORY_EXIT_CODE
    once its peak resident size passes `limit` bytes, looking every
    `interval` seconds (see READER_MEMORY_BYTES). A daemon thread: it never
    holds the process open once the read has finished. `stop` ends the
    watch (the tests; the reader lets the thread die with the process)."""
    stop = stop or threading.Event()

    def watch() -> None:
        while not stop.wait(interval):
            if peak_resident_bytes() > limit:
                os._exit(READER_MEMORY_EXIT_CODE)

    thread = threading.Thread(target=watch, name="memory-watchdog", daemon=True)
    thread.start()
    return thread


def _last_pages(handle: BinaryIO, max_pages: int) -> list[str]:
    """The text of the last `max_pages` pages of the REAL page list. pypdf
    3.17.4's `reader.pages` is a view over the trailer's declared /Count for
    an encrypted file (it flattens the tree only for an unencrypted one), so
    a stale count would drop the closing pages — the bibliography — or make
    a file every viewer opens unreadable: the list is built from the tree
    explicitly, once the walk above has bounded its cost (the pinned
    version's flatten; `flattened_pages` is the public name of its result)."""
    reader = PdfReader(handle, strict=False)
    if reader.is_encrypted:
        reader.decrypt("")
    _bound_page_tree(reader)
    reader._flatten()  # the one way to the real list, see above
    pages = reader.flattened_pages or []
    return [_sendable(page.extract_text() or "") for page in pages[-max_pages:]]


def _sendable(text: str) -> str:
    """The page's text as a str the model request can carry. pypdf decodes a
    font's ToUnicode map with `surrogatepass`, so a broken font (a glyph
    mapped to an unpaired surrogate) yields a str that no UTF-8 encoder
    accepts, and the SDK's request encoding would raise before any request
    — a 500. Each such code unit becomes "?"; a valid pair is untouched."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _bound_page_tree(reader: PdfReader) -> None:
    """Refuse a page tree that would cost more than PAGE_TREE_CEILING nodes
    to flatten, BEFORE pypdf flattens it (see the constant). The declared
    /Count is checked first, as a claim; the walk is the bound: every node
    reached from the root, in pypdf's order, and a node reached twice — the
    shared child that doubles the leaves at every level, or a cycle, which
    pypdf's flatten never leaves — is refused at its second visit."""
    catalog = reader.trailer["/Root"]
    root = catalog.raw_get("/Pages")
    count = root.get_object().get("/Count")
    if isinstance(count, int) and count > PAGE_TREE_CEILING:
        raise ValueError(
            f"the file declares {count} pages, more than the {PAGE_TREE_CEILING} the scan can walk"
        )
    seen: set = set()
    pending = [root]
    while pending:
        reference = pending.pop()
        key = (
            (reference.idnum, reference.generation)
            if isinstance(reference, IndirectObject)
            else id(reference)
        )
        if key in seen:
            raise ValueError("the file's page tree is not a tree (a page node is reached twice)")
        seen.add(key)
        if len(seen) > PAGE_TREE_CEILING:
            raise ValueError(f"the file's page tree has more than {PAGE_TREE_CEILING} nodes")
        node = reference.get_object()
        if not isinstance(node, dict):
            continue
        kids = node.get("/Kids")
        if kids is not None:
            kids = kids.get_object()
        if isinstance(kids, list):
            pending.extend(kids)


def reference_text(pages: list[str], *, max_chars: int = REFERENCE_MAX_CHARS) -> ClosingText:
    """The closing text the model should read, where it came from, and
    whether `max_chars` cut it.

    From the report's reference-list heading forward when it prints one
    (the 51-page Drought report's 72,699-character tail starts inside its
    body; its 11-page Works Cited list starts at the heading), else the
    last `max_chars` of the document (two of four test reports print no
    heading). No text at all is ("", "none"): a scanned PDF must not yield an
    invented bibliography, so the caller makes no model call on that value.

    Which heading: the LAST one — a chapter's own list or a contents-page
    line comes earlier — except that a bibliography spanning several pages
    prints its heading on each page (or every other page) as a running
    header, and slicing from the last of those would drop every earlier
    page. So the rule, in this one place: take the last heading match, walk
    back while the previous match is at most HEADING_RUN_PAGES pages
    earlier, and start at the earliest heading reached, keeping the first
    `max_chars` from there. Each step is judged against the heading before
    it, not against the last one, so a run of any length joins. A heading
    farther back than that — a chapter's list, a contents line — stays
    excluded.

    Where on that page: at the heading word for a plain heading (the list
    starts under it, whatever the page held before), but at the START of
    the page for a running-header form — a page number after the heading
    or a "|"-decorated title, as the GHI reports print on every page of
    their bibliography — because a running header names the page, and
    pypdf emits a footer at the END of the page's text: anchoring there
    would drop the first page's entries, which lie before it.
    """
    text = "\n".join(pages)
    if not text.strip():
        return ClosingText("", "none", False)
    headings = _headings(text)
    if not headings:
        return ClosingText(text[-max_chars:].strip(), "tail", len(text) > max_chars)
    # Where each page starts in the joined text (the join adds one newline).
    page_starts = list(accumulate((len(page) + 1 for page in pages[:-1]), initial=0))
    heading_pages = [bisect_right(page_starts, heading.start) - 1 for heading in headings]
    index = len(headings) - 1
    while index > 0 and heading_pages[index] - heading_pages[index - 1] <= HEADING_RUN_PAGES:
        index -= 1
    heading = headings[index]
    start = page_starts[heading_pages[index]] if heading.running else heading.start
    return ClosingText(
        text[start : start + max_chars].strip(), "heading", len(text) - start > max_chars
    )


def split_reference_text(text: str, *, chunk_chars: int = REFERENCE_CHUNK_CHARS) -> list[str]:
    """The text in pieces of at most `chunk_chars`, each a run of whole lines.

    A cut falls at the last blank line in the second half of the window (an
    entry boundary in most layouts; the second half, so a stray blank line
    near the start cannot make a tiny chunk), else at the window's last
    line break — never inside a line. A line longer than a whole chunk (a
    PDF whose text lost its line breaks) is cut at its last space or, with
    none, at the cap: the output budget is the harder bound. Only the
    separator at a cut is dropped — the line break(s) there, or the one
    space — so a line that opens a chunk keeps its indentation; every line
    reaches exactly one chunk, intact and in order.
    """
    chunks: list[str] = []
    rest = text
    while len(rest) > chunk_chars:
        window = rest[:chunk_chars]
        cut = window.rfind("\n\n")
        if cut < chunk_chars // 2:
            cut = window.rfind("\n")
        if cut <= 0:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = chunk_chars
        chunks.append(rest[:cut])
        rest = rest[cut:]
        # The separator at the cut and nothing else: every line break there
        # (a blank line is two), or the one space of a space cut. A hard cut
        # sits on no separator. The next line's indentation is the line's.
        rest = rest.lstrip("\n") if rest.startswith("\n") else rest.removeprefix(" ")
    chunks.append(rest)
    return chunks


class Reference(BaseModel):
    """One printed reference entry, as fields. `entry` is the anchor: the
    start of the text as printed, shown to the user when the title is null
    and the proof that the row came from the page rather than from memory.
    A prefix, not the whole entry: the whole would dominate the output
    budget (see REFERENCE_CHUNK_CHARS), and the first line finds the entry
    on the page just as well. A blank `entry` is a model slip; whether the
    row survives it is `actionable`'s rule."""

    title: str | None = Field(default=None, description="The cited work's title, as printed")
    authors: list[str] = Field(
        default_factory=list, description="Author names as printed; empty when none are printed"
    )
    year: int | None = Field(default=None, description="Publication year, as printed")
    doi: str | None = Field(
        default=None, description="The DOI printed in the entry (10.xxxx/...), without a URL prefix"
    )
    url: str | None = Field(default=None, description="A web address printed in the entry")
    entry: str = Field(
        description=(
            f"The first {ENTRY_PREFIX_CHARS} characters of the reference entry exactly as "
            "printed (line breaks joined)"
        )
    )

    @field_validator("title", "doi", "url", mode="after")
    @classmethod
    def _blank_is_absent(cls, value: str | None) -> str | None:
        # The model sometimes prints '' where it was told to leave null; the
        # DOI lookup and the printed-URL rule key on absence.
        if value is None:
            return None
        return value.strip() or None

    @field_validator("entry", mode="after")
    @classmethod
    def _entry_stripped(cls, value: str) -> str:
        # Stripped, never None: `entry` is a str, and a value the field's own
        # type rejects would fail the endpoint's re-validation (a 500 for a
        # blank the model printed). A blank one is judged by `actionable`.
        return value.strip()

    @field_validator("authors", mode="after")
    @classmethod
    def _drop_blank_authors(cls, value: list[str]) -> list[str]:
        return [name.strip() for name in value if name.strip()]


class ReferenceList(BaseModel):
    references: list[Reference] = Field(default_factory=list)


def actionable(reference: Reference) -> bool:
    """Whether the row is one the user can act on — the rule, in this one
    place. A reference prints text (`entry`, the proof it came from the
    page) or names its work by title, DOI or address, which the dialog can
    show, match against the user's sources and look up without the text. A
    row with none of these — a model slip, `entry` printed as "" where the
    prompt asked for the entry's first characters — is not a reference:
    extract_references drops it and counts the drops in a warning. Authors
    and a year alone name nothing the dialog can show, so they do not keep
    a row."""
    return bool(reference.entry or reference.title or reference.doi or reference.url)


REFERENCES_SYSTEM = f"""\
You read the closing pages of a report and list the works its reference list
cites. Every entry you return describes a work the report CITES — never the
report itself. Report ONLY what the text actually prints — never guess, never
complete an entry from world knowledge. A field the entry does not print is
null.

- `entry`: the first {ENTRY_PREFIX_CHARS} characters of the entry exactly as
  printed (line breaks joined) — enough for a reader to find it on the page.
  Never the whole of a longer entry.
- `title`: the cited work's title as printed. Null when the entry prints none.
- `authors`: the names as printed, personal or organizational. Empty list if
  none are printed.
- `year`: the publication year as printed.
- `doi`: ONLY a DOI printed in the entry (10.xxxx/...), without a URL prefix.
  Never supply a DOI from memory, even for a famous paper.
- `url`: ONLY a web address printed in the entry.

In-text citations such as "(Smith, 2020)", footnote markers, figure and table
sources, and the report's own title and imprint are not entries. The list
may be long — a report can cite a few hundred works: list every entry it
prints, in order, and never stop early or summarize. A long list arrives in
parts, each its own message saying which part it is; a part may begin or end
in the middle of an entry, and such a fragment is listed as an entry with
exactly what is printed in the part, never completed. A document without a
reference list yields an empty list — that is the correct answer, not a
failure.
"""


def _chunk_prompt(index: int, total: int, chunk: str) -> str:
    """The user message for one chunk: the list itself when it fits one
    call, else which part this is and the one rule a part needs."""
    if total == 1:
        return f"CLOSING PAGES:\n\n{chunk}"
    return (
        f"CLOSING PAGES, part {index + 1} of {total} of the reference list. This part may "
        "begin or end in the middle of an entry: list such a fragment as an entry with "
        f"exactly what is printed here, never completed.\n\n{chunk}"
    )


def _bounded(reference: Reference) -> Reference:
    """The reference with `entry` cut to ENTRY_PREFIX_CHARS — the prompt asks
    for that many, and the model does not always count — and the title and
    author names to their caps (see the constants)."""
    return reference.model_copy(
        update={
            "entry": reference.entry[:ENTRY_PREFIX_CHARS],
            "title": reference.title[:TITLE_MAX_CHARS] if reference.title else reference.title,
            "authors": [name[:AUTHOR_MAX_CHARS] for name in reference.authors[:MAX_AUTHORS]],
        }
    )


class Extraction(NamedTuple):
    """The references kept, and how many rows were dropped on the way —
    blank rows and the rows past MAX_REFERENCES — reported to the dialog as
    the scan's `limits.references_dropped`."""

    references: list[Reference]
    dropped: int


def extract_references(llm: LLM, model: str, closing_text: str) -> Extraction:
    """The model's reading of the closing text: one structured call per
    REFERENCE_CHUNK_CHARS chunk, LOOKUP_WORKERS at a time, the answers
    concatenated in chunk order, every `entry` cut to ENTRY_PREFIX_CHARS
    (title and authors to their caps, likewise), rows that are not
    `actionable` dropped (counted in a warning), and the
    list cut to MAX_REFERENCES in code. The caps are deliberately NOT in the
    schema, where structured outputs may not honour them and a validation
    failure would fail the whole scan instead of trimming it.

    A chunk whose call fails fails the scan — the endpoint's 500, as for any
    model failure. The other chunks' entries are not returned in its place:
    a bibliography with a silent hole would tell the dialog that the unread
    works are absent from the user's sources, and the user would go looking
    for works that were merely unread. Chunks not yet started are cancelled;
    those in flight, at most LOOKUP_WORKERS, finish and are discarded.
    """
    chunks = split_reference_text(closing_text)

    def extract(index: int) -> ReferenceList:
        return llm.parse(
            model=model,
            system=REFERENCES_SYSTEM,
            prompt=_chunk_prompt(index, len(chunks), chunks[index]),
            output_type=ReferenceList,
        )

    # Executor.map yields in submission order whatever order the calls finish
    # in, raises a failure when its chunk's turn comes, and on the way out
    # cancels the chunks not yet started; leaving the block waits for the
    # in-flight ones.
    with ThreadPoolExecutor(max_workers=LOOKUP_WORKERS) as pool:
        parts = list(pool.map(extract, range(len(chunks))))
    answered = [_bounded(reference) for part in parts for reference in part.references]
    references = [reference for reference in answered if actionable(reference)]
    if len(references) < len(answered):
        logger.warning(
            "dropping %d of %d references with no printed entry and no title, DOI or address",
            len(answered) - len(references),
            len(answered),
        )
    if len(references) > MAX_REFERENCES:
        logger.warning(
            "the model returned %d references over %d parts — keeping the first %d",
            len(references),
            len(chunks),
            MAX_REFERENCES,
        )
        references = references[:MAX_REFERENCES]
    return Extraction(references, len(answered) - len(references))


# --- Unpaywall ------------------------------------------------------------------

UNPAYWALL_BASE = "https://api.unpaywall.org"
UNPAYWALL_TIMEOUT = 10.0
UNPAYWALL_RETRIES = 2

Retrievability = Literal["pdf", "landing", "paywalled", "unknown"]
Resolved = tuple[Retrievability, str | None]

NOT_RESOLVED: Resolved = ("unknown", None)


class UnpaywallClient:
    """DOI → Unpaywall record, under the registry-GET policy credibility.py
    established: 200 is a payload; 404 (an HTML page, live) is an answer,
    None; 429/5xx and transport failures retry, then raise. A contact email
    is required, not polite: Unpaywall answers HTTP 422 without one, so an
    empty address is refused at construction rather than on every request."""

    def __init__(self, mailto: str, timeout: float = UNPAYWALL_TIMEOUT):
        if not mailto or not mailto.strip():
            raise ValueError(
                "Unpaywall needs a contact email (AUTHORAI_CROSSREF_MAILTO) — "
                "it refuses requests without one"
            )
        self._mailto = mailto.strip()
        self._client = registry_client(self._mailto, base_url=UNPAYWALL_BASE, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def by_doi(self, doi: str) -> dict | None:
        # clean_doi rejects anything not DOI-shaped BEFORE it enters the path:
        # the DOI is model-extracted text, and an unvalidated string would
        # look up something else (a `?` starts the query string).
        cleaned = clean_doi(doi)
        if cleaned is None:
            return None
        return get_json_with_retries(
            self._client,
            f"/v2/{quote(cleaned, safe='/')}",
            {"email": self._mailto},
            retries=UNPAYWALL_RETRIES,
            provider="Unpaywall",
        )


def offerable_url(url: str) -> str:
    """The address in the form a pasted link takes, or ValueError when the
    app would refuse the link: every refusal the upload dialog's own link
    check makes (lib/links.ts checkLink) — the syntax gate a pasted link
    passes and the YouTube refusal `POST /api/runs` makes, so the scan never
    suggests a link the dialog then refuses to add — then the one refusal
    the fetcher would make that needs no network. A LITERAL address host
    must pass fetch.is_public_address — the gate the fetcher applies to
    every resolved hop — and `localhost`, or any name under it, is the local
    machine by definition. A NAME that resolves to private space is
    deliberately not caught here: a pre-upload aid does no DNS, and the
    fetch gate resolves and pins every hop at ingest, where such a link is
    refused — and so is a numeric spelling of an address that ip_address
    does not parse ("2130706433", "127.1", "0x7f.0.0.1"), which is a NAME
    to this gate and a loopback literal to the resolver: it is offered here
    and refused at ingest, where every resolved answer is gated. An offer
    is only ever as good as that gate."""
    normalized = validate_source_url(url)
    if is_youtube_url(normalized):
        raise ValueError(f"Source URL {normalized!r}: YouTube links are not supported yet")
    host = url_host(normalized) or ""
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError(f"Source URL {normalized!r} names the local machine")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        return normalized  # a name: resolved, and gated, at ingest
    if not is_public_address(literal):
        raise ValueError(f"Source URL {normalized!r} names a private or reserved address")
    return normalized


def _usable(url: str, *, what: str) -> str | None:
    """The address in the form a pasted link takes, or None when the offer
    gate refuses it — an address the app would not read as a link is not
    offered as one."""
    try:
        return offerable_url(url)
    except ValueError as exc:
        logger.warning("dropping the %s address Unpaywall offered: %s", what, exc)
        return None


def retrievability(record: dict | None) -> Resolved:
    """What the record lets us offer: the kind of free copy and its address.

    A PDF beats a landing page; a landing page is still a free copy. A record
    that does not say whether the work is open is unknown, not paywalled —
    paywalled is a claim about the work, made only when the registry makes it.
    An open work whose every address fails the gate is unknown too, since an
    offer needs an address. The record is trusted for its values, never its
    types: get_json_with_retries guarantees an object at the top, and here a
    `best_oa_location` that is not one, or an address that is not a string,
    is no address (a warning) — this runs in the caller's thread, where an
    exception would be the scan's 500.
    """
    if record is None:
        return NOT_RESOLVED
    is_oa = record.get("is_oa")
    if is_oa is False:
        return ("paywalled", None)
    if is_oa is not True:
        return NOT_RESOLVED
    best = record.get("best_oa_location")
    if best is not None and not isinstance(best, dict):
        logger.warning("Unpaywall's best_oa_location is a %s, not an object", type(best).__name__)
        best = None
    best = best or {}
    candidates: list[tuple[Retrievability, object]] = [
        ("pdf", best.get("url_for_pdf")),
        ("landing", best.get("url_for_landing_page")),
        ("landing", best.get("url")),
    ]
    for kind, url in candidates:
        if not url:
            continue
        if not isinstance(url, str):
            logger.warning("Unpaywall's %s address is a %s, not a string", kind, type(url).__name__)
            continue
        usable = _usable(url, what=kind)
        if usable is not None:
            return (kind, usable)
    return NOT_RESOLVED


def printed_url(reference: Reference) -> str | None:
    """The address a DOI-less entry prints, through the offer gate, or None.

    Only without a DOI: with one, the registry's verdict rules (a paywalled
    work is never offered, whatever its entry prints). The address is not
    checked against any registry — the frontend labels it "printed in the
    entry, not checked" — so it passes exactly the gate a suggested address
    passes, and nothing more.
    """
    if reference.doi or not reference.url:
        return None
    try:
        return offerable_url(reference.url)
    except ValueError as exc:
        logger.info("not offering the printed address: %s", exc)
        return None


class RegistryUnavailable(RuntimeError):
    """Unpaywall gave no answer for a DOI (throttled, down, or unreachable
    after retries). Carries what WAS resolved before the failure, aligned with
    the references given, so the caller can still show the list — the
    citations are useful without retrievability — under an explicit flag,
    never as a silent "unknown"."""

    def __init__(self, message: str, resolved: list[Resolved]):
        super().__init__(message)
        self.resolved = resolved


class _Skipped(Exception):
    """A lookup that never ran: the registry had already failed."""


def lookup_retrievability(client: UnpaywallClient, references: list[Reference]) -> list[Resolved]:
    """Retrievability for every reference, in order — one lookup for each
    DISTINCT printed DOI (a report cites one work in several chapters; the
    verdict reaches every row that prints it), LOOKUP_WORKERS at a time,
    the rest NOT_RESOLVED.

    Capped at MAX_REFERENCES (extract_references already trims; this is the
    second wall). The first registry failure ends the whole lookup: pending
    lookups are cancelled, in-flight ones finish (no request is abandoned
    mid-way, and nothing runs on after the caller has the answer), and
    RegistryUnavailable carries every verdict that was reached. A failure
    is whatever a lookup raises: the retry policy's RuntimeError, or an
    httpx error it does not cover (a DecodingError from a body that
    contradicts its Content-Encoding, an InvalidURL from a DOI past httpx's
    own length limit) or a ValueError — one policy, so no failure route
    reaches the caller as a 500. So does LOOKUP_DEADLINE_SECONDS passing:
    the same RegistryUnavailable with what was resolved, except that the
    in-flight lookups are then abandoned rather than awaited — waiting on
    them is what the deadline exists to stop; the caller's close() ends
    them, and their answers are discarded. A failure in THIS thread
    (reading a result) is not the registry's and propagates, but the queue
    is cancelled on the way out all the same.
    """
    results: list[Resolved] = [NOT_RESOLVED] * len(references)
    # Raised by the failing lookup itself, before its worker returns to the
    # queue: a worker that then dequeues a pending lookup sees it and makes
    # no request. Cancelling from this thread alone is a race the worker wins.
    failed = threading.Event()

    def lookup(doi: str) -> dict | None:
        if failed.is_set():
            raise _Skipped()
        try:
            return client.by_doi(doi)
        except RuntimeError:
            failed.set()
            raise
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
            failed.set()
            raise RuntimeError(
                f"Unpaywall lookup failed: {type(exc).__name__}: {shown_text(str(exc))}"
            ) from exc

    rows_by_doi: dict[str, list[int]] = {}
    for index, reference in enumerate(references[:MAX_REFERENCES]):
        if reference.doi:
            rows_by_doi.setdefault(reference.doi, []).append(index)
    pool = ThreadPoolExecutor(max_workers=LOOKUP_WORKERS)
    futures: dict[Future, list[int]] = {
        pool.submit(lookup, doi): rows for doi, rows in rows_by_doi.items()
    }

    def resolve(future: Future) -> None:
        verdict = retrievability(future.result())
        for index in futures[future]:
            results[index] = verdict

    def keep_what_was_reached() -> None:
        for future in futures:
            if future.done() and not future.cancelled() and future.exception() is None:
                resolve(future)

    try:
        for future in as_completed(futures, timeout=LOOKUP_DEADLINE_SECONDS):
            try:
                future.result()
            except _Skipped:
                continue
            resolve(future)
    except TimeoutError:
        failed.set()
        pool.shutdown(wait=False, cancel_futures=True)
        keep_what_was_reached()
        raise RegistryUnavailable(
            f"Unpaywall lookups timed out after {LOOKUP_DEADLINE_SECONDS:g} seconds", results
        ) from None
    except RuntimeError as exc:
        pool.shutdown(wait=True, cancel_futures=True)
        keep_what_was_reached()
        raise RegistryUnavailable(str(exc), results) from exc
    finally:
        # Idempotent after the paths above. For any other escape: the queue
        # is cancelled, the flag stops a lookup dequeued meanwhile, and the
        # caller is not held for an in-flight one — its answer is discarded.
        failed.set()
        pool.shutdown(wait=False, cancel_futures=True)
    return results
