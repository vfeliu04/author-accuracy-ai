"""Web-page sources: readable article text and self-declared metadata from HTML.

One page, two independent readings:

- BODY: the page is parsed once and edited before trafilatura reads it.
  Consent banners, modal dialogs and paywall prompts are pruned (on some page
  shapes trafilatura returns them as the article). Emphasis tags are
  unwrapped: chunk text is later quoted verbatim as evidence, and markdown
  markers cannot be stripped afterwards because trafilatura does not escape a
  page's own asterisks. Super/subscripts become plain text (10^6, CO2), and a
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
  scoring), never from the hostname (publisher authority matches whole words,
  so "united-nations-fan-club.org" would borrow the United Nations'
  authority), and never from body text (reference lists describe OTHER works).

The result is a ParsedDocument like a PDF's: everything downstream of parsing
treats both the same.
"""

import codecs
import copy
import json
import math
import multiprocessing
import os
import re
import resource
import shutil
import signal
import tempfile
import threading
import time
import traceback
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date
from html import unescape
from html.parser import HTMLParser
from multiprocessing import connection, resource_tracker
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

import httpx
import trafilatura
from lxml import etree

# trafilatura's markdown writer — the function its extract() uses — applied
# per section. Internal module; trafilatura is pinned exactly in pyproject.
from trafilatura.xml import xmltotxt

from authorai.credibility import clean_doi
from authorai.fetch import shown_text, upper_escapes
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

# Bounds on what a page declares about itself: every declared string (its
# title, publisher, date, author names, and each heading, which becomes a
# chunk's section name) and how many authors are kept. The page writes these
# values and decides their length; they are stored with the run and shown in
# the UI. Real ones are far below: a long title is ~200 characters, and the
# corroboration a long author list feeds needs one matching family name.
MAX_METADATA_CHARS = 500
MAX_AUTHORS = 100


class ThinPageError(ValueError):
    """The page yielded no readable article text — typically a JavaScript-rendered shell."""


class ExtractionTimeoutError(RuntimeError):
    """Reading the page exceeded its wall-clock budget and was stopped."""


class ResultTooLargeError(RuntimeError):
    """The reader process reported a result longer than RESULT_MAX_BYTES; it
    was refused on its length header, unread. A bound of the reader's, like
    the deadline: the message names the reader's input, as every reader
    failure does."""


class ReaderExitedError(RuntimeError):
    """The reader process exited without sending a result. `exitcode` is
    its status: negative for a signal (the CPU limit's SIGXCPU, an
    out-of-memory kill), a code the child chose (os._exit), 1 for an
    unhandled exception at start-up or in its target, 0 for a return with
    nothing reported. What the status means is the caller's to decide —
    the same status is a bound for one reader and a bug for another — so
    the message only names the reader's input, as every reader failure
    does."""

    def __init__(self, message: str, exitcode: int | None) -> None:
        super().__init__(message)
        self.exitcode = exitcode


# The most a reader may REPORT, as opposed to cost. The child's deadline,
# CPU limit and memory watchdog bound its work, and each reader caps the
# text it builds (references.PAGE_TEXT_MAX_CHARS and READER_MAX_TEXT_CHARS
# for a PDF; a fetched page is at most its body cap), but a parent that
# received an unbounded frame held the frame, the unpickled value and its
# own copies at once, in the request thread, for as long as they took to
# process. recv_bytes refuses a frame on its length header, before a byte
# of the body is read (the connection is then no longer readable, as
# documented), and in_bounded_child reports the refusal as a bound of the
# reader's. 64 MiB is twice the largest result either reader can send: the
# PDF reader's 8,000,000 characters pickle as at most 32 MiB of UTF-8 (~8
# MiB for the Latin text of a real report), and a page's sections are at
# most the 10 MB body they came from.
RESULT_MAX_BYTES = 64 * 1024 * 1024


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


@dataclass(frozen=True)
class CappedText:
    """What the page cap left, and what it took.

    `truncated` is the record the caller keeps in the stored page's provenance,
    so a page read in part says so wherever the page is read back — a retry, the
    report, the source list — and not only in the server log. It holds the two
    numbers rather than a bare flag: "read in part" alone cannot tell 2% of a
    page from 95% of it. It is None when the whole page was kept, so a snapshot
    of a whole page carries nothing new and reads exactly as it always did.
    """

    sections: list[ParsedSection]
    truncated: dict | None = None


def cap_sections(sections: list[ParsedSection], *, limit: int, url: str) -> CappedText:
    """The page's sections, cut to at most `limit` characters of text (the
    caller's AUTHORAI_WEB_MAX_CHARS). Applied by the ingest step, so a stored
    page is already bounded; PDFs keep their own path.

    The fetch bounds a page's MARKUP (fetch_max_bytes, 10 MB) and nothing
    bounds what the reader gets out of it, while ingest chunks and embeds a
    document's sections in ONE list: a page at the byte cap becomes thousands
    of chunks and about a gigabyte of live Python floats, times the links a
    run may hold.

    Whole sections are kept, so no chunk ends mid-sentence, and the cut is both
    logged naming the page and reported back for the page's provenance — a
    report scored against a page we quietly truncated would misstate what was
    checked. A FIRST section that alone exceeds the limit is cut at its last
    line break instead (a page with no headings is one section: dropping it
    whole would lose the page, keeping it whole would leave the limit
    unenforced), and one with no line break at all at the limit itself.
    """
    total = sum(len(section.text) for section in sections)
    if total <= limit:
        return CappedText(sections)
    kept: list[ParsedSection] = []
    used = 0
    for section in sections:
        if used + len(section.text) <= limit:
            kept.append(section)
            used += len(section.text)
            continue
        if not kept:
            head = section.text[:limit]
            break_at = head.rfind("\n")
            text = head[:break_at].rstrip("\n") if break_at > 0 else head
            kept.append(replace(section, text=text))
            used += len(text)
        break
    logger.warning(
        "%s: page text truncated to %d of %d characters (AUTHORAI_WEB_MAX_CHARS is %d) "
        "— %d characters dropped",
        url,
        used,
        total,
        limit,
        total - used,
    )
    return CappedText(kept, {"kept_chars": used, "dropped_chars": total - used})


def extract_web_bounded(
    html: bytes | str, *, url: str, timeout: float
) -> tuple[ParsedDocument, PageMetadata]:
    """extract_web in a child process, stopped after `timeout` seconds of wall
    clock. The tree edits in _prepare_tree are linear, but trafilatura's own
    cleaning still goes quadratic over a run of inline tags it strips itself
    (<abbr>x</abbr> repeated: 6 s at 0.4 MB, an hour at the 10 MB fetch cap),
    and nothing else can interrupt the single worker thread. A fresh
    interpreter (spawned, never forked: the caller is a thread) costs about
    half a second per page, inside the budget: the page reaches the child
    through a private file (handover_file), so start() returns as soon as
    the child is exec'd and the deadline covers its start-up too.

    Raises ExtractionTimeoutError at the deadline. The child's ThinPageError
    or ValueError is re-raised here as the same type with the same message;
    any other failure of the child, or of handing it the page, is a
    RuntimeError naming the URL, with the child's traceback or exit status in
    the log.
    """
    page_path = handover_file(html, url)
    kind, payload = in_bounded_child(
        _extract_in_child,
        (isinstance(html, str), url),
        payload_path=page_path,
        url=url,
        timeout=timeout,
    )
    if kind == "result":
        return payload
    name, message, child_traceback = payload
    if kind == "thin":
        raise ThinPageError(message)
    logger.warning("reading %s failed in the reader process:\n%s", url, child_traceback)
    if kind == "value":
        raise ValueError(message)
    raise RuntimeError(f"{url} could not be read: {name}: {message}")


def in_bounded_child(
    target, args: tuple, *, payload_path: str, url: str, timeout: float
) -> tuple[str, Any]:
    """Run `target(sender, payload_path, *args, cpu_seconds)` in a spawned
    child under a wall-clock budget and return the (kind, payload) pair it
    sent back (report_outcome is the child's side of that exchange). The one
    bounded-reader primitive: extract_web_bounded reads a
    page through it, and the bibliography scan (references.read_pages) reads
    a PDF through it, so a hostile input of either kind meets one deadline,
    one CPU limit and one parent-death watcher (end_with_parent, which the
    target calls first) rather than a second copy of them.

    The payload reaches the child through the file at `payload_path`
    (handover_file), which is removed on every path — by the child once read,
    here for a child that never read it. Raises ExtractionTimeoutError at the
    deadline, with the child stopped, and ReaderExitedError (a RuntimeError)
    naming `url` when the child exits without a result (killed, out of
    memory, a frame cut short, a failed start-up), carrying its exit status,
    which is also in the log.
    """
    context = multiprocessing.get_context("spawn")
    outcome = None
    exitcode = None
    try:
        receiver, sender = context.Pipe(duplex=False)
        child = context.Process(
            target=target,
            args=(sender, payload_path, *args, _orphan_cpu_seconds(timeout)),
            daemon=True,
        )
        try:
            deadline = time.monotonic() + timeout
            _start(child)
            sender.close()  # the child holds the only writing end: its exit reads as EOF
            if not receiver.poll(max(0.0, deadline - time.monotonic())):
                raise ExtractionTimeoutError(
                    f"{url} took longer than {timeout:g} seconds to read "
                    "(the page is too large or complex)"
                )
            try:
                frame = receiver.recv_bytes(RESULT_MAX_BYTES)
            except EOFError:  # the child died before reporting
                pass
            except OSError:
                # A frame cut short (the child died while reporting: killed,
                # out of memory) — or one refused on its length header, which
                # leaves the connection closed: the reader's result was past
                # the bound, and that is a bound's verdict, not an exit's.
                if receiver.closed:
                    logger.warning(
                        "the reader process for %s reported a result larger than %d bytes "
                        "— refused unread",
                        url,
                        RESULT_MAX_BYTES,
                    )
                    raise ResultTooLargeError(
                        f"{url} could not be read: the reader reported a result larger than "
                        f"{RESULT_MAX_BYTES} bytes"
                    ) from None
            else:
                outcome = ForkingPickler.loads(frame)  # what recv() does, after its own read
            if outcome is None:
                # A child that reported nothing is judged by its exit status
                # (a bound or a bug — the caller's call), so let it end by
                # itself before _stop reaps it: one still tearing down when
                # its pipe closed would otherwise be terminated, and the
                # status would always read SIGTERM. One that closed its pipe
                # and reads on is stopped after the grace. A child that
                # reported is reaped at once: its status is never read.
                child.join(_EXIT_GRACE_SECONDS)
        finally:
            sender.close()
            receiver.close()
            if child.pid is not None:
                exitcode = _stop(child)
    finally:
        Path(payload_path).unlink(missing_ok=True)  # the child removes it once read
    if outcome is None:
        logger.warning(
            "the reader process for %s exited without a result (%s)", url, _exit_status(exitcode)
        )
        raise ReaderExitedError(
            f"{url} could not be read: the reader process exited without a result", exitcode
        )
    return outcome


# How long a child that reported nothing may take to end by itself before
# _stop terminates it: an interpreter's teardown is tens of milliseconds,
# so a real exit is never cut short, and a child that closed its pipe and
# reads on costs at most this before it is stopped.
_EXIT_GRACE_SECONDS = 1.0


def handover_file(payload: bytes | str | BinaryIO, url: str) -> str:
    """Write the payload to a private temporary file for the child to read
    back, and return its path. The payload never travels as a Process
    argument: the spawn launcher writes the arguments to the child over a
    pipe while still holding the child's end of it, so arguments past the
    pipe buffer (64 KiB) block start() until the child reads them — forever
    when the child dies first (an import that fails because a module was
    edited on disk under a running server, a kill during start-up), before
    any deadline is in force. A str is stored as UTF-8 with surrogates passed
    through, so it reads back byte-exact whatever jobs._page_text decoded; a
    readable binary handle (a spooled upload) is copied without being held
    in memory whole. The child deletes the file as soon as it has read it,
    since a server killed outright never reaches its own removal; the caller
    removes it too, for a child that never read it."""
    try:
        handle, path = tempfile.mkstemp(prefix="authorai-page-")
        try:
            with os.fdopen(handle, "wb") as page_file:
                if isinstance(payload, str):
                    page_file.write(payload.encode("utf-8", "surrogatepass"))
                elif isinstance(payload, bytes):
                    page_file.write(payload)
                else:
                    shutil.copyfileobj(payload, page_file)
        except BaseException:
            os.unlink(path)
            raise
    except OSError as exc:
        raise RuntimeError(f"{url} could not be read: {type(exc).__name__}: {exc}") from exc
    return path


def _start(child) -> None:
    """Start the reader with SIGINT blocked for its whole life. A terminal
    Ctrl-C signals the server's whole process group, the reader included; one
    killed by it reports nothing, and the run would be recorded FAILED blaming
    the link instead of staying RUNNING for startup recovery. The mask is per
    thread and survives spawn's fork+exec. The resource tracker is started
    first: its first launch UNBLOCKS these signals in the calling thread."""
    resource_tracker.ensure_running()
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        child.start()
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def report_outcome(
    sender, work: Callable[[], Any], failure_kind: Callable[[Exception], str]
) -> None:
    """The child's side of in_bounded_child's exchange, in one place: the
    value `work` returns is sent back as ("result", value), the exception
    it raises as (failure_kind(exc), (type name, message, traceback)) —
    exceptions themselves need not pickle, and which failures a reader
    tells apart is its own rule. The pipe is closed once the outcome is
    sent."""
    try:
        outcome = ("result", work())
    except Exception as exc:  # noqa: BLE001 - every failure is reported; the parent judges it
        outcome = (failure_kind(exc), (type(exc).__name__, str(exc), traceback.format_exc()))
    sender.send(outcome)
    sender.close()


def _extract_in_child(sender, page_path: str, is_text: bool, url: str, cpu_seconds: int) -> None:
    """The child's side of extract_web_bounded: the page read back from its
    file, then the result or the failure, reported through report_outcome —
    a thin page and a ValueError each by their own kind, anything else as
    "other"."""
    end_with_parent(cpu_seconds)

    def read_and_extract():
        with open(page_path, "rb") as page_file:
            raw = page_file.read()
        Path(page_path).unlink(missing_ok=True)
        html = raw.decode("utf-8", "surrogatepass") if is_text else raw
        return extract_web(html, url=url)

    report_outcome(sender, read_and_extract, _failure_kind)


def _failure_kind(exc: Exception) -> str:
    if isinstance(exc, ThinPageError):
        return "thin"
    return "value" if isinstance(exc, ValueError) else "other"


def end_with_parent(cpu_seconds: int) -> None:
    """Tie the reader's life to its parent's. Only the parent stops a reader
    (_stop at the deadline, multiprocessing's exit hook for daemons), and a
    server killed outright — SIGTERM's default action after uvicorn's graceful
    stop, SIGKILL, a crash — runs neither: the orphan would read on for as long
    as the page takes, and startup recovery would start another beside it. A
    watcher exits the moment the parent is gone (its sentinel reads EOF), and
    the kernel's CPU limit ends a reader the watcher cannot reach — one inside
    a long C call holding the GIL — with neither the parent nor the GIL."""
    parent = multiprocessing.parent_process()
    threading.Thread(
        target=lambda: (connection.wait([parent.sentinel]), os._exit(1)), daemon=True
    ).start()
    _limit_cpu(cpu_seconds)


def _limit_cpu(cpu_seconds: int) -> None:
    """End this process (SIGXCPU) after `cpu_seconds` of CPU time, never
    loosening a stricter limit it inherited."""
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    limits = (value for value in (cpu_seconds, soft, hard) if value != resource.RLIM_INFINITY)
    resource.setrlimit(resource.RLIMIT_CPU, (min(limits), hard))


def _orphan_cpu_seconds(timeout: float) -> int:
    """The reader's CPU limit: its wall-clock budget plus a margin. Single-
    threaded, its CPU time never exceeds its wall time, so a reader with a
    live parent is always stopped by the deadline first."""
    return math.ceil(timeout) + 5


def _stop(child) -> int | None:
    """Reap the reader process: one still running is terminated, then killed.
    Returns its exit status (negative: the signal that ended it)."""
    if child.is_alive():
        child.terminate()
        child.join(1.0)
        if child.is_alive():
            child.kill()
    child.join()
    exitcode = child.exitcode
    child.close()
    return exitcode


def _exit_status(exitcode: int | None) -> str:
    if exitcode is not None and exitcode < 0:
        try:
            return f"killed by signal {signal.Signals(-exitcode).name}"
        except ValueError:
            return f"killed by signal {-exitcode}"
    return f"exit code {exitcode}"


# --- decoding ----------------------------------------------------------------

# Where browsers look for a <meta> charset declaration (WHATWG's prescan reads
# 1024 bytes; real heads with inline scripts often run longer).
_CHARSET_SCAN_BYTES = 4096
_CONTENT_CHARSET = re.compile(r"charset\s*=\s*[\"']?\s*([A-Za-z0-9._:-]+)", re.IGNORECASE)


def _decode(raw: bytes, url: str) -> str:
    """A byte-order mark (UTF-16 or UTF-8) decides first, over any <meta>
    charset; then UTF-8 whenever the bytes are valid UTF-8 (whatever the page
    claims); then the charset the page declares, decoded the way a browser
    does; otherwise a loud failure — a guessed encoding would silently corrupt
    text that is later quoted as evidence."""
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return _decode_as(raw, "utf-16", "UTF-16 byte-order mark", url)
    if raw.startswith(codecs.BOM_UTF8):
        return _decode_as(raw[3:], "utf-8", "UTF-8 byte-order mark", url)
    try:
        return raw.decode("utf-8")
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
        info = codecs.lookup(label)
    except (LookupError, ValueError) as exc:  # ValueError: a label holding a NUL byte
        raise _unknown_charset(url, label) from exc
    # A byte-to-byte codec (browser_codec) and a text codec that refuses any
    # byte ("undefined", "idna") are both charsets no browser knows, and neither
    # failure names the page on its own.
    codec = browser_codec(info)
    if codec is None:
        raise _unknown_charset(url, label)
    if codec.startswith("utf-16"):
        codec = "utf-8"  # WHATWG: a declaration readable as ASCII cannot mean UTF-16
    try:
        return _decode_as(raw, codec, f"declared charset {label!r}", url)
    except UnicodeError as exc:
        raise _unknown_charset(url, label) from exc


def browser_codec(info: codecs.CodecInfo) -> str | None:
    """The codec a browser decodes a looked-up charset label with, or None when
    the label names no text encoding: codecs.lookup also resolves Python's
    byte-to-byte codecs (bz2, base64, hex ...), which bytes.decode refuses with
    a LookupError. The one place both the page's own declaration (_decode) and
    the HTTP Content-Type charset (jobs._page_text) are read the browser's way."""
    if not info._is_text_encoding:
        return None
    if info.name in ("iso8859-1", "ascii"):
        return "cp1252"  # WHATWG: browsers decode these labels as windows-1252
    return info.name


def _unknown_charset(url: str, label: str) -> ValueError:
    """The label is the PAGE's own text, and this failure becomes the stored run
    error the operator reads — so it is bounded by the same rule the fetch's
    refusals use. Its only other bound is the 4096-byte <meta> prescan."""
    return ValueError(f"{url} declares an unknown charset {shown_text(label)!r}")


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
# The tags _prepare_tree renames an element to, so that one C-level pass can
# unwrap or remove them all at the end. Custom names no real page uses; a page
# that did would only get that element unwrapped or removed.
_UNWRAP = "authorai-unwrap"
_DROP = "authorai-drop"
_SCRIPT_CHARACTERS = frozenset("0123456789+-−=()")
_ASCII_MINUS = str.maketrans({"−": "-"})
_HEADING_PREFIX = re.compile(r"^#{1,6} ?")
# A numeric reference to a code point XML forbids (&#12;, Word's &#11;) parses
# into that character, which lxml refuses to set as text: the tree edits raised
# over one, and trafilatura discarded the page as thin. Raw control bytes need
# nothing — the parser drops them. Decimal values are matched digit by digit,
# never converted, so a reference with thousands of digits costs nothing extra.
_XML_INVALID_REFERENCE = re.compile(
    r"&#(?:0*(?:[1-8]|1[124-9]|2\d|3[01]|6553[45])(?!\d)"
    r"|[xX]0*(?:[1-8bBcCeEfF]|1[\da-fA-F]|[fF]{3}[eEfF])(?![\da-fA-F]));?"
)


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
    # The body's reading only: the stdlib parser that reads the metadata drops
    # these references or reads them as whitespace, and never refuses one.
    tree = trafilatura.load_html(_XML_INVALID_REFERENCE.sub(" ", text))
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
    - a <sup>/<sub> holding only digits and signs is written in plain text, a
      superscript as ^6 (10<sup>6</sup> must not read "106") and a subscript
      inline (CO2), so quotes, claims and keyword search spell it the same way
      (the verdict quote check does not fold Unicode sub/superscripts); a <sup>
      holding a link
      is a footnote or reference marker and is removed; any other
      <sup>/<sub> is unwrapped to plain text;
    - a table nested in a table becomes its text inside the outer cell —
      cells joined by ", ", rows by "; " — innermost first.

    Each edit only renames the element to a sentinel tag (a rewritten script
    or table keeps its new text inside it, so an enclosing script still reads
    it); lxml's C-level strip_elements/strip_tags then apply them all in one
    pass, and the parents they spliced text into get their adjacent text nodes
    joined. Editing element by element was quadratic in the inline elements
    under one parent, and a 10 MB page of them held the worker for hours:
    libxml2 merges an XPath union ("//b | //i") by scanning one branch for
    every node of the other, and lxml.html's drop_tag/drop_tree re-copy a
    parent's growing text for every child they remove.
    """
    root = tree.getroottree()
    for element in list(root.iter("b", "strong", "i", "em", "u")):
        # Yoast FAQ questions are <strong> elements trafilatura turns into headings.
        if element.tag != "strong" or "schema-faq-question" not in (element.get("class") or ""):
            element.tag = _UNWRAP
    for element in reversed(list(root.iter("sup", "sub"))):
        content = element.text_content().strip()
        if element.tag == "sup" and element.find(".//a") is not None:
            element.clear(keep_tail=True)
            element.tag = _DROP
        elif content and set(content) <= _SCRIPT_CHARACTERS:
            plain = content.translate(_ASCII_MINUS)
            _replace_with_text(element, f"^{plain}" if element.tag == "sup" else plain)
        else:
            element.tag = _UNWRAP
    # Reverse document order visits a nested table before the table holding it.
    for table in reversed(tree.xpath("//table[ancestor::table]")):
        _replace_with_text(table, f" {_table_text(table)} ")
    runs = {}
    for element in root.iter(_UNWRAP, _DROP):
        parent = element.getparent()
        # A parent that is itself unwrapped is gone: its nearest kept ancestor reads it.
        if parent.tag not in (_UNWRAP, _DROP) and parent not in runs:
            runs[parent] = _joined_runs(parent)
    etree.strip_elements(root, _DROP, with_tail=False)
    etree.strip_tags(root, _UNWRAP)
    for parent, (text, tails) in runs.items():
        parent.text = text
        for child, tail in tails:
            child.tail = tail


def _replace_with_text(element, text: str) -> None:
    """Rewrite an element as plain text; _prepare_tree's final strip applies it."""
    element.clear(keep_tail=True)
    element.text = text
    element.tag = _UNWRAP


def _joined_runs(parent) -> tuple[str | None, list[tuple[object, str | None]]]:
    """The text a parent will hold once the strips have spliced its sentinel
    descendants away: its own text, and the tail of each element that stays
    under it, each joined into one string.

    strip_tags/strip_elements splice text nodes without joining them, and
    libxml2's XPath, which trafilatura runs over the tree, is quadratic over a
    run of adjacent text nodes; setting .text or .tail replaces a run with one
    node. The runs are read HERE, before the strips, while every piece is still
    a single node: lxml's getter reads a spliced run by concatenation, one node
    at a time, which is quadratic again. The walk mirrors the strips — an
    unwrapped element leaves its text, its children and its tail; a dropped
    one only its tail — with an explicit stack, as nesting is page-controlled."""
    runs = [[parent.text or ""]]
    kept = []
    stack = [(None, iter(parent))]
    while stack:
        owner, children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            if owner is not None:
                runs[-1].append(owner.tail or "")
        elif child.tag == _UNWRAP:
            runs[-1].append(child.text or "")
            stack.append((child, iter(child)))
        elif child.tag == _DROP:
            runs[-1].append(child.tail or "")
        else:
            kept.append(child)
            runs.append([child.tail or ""])
    tails = [(child, "".join(run) or None) for child, run in zip(kept, runs[1:], strict=True)]
    return "".join(runs[0]) or None, tails


def _table_text(table) -> str:
    rows = []
    caption = next(_children(table, ("caption",)), None)
    if caption is not None:
        rows.append(" ".join(caption.text_content().split()))
    for row in table.iter("tr"):
        cells = (" ".join(cell.text_content().split()) for cell in _children(row, ("td", "th")))
        rows.append(", ".join(cell for cell in cells if cell))
    return "; ".join(row for row in rows if row)


def _children(parent, tags: tuple[str, ...]):
    """The children with these tags, looking through the emphasis and script
    wrappers _prepare_tree has only renamed so far: the parser keeps a
    malformed <tr><em><td> as written, and the strip that frees the cell runs
    after the nested table is flattened."""
    for child in parent:
        if child.tag in tags:
            yield child
        elif child.tag == _UNWRAP:
            yield from _children(child, tags)


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
        title = _cut(" ".join(_HEADING_PREFIX.sub("", _markdown(child)).split())) or ""
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
# WebPage and schema.org's page-shaped subtypes (CMS detail pages often say ItemPage).
_PAGE_TYPES = frozenset(
    {
        "WebPage",
        "AboutPage",
        "CollectionPage",
        "ContactPage",
        "FAQPage",
        "ItemPage",
        "MedicalWebPage",
        "ProfilePage",
        "QAPage",
        "SearchResultsPage",
    }
)
# schema.org Organization and its subtypes whose names do not end in "Organization".
_ORGANIZATION_TYPES = frozenset({"Organization", "NGO", "Corporation", "Consortium"})
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
        if tag == "body":
            self._in_head = False
        elif tag in _FOREIGN_CONTENT:
            self._foreign_depth += 1
        elif tag == "meta" and self._in_head:
            attributes = dict(attrs)
            content = attributes.get("content")
            keys = (attributes.get("name"), attributes.get("property"))
            for key in dict.fromkeys(key.strip().lower() for key in keys if key):
                if content is not None:
                    self.meta.append((key, content))
        elif tag == "script":
            kind = (dict(attrs).get("type") or "").split(";")[0].strip().lower()
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
    authors     citation_author (all) > JSON-LD personal author names > meta author
                (one, unless it names an organization the page declares;
                og:site_name alone declares none)
    publisher   citation_publisher > JSON-LD publisher name > og:site_name
                > JSON-LD Organization author name
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
    work = _primary_work(nodes, url, by_id) or {}

    title = (
        first("citation_title") or _jsonld_title(work) or first("og:title") or _clean(parser.title)
    )
    # The untyped <meta name=author> carries no Person/Organization type, so only
    # a name the page itself declares for an organization excludes it: a JSON-LD
    # author typed as an organization, a JSON-LD publisher not typed Person, or
    # citation_publisher. og:site_name has no type either and is no such
    # declaration, so a personal site named after its author keeps that author.
    # The accepted cost (code cannot tell the two names apart): an institution
    # whose meta author matches only its og:site_name, with no other
    # organization signal, counts as a personal author too.
    organization_authors = _names(work.get("author"), by_id, kind="organization")
    organizations = {
        name.casefold()
        for name in (
            *organization_authors,
            *_names(work.get("publisher"), by_id, kind="not_person"),
            *meta.get("citation_publisher", []),
        )
    }
    meta_authors = [n for n in meta.get("author", [])[:1] if n.casefold() not in organizations]
    authors = (
        meta.get("citation_author")
        or _names(work.get("author"), by_id, kind="person")
        or meta_authors
    )
    publisher = (
        first("citation_publisher")
        or next(iter(_names(work.get("publisher"), by_id)), None)
        or first("og:site_name")
        # An institution credited as the author published the page when nothing
        # else says so. It is never a PERSONAL author (SourceMetadata.authors).
        or next(iter(organization_authors), None)
    )
    published = (
        first("citation_publication_date", "citation_date")
        or _jsonld_date(work)
        or first("article:published_time")
    )
    declared_doi = first("citation_doi") or next(iter(_doi_candidates(work)), None)
    return _bounded(
        PageMetadata(
            title=title,
            authors=_dedupe(authors),
            publisher=publisher,
            publication_date=_normalize_date(published) if published else None,
            doi=clean_doi(declared_doi) if declared_doi else None,
            scholarly=any(key.startswith("citation_") for key, _ in parser.meta),
        )
    )


def _bounded(page: PageMetadata) -> PageMetadata:
    """What the page declares about itself, bounded — the ONE place a declared
    value is cut. A page writes its own metadata and chooses how long it is,
    and every field here is stored with the run and shown in the UI; real
    values are orders of magnitude below these bounds.

    The DOI is an identifier, not free text: one past the bound is dropped, not
    cut, because a cut DOI names a DIFFERENT work (and would be looked up)."""
    return replace(
        page,
        title=_cut(page.title),
        authors=[_cut(name) or "" for name in page.authors[:MAX_AUTHORS]],
        publisher=_cut(page.publisher),
        publication_date=_cut(page.publication_date),
        doi=page.doi if page.doi is None or len(page.doi) <= MAX_METADATA_CHARS else None,
    )


def _cut(value: str | None) -> str | None:
    return value[:MAX_METADATA_CHARS] if value else value


# strict=False: CMSes emit raw newlines inside JSON-LD strings.
_JSONLD_DECODER = json.JSONDecoder(strict=False)


def _load_jsonld(block: str) -> object:
    """One JSON value, tolerating what templates leave after it: whitespace and
    semicolons (a real WHO fact sheet ends the Article block holding its only
    publisher and date with a stray ";"). Anything else after the value is still
    malformed."""
    text = block.strip()
    data, end = _JSONLD_DECODER.raw_decode(text)
    if text[end:].strip(" \t\r\n;"):
        raise ValueError(f"unexpected content after the JSON value (character {end})")
    return data


def _jsonld_nodes(blocks: list[str], url: str) -> list[dict]:
    """Every object in the page's JSON-LD — a block may hold one object, a
    list, or an {"@graph": [...]} wrapper. A malformed block is the page's
    defect, not ours: it is skipped with a warning naming the page."""
    nodes: list[dict] = []
    for number, block in enumerate(blocks, start=1):
        if not block.strip():
            continue
        try:
            data = _load_jsonld(block)
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


def _primary_work(nodes: list[dict], url: str, by_id: dict[str, dict]) -> dict | None:
    """The node describing this page: among article-type nodes, the one whose
    url, @id or mainEntityOfPage names this URL, or the only one there is.
    Several article nodes with none — or more than one — naming the page
    decide nothing: document order is no evidence of which is the page's own
    (a news story's JSON-LD may list the study it reports on first, and its
    DOI would then be verified as the page's). Unless those nodes declare the
    same work — the same title, authors, publisher, date and DOI, as when a
    CMS and a theme plugin each emit the page's Article — which is one work,
    not an ambiguity. Only without any article node, the same choice among
    page-type nodes — a WebPage node describes the container ("Title - Site
    Name"), not the work."""
    page = _comparable_url(url)
    for kinds in (_ARTICLE_TYPES, _PAGE_TYPES):
        candidates = [node for node in nodes if _types(node) & kinds]
        if not candidates:
            continue
        pool = [node for node in candidates if page in _named_urls(node)] or candidates
        work = _work_signature(pool[0], by_id)
        if all(_work_signature(node, by_id) == work for node in pool[1:]):
            return pool[0]
        return None
    return None


def _work_signature(node: dict, by_id: dict[str, dict]) -> tuple:
    """What _page_metadata takes from a node, as it would read it."""
    published = _jsonld_date(node)
    return (
        _jsonld_title(node),
        tuple(_names(node.get("author"), by_id)),
        tuple(_names(node.get("publisher"), by_id)),
        _normalize_date(published) if published else None,
        tuple(_doi_candidates(node)),
    )


# A node's title and date as both _page_metadata and _work_signature read them.
def _jsonld_title(node: dict) -> str | None:
    headline = _clean_jsonld(_first_str(node.get("headline")))
    return headline or _clean_jsonld(_first_str(node.get("name")))


def _jsonld_date(node: dict) -> str | None:
    return _clean_jsonld(_first_str(node.get("datePublished")))


def _named_urls(node: dict) -> set[str]:
    entity = node.get("mainEntityOfPage")
    values = [node.get("url"), node.get("@id"), entity]
    if isinstance(entity, dict):
        values += [entity.get("@id"), entity.get("url")]
    return {_comparable_url(value) for value in values if isinstance(value, str)}


# Query parameters that record how a reader arrived, not which page; any other
# query is part of the address (?id=2 is not the page ?id=1 describes).
_TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid")
# What a path keeps as written. "%" is kept too: an existing escape is never
# decoded, so an escaped "%2F" stays distinct from the separator "/".
_PATH_SAFE = "/%:@!$&'()*+,;=-._~"


def _comparable_url(value: str) -> str:
    """The address without what a pasted link adds to the canonical one: the
    scheme, a leading www., host case, a fragment, a trailing slash and
    tracking parameters — spelled as the fetched URL is: an international host
    in punycode (encoded as httpx encodes the link), the path's non-ASCII
    characters and its escapes as upper-case %XX. A CMS writes its own url
    decoded, or with lower-case escapes."""
    try:
        parts = urlsplit(value.strip())
        host = (parts.hostname or "").removeprefix("www.")
        if not host.isascii():
            host = httpx.URL(scheme="https", host=host).raw_host.decode("ascii")
        path = upper_escapes(quote(parts.path, safe=_PATH_SAFE))
        query = urlencode(
            [
                (key, item)
                for key, item in parse_qsl(parts.query, keep_blank_values=True)
                if not key.lower().startswith(_TRACKING_PARAMS)
            ]
        )
    # Not a URL at all, a host IDNA refuses, or text UTF-8 cannot encode (a
    # lone surrogate): no page's address, compared as written.
    except (ValueError, httpx.InvalidURL):
        return value.strip()
    return host + path.rstrip("/") + (f"?{query}" if query else "")


def _types(node: dict) -> set[str]:
    declared = node.get("@type")
    names = declared if isinstance(declared, list) else [declared]
    # "schema:Article" and "https://schema.org/Article" name the same type.
    return {n.rsplit("/", 1)[-1].rsplit(":", 1)[-1] for n in names if isinstance(n, str)}


def _is_organization(node: dict) -> bool:
    return any(t in _ORGANIZATION_TYPES or t.endswith("Organization") for t in _types(node))


def _names(value: object, by_id: dict[str, dict], *, kind: str | None = None) -> list[str]:
    """Names from a string, an object (a Person or an Organization), or a list
    of either. An object that is only an {"@id": ...} reference resolves
    against the page's graph. kind="person" skips Organization-typed objects (a
    plain string counts as a person); kind="organization" keeps only them;
    kind="not_person" skips Person-typed objects, a Person type winning over an
    Organization type given with it (a plain string or untyped object stays)."""
    names: list[str] = []
    for item in value if isinstance(value, list) else [value]:
        name = item
        is_organization = is_person = False
        if isinstance(item, dict):
            node = item
            if "name" not in node and isinstance(node.get("@id"), str):
                node = by_id.get(node["@id"], node)
            is_organization = _is_organization(node)
            is_person = "Person" in _types(node)
            name = _first_str(node.get("name"))
        if (
            (kind == "person" and is_organization)
            or (kind == "organization" and not is_organization)
            or (kind == "not_person" and is_person)
        ):
            continue
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
                candidates.append(_jsonld_text(item.strip()))
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


# JSON can spell half a surrogate pair ("\ud83d": JavaScript that cuts a string
# mid-emoji emits one) and json decodes it to a lone surrogate, which no UTF-8
# writer (the stored page) can encode. json has already joined every real pair.
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def _jsonld_text(value: str) -> str:
    """A JSON-LD string entity-decoded — <script> content is not decoded by
    HTML parsers, and CMSes entity-encode it ("Tom&aacute;s") — with any lone
    surrogate replaced."""
    return _LONE_SURROGATE.sub("�", unescape(value))


def _clean_jsonld(value: object) -> str | None:
    return _clean(_jsonld_text(value)) if isinstance(value, str) else None


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
