"""Bibliography scan: the works a report cites, read from its own closing pages.

A pre-upload aid. When the user picks the report PDF, the frontend asks for the
report's reference list so it can show which cited works are missing from the
user's sources and offer the free copies as links. Nothing here persists: no
run, no upload row, no file — the spooled upload is read in place.

Two judgments, split the way the rest of the pipeline splits them: the model
turns printed reference entries into fields (the language judgment), and code
decides where the bibliography is, caps what the model sees and returns, and
checks every address before it is offered. The prompt is the mirror image of
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
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import BinaryIO, Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader

from authorai.credibility import clean_doi, get_json_with_retries
from authorai.fetch import is_public_address, url_host, validate_source_url
from authorai.llm import LLM
from authorai.log import setup_logger

logger = setup_logger(__name__)

# Bounds are code, not schema: structured outputs do not honour max_length
# and a client-side validation failure would 500 the scan (see
# extract_references). 30,000 characters is 4–7x the measured real
# bibliographies (~7.5k Haiku tokens); 80 entries is more than any report in
# the example sets prints; 600 pages from the end covers any whole report
# the pipeline accepts while bounding a hostile file's page walk.
REFERENCE_MAX_CHARS = 30_000
MAX_REFERENCES = 80
REFERENCE_MAX_PAGES = 600

TextSource = Literal["heading", "tail", "none"]

# A line that IS a reference-list heading — optionally numbered ("7.
# References"), optionally colon-terminated, nothing else on the line. A
# mid-line mention ("see References") never matches. Extend the alternation
# only with evidence: a miss falls back to the document tail, which still works.
_HEADING = re.compile(
    r"^\s*(?:\d+[.\s]*)?(?:references|bibliography|works cited|reference list|literature cited)"
    r"\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def read_pages(handle: BinaryIO, *, max_pages: int = REFERENCE_MAX_PAGES) -> list[str]:
    """The text of the LAST `max_pages` pages, one string per page (empty for
    a page without text, such as a scan).

    pypdf, not Docling: a layout pass is seconds per page, far too slow for a
    dialog that must answer while the user is still picking sources, and the
    reference list needs no layout. `strict=False` tolerates the malformed
    xref tables real PDFs carry; a file encrypted with an empty user password
    (print/copy restrictions) is opened as readers do. Anything pypdf cannot
    read — its own exception family, or the KeyError/RecursionError a hostile
    file provokes — is one ValueError, the caller's 400.
    """
    try:
        reader = PdfReader(handle, strict=False)
        if reader.is_encrypted:
            reader.decrypt("")
        pages = reader.pages
        first = max(0, len(pages) - max_pages)
        return [pages[index].extract_text() or "" for index in range(first, len(pages))]
    except Exception as exc:  # noqa: BLE001 - every reader failure is "unreadable"
        raise ValueError(f"could not read the PDF ({type(exc).__name__}: {exc})") from exc


def reference_text(
    pages: list[str], *, max_chars: int = REFERENCE_MAX_CHARS
) -> tuple[str, TextSource]:
    """The closing text the model should read, and where it came from.

    From the LAST reference-list heading forward when the report prints one
    (the 51-page Drought report drops from 72,699 to 4,443 characters), else
    the last `max_chars` of the document (two of four test reports print no
    heading). No text at all is ("", "none"): a scanned PDF must not yield an
    invented bibliography, so the caller makes no model call on that value.
    """
    text = "\n".join(pages)
    if not text.strip():
        return "", "none"
    last = None
    for match in _HEADING.finditer(text):
        last = match
    if last is not None:
        return text[last.start() : last.start() + max_chars].strip(), "heading"
    return text[-max_chars:].strip(), "tail"


class Reference(BaseModel):
    """One printed reference entry, as fields. `entry` is the anchor: the
    text as printed, shown to the user when the title is null and the proof
    that the row came from the page rather than from memory."""

    title: str | None = Field(default=None, description="The cited work's title, as printed")
    authors: list[str] = Field(
        default_factory=list, description="Author names as printed; empty when none are printed"
    )
    year: int | None = Field(default=None, description="Publication year, as printed")
    doi: str | None = Field(
        default=None, description="The DOI printed in the entry (10.xxxx/...), without a URL prefix"
    )
    url: str | None = Field(default=None, description="A web address printed in the entry")
    entry: str = Field(description="The reference entry verbatim, as printed")

    @field_validator("title", "doi", "url", "entry", mode="after")
    @classmethod
    def _blank_is_absent(cls, value: str | None) -> str | None:
        # The model sometimes prints '' where it was told to leave null; the
        # DOI lookup and the printed-URL rule key on absence.
        if value is None:
            return None
        return value.strip() or None

    @field_validator("authors", mode="after")
    @classmethod
    def _drop_blank_authors(cls, value: list[str]) -> list[str]:
        return [name.strip() for name in value if name.strip()]


class ReferenceList(BaseModel):
    references: list[Reference] = Field(default_factory=list)


REFERENCES_SYSTEM = """\
You read the closing pages of a report and list the works its reference list
cites. Every entry you return describes a work the report CITES — never the
report itself. Report ONLY what the text actually prints — never guess, never
complete an entry from world knowledge. A field the entry does not print is
null.

- `entry`: the reference as printed, verbatim (line breaks joined), so a
  reader can find it on the page.
- `title`: the cited work's title as printed. Null when the entry prints none.
- `authors`: the names as printed, personal or organizational. Empty list if
  none are printed.
- `year`: the publication year as printed.
- `doi`: ONLY a DOI printed in the entry (10.xxxx/...), without a URL prefix.
  Never supply a DOI from memory, even for a famous paper.
- `url`: ONLY a web address printed in the entry.

In-text citations such as "(Smith, 2020)", footnote markers, figure and table
sources, and the report's own title and imprint are not entries. A document
without a reference list yields an empty list — that is the correct answer,
not a failure.
"""


def extract_references(llm: LLM, model: str, closing_text: str) -> ReferenceList:
    """The model's reading of the closing text, capped in code at
    MAX_REFERENCES — the cap is deliberately NOT in the schema, where
    structured outputs may not honour it and a validation failure would fail
    the whole scan instead of trimming it."""
    result = llm.parse(
        model=model,
        system=REFERENCES_SYSTEM,
        prompt=f"CLOSING PAGES:\n\n{closing_text}",
        output_type=ReferenceList,
    )
    if len(result.references) > MAX_REFERENCES:
        logger.warning(
            "the model returned %d references — keeping the first %d",
            len(result.references),
            MAX_REFERENCES,
        )
        result = ReferenceList(references=result.references[:MAX_REFERENCES])
    return result


# --- Unpaywall ------------------------------------------------------------------

UNPAYWALL_BASE = "https://api.unpaywall.org"
UNPAYWALL_TIMEOUT = 10.0
UNPAYWALL_RETRIES = 2
# httpx.Client is thread-safe; four lookups at a time keep a long list to
# seconds instead of a serial ten-per-second crawl, without leaning on the
# registry (Unpaywall asks for polite use, not a rate).
LOOKUP_WORKERS = 4

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
        self._client = httpx.Client(
            base_url=UNPAYWALL_BASE,
            timeout=timeout,
            headers={"User-Agent": f"AuthorAI/2.0 (mailto:{self._mailto})"},
        )

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
    app would refuse the link: the syntax gate a pasted link passes, then the
    one refusal the fetcher would make that needs no network. A LITERAL
    address host must pass fetch.is_public_address — the gate the fetcher
    applies to every resolved hop — and `localhost`, or any name under it,
    is the local machine by definition. A NAME that resolves to private
    space is deliberately not caught here: a pre-upload aid does no DNS,
    and the fetch gate resolves and pins every hop at ingest, where such a
    link is refused. An offer is only ever as good as that gate."""
    normalized = validate_source_url(url)
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
    offer needs an address.
    """
    if record is None:
        return NOT_RESOLVED
    is_oa = record.get("is_oa")
    if is_oa is False:
        return ("paywalled", None)
    if is_oa is not True:
        return NOT_RESOLVED
    best = record.get("best_oa_location") or {}
    candidates: list[tuple[Retrievability, str | None]] = [
        ("pdf", best.get("url_for_pdf")),
        ("landing", best.get("url_for_landing_page")),
        ("landing", best.get("url")),
    ]
    for kind, url in candidates:
        if not url:
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
    """Retrievability for every reference, in order — a lookup for each
    printed DOI, LOOKUP_WORKERS at a time, the rest NOT_RESOLVED.

    Capped at MAX_REFERENCES (extract_references already trims; this is the
    second wall). The first registry failure ends the whole lookup: pending
    lookups are cancelled, in-flight ones finish (no request is abandoned
    mid-way, and nothing runs on after the caller has the answer), and
    RegistryUnavailable carries every verdict that was reached.
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

    pool = ThreadPoolExecutor(max_workers=LOOKUP_WORKERS)
    futures: dict[Future, int] = {
        pool.submit(lookup, reference.doi): index
        for index, reference in enumerate(references[:MAX_REFERENCES])
        if reference.doi
    }
    try:
        for future in as_completed(futures):
            try:
                record = future.result()
            except _Skipped:
                continue
            results[futures[future]] = retrievability(record)
    except RuntimeError as exc:
        pool.shutdown(wait=True, cancel_futures=True)
        for future, index in futures.items():
            if future.done() and not future.cancelled() and future.exception() is None:
                results[index] = retrievability(future.result())
        raise RegistryUnavailable(str(exc), results) from exc
    finally:
        pool.shutdown(wait=True)
    return results
