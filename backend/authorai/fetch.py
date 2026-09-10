"""Source-URL fetching: the server-side request forgery boundary.

A source can be a web page, so the backend fetches URLs that users supply. On
a public repo that is an SSRF surface: a URL, a redirect, or a DNS answer
could otherwise aim the server at loopback, the cloud metadata endpoint, or
the rest of its private network. The defenses, in the order a fetch meets them:

1. `validate_source_url` (syntax only): http/https, a real host, no
   credentials. Applied to the input AND to every redirect target.
2. DNS, then `is_public_address` on EVERY answer, with IPv4 embedded in IPv6
   unwrapped and checked too. One non-public answer blocks the host: which
   address a client would connect to is not ours to predict.
3. IP pinning: the request is sent to the first vetted address as an IP
   literal, with the hostname only in the Host header and the TLS SNI — so
   certificate verification still runs against the hostname, and the HTTP
   client never performs a second, rebindable DNS lookup of its own.
4. Manual redirects: each hop repeats 1-3 against the LOGICAL (hostname) URL.
5. Content-type and declared-size gates BEFORE the body is read, then a cap
   on decoded bytes while it streams (a gzip bomb is measured after inflation).

Time: `fetch_timeout_seconds` is one budget for the whole fetch. It is checked
between hops and after every body chunk, and each request's socket timeouts
are capped at what remains. Two phases cannot be interrupted part-way: the
system DNS lookup (bounded by the OS resolver), and the response-header read —
a server trickling header bytes faster than the read timeout can hold a fetch
past the budget.
"""

import ipaddress
import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from authorai.config import Settings
from authorai.log import setup_logger

logger = setup_logger(__name__)

Resolve = Callable[[str, int], list[str]]
IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

MAX_URL_LENGTH = 2048
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
PDF_TYPES = frozenset({"application/pdf", "application/octet-stream"})
PDF_MAGIC = b"%PDF-"
ACCEPT = "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.1"
ACCEPT_ENCODING = "gzip, deflate"
# The codings httpx decodes with the stdlib alone. br/zstd need optional
# packages, and httpx passes a coding it cannot decode through UNTOUCHED —
# still-compressed bytes would reach extraction as if they were the page.
DECODABLE_ENCODINGS = frozenset({"identity", "gzip", "deflate"})

_YOUTUBE_HOSTS = frozenset(
    f"{prefix}{domain}"
    for prefix in ("", "www.", "m.", "music.")
    for domain in ("youtube.com", "youtu.be", "youtube-nocookie.com")
)
# Applied to httpx's raw_host, which is already IDNA-encoded to ASCII.
_HOSTNAME = re.compile(r"[A-Za-z0-9._-]+")
_URL_CREDENTIALS = re.compile(r"^([^:/?#]*:)?//[^/?#]*@")
_SHOWN_URL_CHARS = 200

_NAT64_PREFIXES = (
    ipaddress.ip_network("64:ff9b::/96"),  # well-known prefix (RFC 6052)
    ipaddress.ip_network("64:ff9b:1::/48"),  # local-use prefix (RFC 8215)
)
_NEVER_PUBLIC = (
    # Carrier-grade NAT shared space: stdlib calls it neither private nor global.
    ipaddress.ip_network("100.64.0.0/10"),
    # Local-use NAT64 only reaches the operator's own network. Listed so the
    # gate does not depend on the stdlib patch level's copy of the IANA tables.
    ipaddress.ip_network("64:ff9b:1::/48"),
    # Deprecated IPv4-compatible addresses (::a.b.c.d): stdlib calls
    # ::127.0.0.1 global, and no legitimate DNS answer lives here.
    ipaddress.ip_network("::/96"),
)


class FetchError(RuntimeError):
    """A source URL could not be fetched. The message always names the URL."""


class BlockedAddressError(FetchError):
    """The URL's host resolves to a private or reserved network address."""


@dataclass(frozen=True)
class FetchedResponse:
    url: str  # the normalized URL that was requested
    final_url: str  # the logical URL after redirects (hostname form, never the pinned IP)
    content_type: str  # media type only, lowercased, no parameters, e.g. "text/html"
    charset: str | None  # from the Content-Type parameters, lowercased, if any
    body: bytes  # fully read, decoded of any Content-Encoding
    is_pdf: bool


def _shown(url: str) -> str:
    """A URL as quoted in an error message: credentials cut, length bounded."""
    text = _URL_CREDENTIALS.sub(r"\1//", url)
    if len(text) > _SHOWN_URL_CHARS:
        text = text[:_SHOWN_URL_CHARS] + "..."
    return repr(text)


def validate_source_url(url: str) -> str:
    """Check a source URL's syntax (no DNS) and return its normalized form.

    Raises ValueError with a short user-facing message quoting the URL
    (credentials removed). The fragment is dropped; httpx lowercases the scheme
    and host, IDNA-encodes the host, and percent-encodes the path.
    """
    candidate = url.strip()
    if not candidate:
        raise ValueError("Source URL is empty")
    shown = _shown(candidate)
    if len(candidate) > MAX_URL_LENGTH:
        raise ValueError(f"Source URL {shown} is longer than {MAX_URL_LENGTH} characters")
    try:
        parsed = httpx.URL(candidate)
    except (httpx.InvalidURL, ValueError) as exc:
        raise ValueError(f"Source URL {shown} is not a valid URL ({exc})") from exc
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Source URL {shown} must start with http:// or https://")
    if parsed.userinfo:
        raise ValueError(f"Source URL {shown} must not include a username or password")
    host = parsed.raw_host.decode("ascii", errors="replace")
    if not host:
        raise ValueError(f"Source URL {shown} has no host")
    if not _is_valid_host(host):
        # httpx percent-encodes rather than rejects whitespace in a host, so
        # "exa mple.org" arrives here as "exa%20mple.org" and fails the pattern.
        raise ValueError(f"Source URL {shown} has an invalid host")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError(f"Source URL {shown} has an invalid port")
    normalized = str(parsed.copy_with(fragment=None))
    if len(normalized) > MAX_URL_LENGTH:
        raise ValueError(
            f"Source URL {shown} is longer than {MAX_URL_LENGTH} characters once encoded"
        )
    return normalized


def _is_valid_host(host: str) -> bool:
    if ":" in host:  # an IPv6 literal (httpx has already removed the brackets)
        if "%" in host:  # a zone ID names a local interface, never a remote server
            return False
        try:
            ipaddress.IPv6Address(host)
        except ValueError:
            return False
        return True
    return _HOSTNAME.fullmatch(host) is not None


def is_youtube_url(url: str) -> bool:
    """True when the URL's host is YouTube, including www./m./music. hosts."""
    try:
        host = httpx.URL(url.strip()).host
    except (httpx.InvalidURL, ValueError):
        return False
    return host.lower().rstrip(".") in _YOUTUBE_HOSTS


def is_public_address(ip: str | IPAddress) -> bool:
    """True only for a globally routable unicast address.

    IPv4 carried inside IPv6 (mapped, NAT64, 6to4, Teredo) is unwrapped and
    must pass the same gate. Raises ValueError for a string that is not an
    IP address.
    """
    address = ipaddress.ip_address(ip) if isinstance(ip, str) else ip
    if isinstance(address, ipaddress.IPv6Address):
        if address.scope_id:  # a zone makes the address interface-local
            return False
        if not all(is_public_address(inner) for inner in _embedded_ipv4(address)):
            return False
    if any(address in network for network in _NEVER_PUBLIC):
        return False
    # is_global alone is not enough: stdlib calls multicast (224.0.0.1, ff02::1) global.
    return address.is_global and not address.is_multicast


def _embedded_ipv4(address: ipaddress.IPv6Address) -> list[ipaddress.IPv4Address]:
    """Every IPv4 destination an IPv6 address can stand for."""
    embedded = []
    if address.ipv4_mapped:
        embedded.append(address.ipv4_mapped)
    if any(address in prefix for prefix in _NAT64_PREFIXES):
        # NAT64 addresses carry the IPv4 in their low 32 bits. stdlib calls
        # 64:ff9b::7f00:1 (NAT64 of loopback) global, so this is load-bearing.
        embedded.append(ipaddress.IPv4Address(int(address) & 0xFFFFFFFF))
    if address.sixtofour:
        embedded.append(address.sixtofour)
    if address.teredo:
        embedded.extend(address.teredo)  # (server, client): both must pass
    return embedded


def _default_resolve(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(info[4][0] for info in infos))


def fetch_url(
    url: str,
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    resolve: Resolve | None = None,
) -> FetchedResponse:
    """Fetch an HTML page or a PDF from a user-supplied URL, SSRF-safely.

    Raises FetchError naming the URL (BlockedAddressError for a non-public
    destination). `resolve` replaces the system resolver (tests inject
    addresses). A supplied `client` must be built with trust_env=False and no
    proxy — a proxy resolves the hostname itself, voiding both the gate and the
    pin — and is left open for its owner; a client created here is closed.
    """
    if client is not None and client.trust_env:
        raise ValueError(
            "fetch_url needs an httpx.Client built with trust_env=False: an environment "
            "proxy would resolve the host itself, bypassing the address gate and the IP pin"
        )
    try:
        requested = validate_source_url(url)
    except ValueError as exc:
        raise FetchError(f"Cannot fetch source URL: {exc}") from exc
    resolver = resolve or _default_resolve
    if client is not None:
        return _fetch(requested, settings, client, resolver)
    with httpx.Client(trust_env=False, follow_redirects=False) as own_client:
        return _fetch(requested, settings, own_client, resolver)


def _fetch(
    requested: str, settings: Settings, client: httpx.Client, resolve: Resolve
) -> FetchedResponse:
    budget = settings.fetch_timeout_seconds
    deadline = time.monotonic() + budget
    current = requested
    for _hop in range(settings.fetch_max_redirects + 1):
        where = _where(requested, current)
        _remaining(deadline, budget, where)
        pinned_url, host_header, extensions = _pin(httpx.URL(current), where, resolve)
        headers = {
            "Host": host_header,
            "User-Agent": settings.fetch_user_agent,
            "Accept": ACCEPT,
            "Accept-Encoding": ACCEPT_ENCODING,
            # A fresh connection per hop: a pooled one reused for a different
            # hostname on the same IP would carry a TLS session verified for
            # the previous name.
            "Connection": "close",
        }
        timeout = httpx.Timeout(_remaining(deadline, budget, where))
        try:
            with client.stream(
                "GET",
                pinned_url,
                headers=headers,
                extensions=extensions,
                timeout=timeout,
                # Per request, because a supplied client may follow by default —
                # an automatic redirect would skip validation, the gate, and the pin.
                follow_redirects=False,
            ) as response:
                if response.status_code in REDIRECT_STATUSES:
                    current = _redirect_target(response, current, where)
                    continue
                return _read(response, settings, deadline, where, requested, current)
        except httpx.TimeoutException as exc:
            raise FetchError(f"Fetching {where} timed out ({budget:g}-second time budget)") from exc
        except httpx.HTTPError as exc:
            raise FetchError(f"Fetching {where} failed: {type(exc).__name__}: {exc}") from exc
        except httpx.InvalidURL as exc:
            # Even with follow_redirects=False, httpx parses a redirect's
            # Location to build response.next_request, and a malformed one
            # ("javascript:alert(1)") raises InvalidURL — not an HTTPError.
            raise FetchError(
                f"Fetching {where} failed: the server sent an invalid redirect Location ({exc})"
            ) from exc
    raise FetchError(
        f"Fetching {_shown(requested)} failed: more than {settings.fetch_max_redirects} redirects"
    )


def _where(requested: str, current: str) -> str:
    if current == requested:
        return _shown(requested)
    return f"{_shown(current)} (redirected from {_shown(requested)})"


def _remaining(deadline: float, budget: float, where: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise FetchError(f"Fetching {where} exceeded its {budget:g}-second time budget")
    return remaining


def _pin(url: httpx.URL, where: str, resolve: Resolve) -> tuple[httpx.URL, str, dict[str, str]]:
    """Resolve the host, gate every answer, and aim the request at the first
    vetted IP. Returns (pinned URL, Host header, request extensions)."""
    host = url.raw_host.decode("ascii")
    port = url.port or (443 if url.scheme == "https" else 80)
    try:
        answers = resolve(host, port)
    except (OSError, UnicodeError) as exc:
        raise FetchError(
            f"Fetching {where} failed: {host!r} could not be resolved ({exc})"
        ) from exc
    if not answers:
        raise FetchError(f"Fetching {where} failed: {host!r} resolved to no addresses")
    try:
        addresses = [ipaddress.ip_address(answer) for answer in answers]
    except ValueError as exc:
        raise FetchError(
            f"Fetching {where} failed: DNS for {host!r} returned an unparseable address"
        ) from exc
    blocked = [str(address) for address in addresses if not is_public_address(address)]
    if blocked:
        # The addresses go to the server log only: echoing them to the requester
        # would let anyone map the internal network one URL at a time.
        logger.warning(
            "Blocked fetch of %s: %r resolves to non-public %s", where, host, ", ".join(blocked)
        )
        raise BlockedAddressError(
            f"Refusing to fetch {where}: {host!r} resolves to a private or reserved network address"
        )
    host_header = f"[{host}]" if ":" in host else host
    if url.port is not None:  # httpx drops a default port, so this is a non-default one
        host_header = f"{host_header}:{url.port}"
    # SNI makes TLS verify the certificate against the hostname, not the pinned IP.
    extensions = {"sni_hostname": host} if url.scheme == "https" else {}
    return url.copy_with(host=str(addresses[0])), host_header, extensions


def _redirect_target(response: httpx.Response, current: str, where: str) -> str:
    location = response.headers.get("Location")
    if not location:
        raise FetchError(
            f"Fetching {where} failed: HTTP {response.status_code} redirect "
            "without a Location header"
        )
    try:
        joined = str(httpx.URL(current).join(location))
    except (httpx.InvalidURL, ValueError) as exc:
        raise FetchError(
            f"Fetching {where} failed: redirected to an invalid URL {_shown(location)}"
        ) from exc
    try:
        return validate_source_url(joined)
    except ValueError as exc:
        raise FetchError(f"Fetching {where} failed: refused redirect. {exc}") from exc


def _read(
    response: httpx.Response,
    settings: Settings,
    deadline: float,
    where: str,
    requested: str,
    final_url: str,
) -> FetchedResponse:
    status = response.status_code
    if not 200 <= status < 300:
        raise FetchError(f"Fetching {where} failed: the server answered HTTP {status}")
    content_type, charset = _parse_content_type(response.headers.get("Content-Type", ""))
    if content_type in HTML_TYPES:
        limit = settings.fetch_max_bytes
    elif content_type in PDF_TYPES:
        limit = settings.max_upload_bytes
    else:
        declared_type = content_type or "(none)"
        raise FetchError(
            f"Fetching {where} failed: unsupported content type {declared_type!r} "
            "(a source must be an HTML page or a PDF)"
        )
    codings = [c.strip().lower() for c in response.headers.get("Content-Encoding", "").split(",")]
    undecodable = [c for c in codings if c and c not in DECODABLE_ENCODINGS]
    if undecodable:
        raise FetchError(
            f"Fetching {where} failed: unsupported Content-Encoding {', '.join(undecodable)!r}"
        )
    declared = response.headers.get("Content-Length")
    if declared is not None:
        if not declared.strip().isdigit():
            raise FetchError(f"Fetching {where} failed: malformed Content-Length {declared!r}")
        if int(declared) > limit:
            raise _too_large(where, limit)
    body = bytearray()
    # iter_bytes yields DECODED bytes, so the cap holds against compression bombs.
    for chunk in response.iter_bytes():
        if len(body) + len(chunk) > limit:
            raise _too_large(where, limit)
        body += chunk
        _remaining(deadline, settings.fetch_timeout_seconds, where)
    data = bytes(body)
    is_pdf = data.startswith(PDF_MAGIC)
    if content_type in PDF_TYPES and not is_pdf:
        raise FetchError(
            f"Fetching {where} failed: served as {content_type!r} but the body is not a PDF"
        )
    return FetchedResponse(
        url=requested,
        final_url=final_url,
        content_type=content_type,
        charset=charset,
        body=data,
        is_pdf=is_pdf,
    )


def _too_large(where: str, limit: int) -> FetchError:
    return FetchError(f"Fetching {where} failed: the response exceeds the {limit:,}-byte limit")


def _parse_content_type(header: str) -> tuple[str, str | None]:
    """'Text/HTML; Charset="UTF-8"' -> ("text/html", "utf-8")."""
    media_type, *params = header.split(";")
    charset = None
    for param in params:
        name, _, value = param.partition("=")
        if name.strip().lower() == "charset":
            charset = value.strip().strip('"').strip().lower() or None
            break
    return media_type.strip().lower(), charset
