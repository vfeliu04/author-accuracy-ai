"""Source-URL fetcher tests: URL validation, the SSRF address gate, IP pinning,
manual redirects, body caps, and the total time budget.

Offline: every test injects its resolver, and respx mocks the PINNED-IP URL —
a request sent to the hostname form (i.e. an unpinned request) is unmocked and
raises. The one integration test really fetches https://example.com/ to prove
that pinning to an IP while sending SNI still passes certificate verification.
"""

import gzip
import ipaddress
import socket
import ssl
import types

import httpx
import pytest
import respx

from authorai import fetch as fetchmod
from authorai.config import Settings
from authorai.fetch import (
    BlockedAddressError,
    FetchedResponse,
    FetchError,
    fetch_url,
    is_public_address,
    is_youtube_url,
    validate_source_url,
)

PUBLIC_V4 = "93.184.216.34"
PUBLIC_V6 = "2606:4700:4700::1111"
DEFAULT_USER_AGENT = "AuthorAccuracyAI/2.0 (+https://github.com/vfeliu04/author-accuracy-ai)"
# Real PDFs open with a version line and a binary-marker comment line.
PDF_BYTES = (
    b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    + b"0" * 20_000
    + b"\n%%EOF\n"
)


def _settings(**overrides) -> Settings:
    # Every value a test relies on is explicit, so no environment variable can
    # change what a test measures. HTML cap 10 kB, PDF cap 50 kB.
    values = {
        "fetch_timeout_seconds": 30.0,
        "fetch_max_bytes": 10_000,
        "max_upload_bytes": 50_000,
        "fetch_max_redirects": 5,
    }
    values.update(overrides)
    return Settings(**values)


class Resolver:
    """Injected DNS: fixed answers per host, recording every lookup. An
    unexpected host raises KeyError, which the fetcher must not swallow."""

    def __init__(self, answers: dict[str, list[str]]):
        self.answers = answers
        self.calls: list[tuple[str, int]] = []

    def __call__(self, host: str, port: int) -> list[str]:
        self.calls.append((host, port))
        return self.answers[host]


def _example(*addresses: str) -> Resolver:
    return Resolver({"example.org": list(addresses or [PUBLIC_V4])})


def _html(body: bytes = b"<html><body>ok</body></html>") -> httpx.Response:
    return httpx.Response(200, headers={"Content-Type": "text/html; charset=utf-8"}, content=body)


def _tracked(chunks, pulled: list):
    """A lazily-read body that records every chunk the reader pulls."""
    for chunk in chunks:
        pulled.append(len(chunk))
        yield chunk


@pytest.fixture()
def clock(monkeypatch):
    """A controllable monotonic clock for the fetcher only (no sleeping)."""
    fake = types.SimpleNamespace(now=1000.0)
    monkeypatch.setattr(fetchmod, "time", types.SimpleNamespace(monotonic=lambda: fake.now))
    return fake


def test_fetch_settings_have_the_documented_defaults_and_accept_overrides():
    settings = Settings()
    assert settings.fetch_timeout_seconds == 30.0
    assert settings.fetch_max_bytes == 10_000_000
    assert settings.fetch_max_redirects == 5
    assert settings.fetch_user_agent == DEFAULT_USER_AGENT
    # extra="ignore" would silently drop a misspelled field — prove kwargs land.
    assert _settings(fetch_max_redirects=1, fetch_max_bytes=7).fetch_max_redirects == 1
    assert _settings(fetch_max_bytes=7).fetch_max_bytes == 7


# --- validate_source_url ----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  https://example.org/a#frag  ", "https://example.org/a"),
        ("\thttps://example.org/report.pdf\r\n", "https://example.org/report.pdf"),
        ("HTTPS://EXAMPLE.ORG/Path?q=1", "https://example.org/Path?q=1"),
        ("https://example.org:443/x", "https://example.org/x"),
        ("http://example.org:80/x", "http://example.org/x"),
        ("http://example.org:8080/x", "http://example.org:8080/x"),
        ("https://bücher.de/ü", "https://xn--bcher-kva.de/%C3%BC"),
        ("http://[2606:4700:4700::1111]:8080/p", "http://[2606:4700:4700::1111]:8080/p"),
        ("http://127.0.0.1#@evil.example/", "http://127.0.0.1"),
    ],
)
def test_validate_source_url_normalizes(raw, expected):
    assert validate_source_url(raw) == expected


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("", "empty"),
        ("   \n", "empty"),
        ("ftp://x", "http:// or https://"),
        ("javascript:alert(1)", "http:// or https://"),
        ("file:///etc/passwd", "http:// or https://"),
        ("example.org/page", "http:// or https://"),
        ("//example.org/page", "http:// or https://"),
        ("https:///nohost", "no host"),
        ("http:example.org", "no host"),
        ("http://user:pass@example.org/", "username or password"),
        ("http://user@example.org/", "username or password"),
        ("http://evil.example\\@127.0.0.1/", "username or password"),
        ("http://exa mple.org/", "invalid host"),
        ("http://exa%20mple.org/", "invalid host"),
        ("http://[fe80::1%25en0]/", "invalid host"),
        ("http://example.org:99999/", "invalid port"),
        ("http://example.org:0/", "invalid port"),
        ("http://example.org:abc/", "not a valid URL"),
        ("http://[::1/", "not a valid URL"),
        ("https://example.org/a\nb", "not a valid URL"),
        ("https://example.org/" + "a" * 2029, "longer than 2048"),
    ],
)
def test_validate_source_url_rejects(raw, message):
    with pytest.raises(ValueError, match=message):
        validate_source_url(raw)


def test_validate_source_url_accepts_exactly_2048_characters():
    url = "https://example.org/" + "a" * 2028
    assert len(url) == 2048
    assert validate_source_url(url) == url


def test_rejections_name_the_url_but_never_echo_credentials():
    with pytest.raises(ValueError) as info:
        validate_source_url("http://alice:hunter2@example.org/private")
    message = str(info.value)
    assert "hunter2" not in message and "alice" not in message
    assert "example.org/private" in message
    with pytest.raises(ValueError, match="ftp://x"):
        validate_source_url("ftp://x")


def test_an_overlong_url_is_quoted_truncated_not_whole():
    with pytest.raises(ValueError) as info:
        validate_source_url("https://example.org/" + "a" * 5000)
    message = str(info.value)
    assert "https://example.org/aaa" in message
    assert len(message) < 300


# --- is_youtube_url ---------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", True),
        ("https://youtube.com/watch?v=x", True),
        ("https://m.youtube.com/watch?v=x", True),
        ("https://music.youtube.com/watch?v=x", True),
        ("https://youtu.be/dQw4w9WgXcQ", True),
        ("https://www.youtu.be/x", True),
        ("https://www.youtube-nocookie.com/embed/x", True),
        ("HTTPS://WWW.YOUTUBE.COM/watch?v=x", True),
        ("https://youtube.com./watch?v=x", True),
        ("https://youtube.com.evil.example/watch", False),
        ("https://notyoutube.com/", False),
        ("https://evil.example/youtube.com", False),
        ("https://evil.example/?u=https://youtube.com", False),
        ("https://gaming.youtube.com/", False),
        ("https://www.m.youtube.com/", False),
        ("not a url", False),
        ("", False),
    ],
)
def test_is_youtube_url(url, expected):
    assert is_youtube_url(url) is expected


# --- the address gate -------------------------------------------------------

# Teredo: server 65.54.227.120 (public), client 10.0.0.1 (private, stored inverted).
TEREDO_PRIVATE_CLIENT = "2001:0:4136:e378:8000:63bf:f5ff:fffe"

BLOCKED_ADDRESSES = [
    "127.0.0.1",
    "10.0.0.1",
    "172.16.0.1",
    "192.168.1.1",
    "169.254.169.254",
    "100.64.0.1",
    "0.0.0.0",
    "224.0.0.1",  # stdlib: is_global=True — the multicast clause is load-bearing
    "255.255.255.255",
    "::1",
    "::",
    "fc00::1",
    "fe80::1",
    "::ffff:127.0.0.1",
    "::ffff:10.0.0.1",
    "64:ff9b::7f00:1",  # stdlib: is_global=True — the NAT64 unwrap is load-bearing
    "2002:7f00:1::1",  # 6to4 of loopback
    TEREDO_PRIVATE_CLIENT,
    "2001:db8::1",
    # Beyond the required table:
    "100.127.255.254",  # top of the CGNAT /10
    "239.255.255.250",  # SSDP multicast, stdlib is_global=True
    "ff02::1",  # IPv6 multicast, stdlib is_global=True
    "64:ff9b::a9fe:a9fe",  # NAT64 of the cloud metadata address
    "64:ff9b:1::808:808",  # RFC 8215 local-use NAT64, even with a public inner
    "2002:a00:1::1",  # 6to4 of 10.0.0.1
    "::127.0.0.1",  # deprecated IPv4-compatible form, stdlib is_global=True
    "fe80::1%en0",  # zoned
    "2606:4700:4700::1111%en0",  # a zone makes even global bits interface-local
    "198.18.0.1",  # benchmarking
    "192.0.2.1",  # documentation
]
PUBLIC_ADDRESSES = [
    "93.184.216.34",
    "8.8.8.8",
    "2606:4700:4700::1111",
    "64:ff9b::808:808",  # NAT64 of a public address — DNS64 networks must keep working
    "::ffff:8.8.8.8",
]


@pytest.mark.parametrize("address", BLOCKED_ADDRESSES)
def test_non_public_addresses_are_blocked(address):
    assert is_public_address(address) is False


@pytest.mark.parametrize("address", PUBLIC_ADDRESSES)
def test_public_addresses_pass(address):
    assert is_public_address(address) is True
    assert is_public_address(ipaddress.ip_address(address)) is True


# Each is blocked by a clause of the gate's own, not by stdlib's is_global. This
# Python's stdlib already calls the CGNAT and local-use NAT64 entries non-global,
# so the table above passes whether or not the explicit clause exists.
BLOCKED_REGARDLESS_OF_STDLIB = [
    "100.64.0.1",  # 100.64.0.0/10 (carrier-grade NAT)
    "100.127.255.254",
    "64:ff9b:1::808:808",  # 64:ff9b:1::/48 (local-use NAT64), even with a public inner
    "::127.0.0.1",  # ::/96 (deprecated IPv4-compatible)
    "224.0.0.1",  # multicast
    "ff02::1",
]


@pytest.mark.parametrize("address", BLOCKED_REGARDLESS_OF_STDLIB)
def test_explicit_guards_block_even_where_stdlib_calls_the_address_global(monkeypatch, address):
    """A stdlib patch level with different IANA tables must not open the gate:
    with is_global forced True, only the gate's own clauses stand."""
    always_global = property(lambda self: True)
    monkeypatch.setattr(ipaddress.IPv4Address, "is_global", always_global)
    monkeypatch.setattr(ipaddress.IPv6Address, "is_global", always_global)
    # Control: the patch really took, so a plain private address now passes.
    assert is_public_address("10.0.0.1") is True
    assert is_public_address(address) is False


def test_address_gate_raises_on_a_non_address():
    with pytest.raises(ValueError):
        is_public_address("example.org")


@pytest.mark.parametrize(
    ("address", "embedded"),
    [
        ("::ffff:10.0.0.1", ["10.0.0.1"]),
        ("64:ff9b::7f00:1", ["127.0.0.1"]),
        ("64:ff9b:1::a00:1", ["10.0.0.1"]),
        ("2002:7f00:1::1", ["127.0.0.1"]),
        (TEREDO_PRIVATE_CLIENT, ["65.54.227.120", "10.0.0.1"]),
        (PUBLIC_V6, []),
    ],
)
def test_ipv4_is_unwrapped_from_every_ipv6_carrier(address, embedded):
    """Tested directly because the gate table cannot see it here: this
    Python's stdlib already calls mapped, 6to4 and Teredo space non-global,
    while older 3.11 patch levels called 6to4 global — there the unwrap is
    the only thing standing between 2002:7f00:1::1 and loopback."""
    found = fetchmod._embedded_ipv4(ipaddress.IPv6Address(address))
    assert [str(ip) for ip in found] == embedded


# --- fetch_url: happy paths and pinning -------------------------------------


@respx.mock
def test_html_fetch_is_pinned_to_the_resolved_ip_with_host_and_sni():
    route = respx.get(f"https://{PUBLIC_V4}/page?id=7").mock(
        return_value=httpx.Response(
            200, headers={"Content-Type": "text/html; charset=UTF-8"}, content=b"<html>hi</html>"
        )
    )
    resolver = _example()
    result = fetch_url("https://example.org/page?id=7#top", _settings(), resolve=resolver)

    assert result == FetchedResponse(
        url="https://example.org/page?id=7",
        final_url="https://example.org/page?id=7",
        content_type="text/html",
        charset="utf-8",
        body=b"<html>hi</html>",
        is_pdf=False,
    )
    assert resolver.calls == [("example.org", 443)]
    request = route.calls.last.request
    assert request.url.host == PUBLIC_V4
    assert request.headers["host"] == "example.org"
    assert request.extensions["sni_hostname"] == "example.org"
    assert request.headers["user-agent"] == DEFAULT_USER_AGENT
    assert request.headers["accept"] == (
        "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.1"
    )
    assert request.headers["accept-encoding"] == "gzip, deflate"
    assert request.headers["connection"] == "close"
    assert request.extensions["timeout"]["read"] == pytest.approx(30.0, abs=1.0)


@respx.mock
def test_pdf_fetch_uses_the_upload_cap_not_the_html_cap():
    assert 10_000 < len(PDF_BYTES) < 50_000
    respx.get(f"https://{PUBLIC_V4}/report.pdf").mock(
        return_value=httpx.Response(
            200, headers={"Content-Type": "application/pdf"}, content=PDF_BYTES
        )
    )
    result = fetch_url("https://example.org/report.pdf", _settings(), resolve=_example())
    assert result.is_pdf is True
    assert (result.content_type, result.charset) == ("application/pdf", None)
    assert result.body == PDF_BYTES


@respx.mock
def test_ipv6_address_is_pinned_in_brackets():
    route = respx.get(f"https://[{PUBLIC_V6}]/page").mock(return_value=_html())
    fetch_url("https://example.org/page", _settings(), resolve=_example(PUBLIC_V6))
    request = route.calls.last.request
    assert request.headers["host"] == "example.org"
    assert request.extensions["sni_hostname"] == "example.org"


@respx.mock
def test_non_default_port_stays_on_the_pin_and_in_host_but_not_in_sni():
    https_route = respx.get(f"https://{PUBLIC_V4}:8443/x").mock(return_value=_html())
    http_route = respx.get(f"http://[{PUBLIC_V6}]:8080/y").mock(return_value=_html())
    resolver = Resolver({"example.org": [PUBLIC_V4], "v6.example.org": [PUBLIC_V6]})

    fetch_url("https://example.org:8443/x", _settings(), resolve=resolver)
    fetch_url("http://v6.example.org:8080/y", _settings(), resolve=resolver)

    assert resolver.calls == [("example.org", 8443), ("v6.example.org", 8080)]
    https_request = https_route.calls.last.request
    assert https_request.headers["host"] == "example.org:8443"
    assert https_request.extensions["sni_hostname"] == "example.org"
    http_request = http_route.calls.last.request
    assert http_request.headers["host"] == "v6.example.org:8080"
    assert "sni_hostname" not in http_request.extensions


@respx.mock
def test_ip_literal_url_is_gated_like_any_host_and_gets_a_bracketed_host_header():
    route = respx.get(f"http://[{PUBLIC_V6}]/p").mock(return_value=_html())
    fetch_url(f"http://[{PUBLIC_V6}]/p", _settings(), resolve=Resolver({PUBLIC_V6: [PUBLIC_V6]}))
    assert route.calls.last.request.headers["host"] == f"[{PUBLIC_V6}]"

    with pytest.raises(BlockedAddressError):
        fetch_url(
            "http://127.0.0.1:9/admin",
            _settings(),
            resolve=Resolver({"127.0.0.1": ["127.0.0.1"]}),
        )


@respx.mock
def test_the_first_vetted_address_is_the_one_pinned():
    route = respx.get("https://8.8.8.8/").mock(return_value=_html())
    fetch_url("https://example.org/", _settings(), resolve=_example("8.8.8.8", PUBLIC_V4))
    assert route.call_count == 1


@respx.mock
def test_gzip_page_with_real_world_header_artifacts_is_decoded():
    """Mixed-case, double-spaced, quoted Content-Type parameters; an upper-case
    coding token; a Latin-1 body with CRLF line endings and NBSPs."""
    page = (
        '<!DOCTYPE html>\r\n<html lang="fr"><head>\r\n'
        '<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">\r\n'
        "<title>Café &amp; crème</title></head>\r\n"
        "<body><p>Prix : 3,50 F</p></body></html>\r\n"
    ).encode("iso-8859-1")
    respx.get(f"https://{PUBLIC_V4}/fr").mock(
        return_value=httpx.Response(
            200,
            headers={
                "Content-Type": 'Text/HTML;  Charset="ISO-8859-1"',
                "Content-Encoding": "GZIP",
                "Vary": "Accept-Encoding",
            },
            content=gzip.compress(page),
        )
    )
    result = fetch_url("https://example.org/fr", _settings(), resolve=_example())
    assert result.body == page
    assert (result.content_type, result.charset) == ("text/html", "iso-8859-1")


@pytest.mark.parametrize(
    ("content_type", "body", "is_pdf"),
    [
        ("application/xhtml+xml", b"<?xml version='1.0'?><html/>", False),
        ("application/octet-stream", PDF_BYTES, True),
        ("text/html", b"%PDF-1.4\n%\xc7\xec\x8f\xa2\n", True),  # a mislabeled PDF
    ],
)
def test_accepted_content_types_and_pdf_sniffing(content_type, body, is_pdf):
    with respx.mock:
        respx.get(f"https://{PUBLIC_V4}/doc").mock(
            return_value=httpx.Response(200, headers={"Content-Type": content_type}, content=body)
        )
        result = fetch_url("https://example.org/doc", _settings(), resolve=_example())
    assert (result.content_type, result.is_pdf, result.body) == (content_type, is_pdf, body)


# --- SSRF: blocked destinations ---------------------------------------------


@respx.mock
def test_privately_resolving_host_is_blocked_without_a_request_or_an_ip_leak(monkeypatch):
    warnings = []
    monkeypatch.setattr(fetchmod.logger, "warning", lambda msg, *args: warnings.append(msg % args))
    # No route is mocked: any request at all would raise respx's AllMockedAssertionError.
    with pytest.raises(BlockedAddressError, match="private or reserved network address") as info:
        fetch_url(
            "https://intranet.example.org/wiki",
            _settings(),
            resolve=Resolver({"intranet.example.org": ["10.20.30.40"]}),
        )
    message = str(info.value)
    assert isinstance(info.value, FetchError)
    assert "https://intranet.example.org/wiki" in message
    assert "10.20.30.40" not in message  # recon leak
    assert len(warnings) == 1 and "10.20.30.40" in warnings[0]


@pytest.mark.parametrize(
    "answers",
    [
        [PUBLIC_V4, "127.0.0.1"],
        ["::1", PUBLIC_V4],
        [PUBLIC_V6, "64:ff9b::a9fe:a9fe"],
    ],
)
def test_one_non_public_answer_blocks_the_whole_host(answers):
    with respx.mock:  # no routes: a request to the "good" address would raise
        with pytest.raises(BlockedAddressError):
            fetch_url("https://example.org/", _settings(), resolve=_example(*answers))


def test_default_resolver_gates_loopback_literals_offline():
    # Real getaddrinfo parses IP literals without touching the network.
    with respx.mock:
        for url in ("http://127.0.0.1:9/", "http://[::1]:9/"):
            with pytest.raises(BlockedAddressError):
                fetch_url(url, _settings())


@respx.mock
def test_default_resolver_asks_for_stream_sockets_and_vets_every_answer(monkeypatch):
    lookups = []

    def fake_getaddrinfo(host, port, *args, **kwargs):
        lookups.append((host, port, args, kwargs))
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_V4, port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_V4, port)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fd00::5", port, 0, 0)),
        ]

    monkeypatch.setattr(fetchmod.socket, "getaddrinfo", fake_getaddrinfo)
    assert fetchmod._default_resolve("example.org", 443) == [PUBLIC_V4, "fd00::5"]
    with pytest.raises(BlockedAddressError):
        fetch_url("https://example.org/", _settings())
    assert lookups[-1] == ("example.org", 443, (), {"type": socket.SOCK_STREAM})


@respx.mock
def test_redirect_to_a_privately_resolving_host_is_blocked():
    first = respx.get(f"https://{PUBLIC_V4}/go").mock(
        return_value=httpx.Response(
            302, headers={"Location": "http://metadata.internal/latest/meta-data/"}
        )
    )
    resolver = Resolver({"example.org": [PUBLIC_V4], "metadata.internal": ["169.254.169.254"]})
    with pytest.raises(BlockedAddressError) as info:
        fetch_url("https://example.org/go", _settings(), resolve=resolver)
    assert first.call_count == 1
    assert resolver.calls[-1] == ("metadata.internal", 80)
    message = str(info.value)
    assert "http://metadata.internal/latest/meta-data/" in message
    assert "https://example.org/go" in message
    assert "169.254.169.254" not in message


@respx.mock
def test_dns_rebinding_between_hops_is_caught():
    answers = iter([[PUBLIC_V4], ["127.0.0.1"]])
    respx.get(f"https://{PUBLIC_V4}/start").mock(
        return_value=httpx.Response(302, headers={"Location": "/again"})
    )
    with pytest.raises(BlockedAddressError):
        fetch_url("https://example.org/start", _settings(), resolve=lambda h, p: next(answers))


@pytest.mark.parametrize(
    ("location", "message"),
    [
        ("file:///etc/passwd", "http:// or https://"),
        ("ftp://example.org/pub", "http:// or https://"),
        ("gopher://127.0.0.1:6379/_INFO", "http:// or https://"),
        # httpx itself parses the Location (to build response.next_request, even
        # with follow_redirects=False) and raises InvalidURL on this one.
        ("javascript:alert(1)", "invalid redirect Location"),
    ],
)
def test_redirect_to_a_non_http_scheme_is_refused(location, message):
    resolver = _example()
    with respx.mock:
        respx.get(f"https://{PUBLIC_V4}/go").mock(
            return_value=httpx.Response(302, headers={"Location": location})
        )
        with pytest.raises(FetchError, match=message) as info:
            fetch_url("https://example.org/go", _settings(), resolve=resolver)
    assert not isinstance(info.value, BlockedAddressError)
    assert resolver.calls == [("example.org", 443)]
    assert "https://example.org/go" in str(info.value)


@respx.mock
def test_redirect_carrying_credentials_is_refused_without_echoing_them():
    respx.get(f"https://{PUBLIC_V4}/go").mock(
        return_value=httpx.Response(302, headers={"Location": "https://bob:s3cret@example.org/in"})
    )
    with pytest.raises(FetchError, match="username or password") as info:
        fetch_url("https://example.org/go", _settings(), resolve=_example())
    assert "s3cret" not in str(info.value)


@respx.mock
def test_a_supplied_redirect_following_client_cannot_bypass_the_gate():
    respx.get(f"https://{PUBLIC_V4}/go").mock(
        return_value=httpx.Response(302, headers={"Location": "http://intranet.example.org/"})
    )
    resolver = Resolver({"example.org": [PUBLIC_V4], "intranet.example.org": ["10.0.0.7"]})
    with httpx.Client(trust_env=False, follow_redirects=True) as client:
        # If httpx followed the redirect itself, it would request the unmocked
        # hostname URL and respx would raise instead.
        with pytest.raises(BlockedAddressError):
            fetch_url("https://example.org/go", _settings(), client=client, resolve=resolver)
        assert not client.is_closed  # the caller owns a supplied client


def test_a_supplied_client_that_trusts_the_environment_is_refused():
    with httpx.Client() as client:  # trust_env defaults to True
        with pytest.raises(ValueError, match="trust_env=False"):
            fetch_url("https://example.org/", _settings(), client=client, resolve=Resolver({}))


@respx.mock
def test_the_fetchers_own_client_is_unproxied_non_redirecting_and_always_closed(monkeypatch):
    created = []

    class SpyClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(fetchmod.httpx, "Client", SpyClient)
    respx.get(f"https://{PUBLIC_V4}/ok").mock(return_value=_html())
    respx.get(f"https://{PUBLIC_V4}/missing").mock(return_value=httpx.Response(404))

    fetch_url("https://example.org/ok", _settings(), resolve=_example())
    with pytest.raises(FetchError, match="HTTP 404"):
        fetch_url("https://example.org/missing", _settings(), resolve=_example())

    assert len(created) == 2
    for client in created:
        assert client.trust_env is False
        assert client.follow_redirects is False
        assert client.is_closed


# --- redirects --------------------------------------------------------------


@respx.mock
def test_relative_location_is_resolved_against_the_logical_url_and_re_resolved():
    respx.get(f"https://{PUBLIC_V4}/a/b/page").mock(
        return_value=httpx.Response(301, headers={"Location": "../c/next?x=1"})
    )
    second = respx.get(f"https://{PUBLIC_V4}/a/c/next?x=1").mock(return_value=_html())
    resolver = _example()

    result = fetch_url("https://example.org/a/b/page", _settings(), resolve=resolver)

    assert result.url == "https://example.org/a/b/page"
    assert result.final_url == "https://example.org/a/c/next?x=1"
    assert second.calls.last.request.headers["host"] == "example.org"
    assert resolver.calls == [("example.org", 443), ("example.org", 443)]  # every hop


@respx.mock
def test_cross_host_redirect_re_pins_and_resends_host_and_sni():
    respx.get(f"http://{PUBLIC_V4}/").mock(
        return_value=httpx.Response(308, headers={"Location": "https://www.example.org/home"})
    )
    second = respx.get("https://8.8.8.8/home").mock(return_value=_html())
    resolver = Resolver({"example.org": [PUBLIC_V4], "www.example.org": ["8.8.8.8"]})

    result = fetch_url("http://example.org/", _settings(), resolve=resolver)

    assert result.final_url == "https://www.example.org/home"
    request = second.calls.last.request
    assert request.headers["host"] == "www.example.org"
    assert request.extensions["sni_hostname"] == "www.example.org"


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_every_redirect_status_is_followed(status):
    with respx.mock:
        respx.get(f"https://{PUBLIC_V4}/old").mock(
            return_value=httpx.Response(status, headers={"Location": "/new"})
        )
        respx.get(f"https://{PUBLIC_V4}/new").mock(return_value=_html())
        result = fetch_url("https://example.org/old", _settings(), resolve=_example())
    assert result.final_url == "https://example.org/new"


@pytest.mark.parametrize("status", [300, 304])
def test_other_3xx_statuses_are_errors_not_redirects(status):
    with respx.mock:
        respx.get(f"https://{PUBLIC_V4}/old").mock(
            return_value=httpx.Response(status, headers={"Location": "/new"})
        )
        with pytest.raises(FetchError, match=f"HTTP {status}"):
            fetch_url("https://example.org/old", _settings(), resolve=_example())


@respx.mock
def test_redirect_cap_is_enforced_and_the_next_hop_is_never_requested():
    for i in range(3):
        respx.get(f"https://{PUBLIC_V4}/{i}").mock(
            return_value=httpx.Response(302, headers={"Location": f"/{i + 1}"})
        )
    # /3 has no route: requesting it would raise instead of FetchError.
    with pytest.raises(FetchError, match="more than 2 redirects"):
        fetch_url("https://example.org/0", _settings(fetch_max_redirects=2), resolve=_example())


@respx.mock
def test_exactly_the_maximum_number_of_redirects_succeeds():
    for i in range(2):
        respx.get(f"https://{PUBLIC_V4}/{i}").mock(
            return_value=httpx.Response(302, headers={"Location": f"/{i + 1}"})
        )
    respx.get(f"https://{PUBLIC_V4}/2").mock(return_value=_html())
    result = fetch_url(
        "https://example.org/0", _settings(fetch_max_redirects=2), resolve=_example()
    )
    assert result.final_url == "https://example.org/2"


@respx.mock
def test_redirect_without_a_location_is_an_error():
    respx.get(f"https://{PUBLIC_V4}/go").mock(return_value=httpx.Response(302))
    with pytest.raises(FetchError, match="without a Location header"):
        fetch_url("https://example.org/go", _settings(), resolve=_example())


# --- status, type, and size -------------------------------------------------


@respx.mock
def test_non_2xx_after_a_redirect_names_both_urls_and_the_status():
    respx.get(f"https://{PUBLIC_V4}/old").mock(
        return_value=httpx.Response(301, headers={"Location": "/gone"})
    )
    respx.get(f"https://{PUBLIC_V4}/gone").mock(return_value=httpx.Response(410))
    with pytest.raises(FetchError) as info:
        fetch_url("https://example.org/old", _settings(), resolve=_example())
    message = str(info.value)
    assert "HTTP 410" in message
    assert "https://example.org/gone" in message and "https://example.org/old" in message


@pytest.mark.parametrize("content_type", ["image/png", "text/plain", "application/json", None])
def test_unsupported_content_type_is_refused_before_the_body_is_read(content_type):
    pulled = []
    headers = {"Content-Type": content_type} if content_type else {}
    with respx.mock:
        respx.get(f"https://{PUBLIC_V4}/f").mock(
            return_value=httpx.Response(
                200, headers=headers, content=_tracked([b"x" * 100] * 3, pulled)
            )
        )
        with pytest.raises(FetchError, match="unsupported content type"):
            fetch_url("https://example.org/f", _settings(), resolve=_example())
    assert pulled == []


@pytest.mark.parametrize(
    ("content_type", "declared"),
    [("text/html", 10_001), ("application/pdf", 50_001)],
)
def test_declared_content_length_over_the_cap_fails_before_reading(content_type, declared):
    pulled = []
    with respx.mock:
        respx.get(f"https://{PUBLIC_V4}/big").mock(
            return_value=httpx.Response(
                200,
                headers={"Content-Type": content_type, "Content-Length": str(declared)},
                content=_tracked([b"%PDF-" + b"x" * 100], pulled),
            )
        )
        with pytest.raises(FetchError, match="exceeds"):
            fetch_url("https://example.org/big", _settings(), resolve=_example())
    assert pulled == []


@respx.mock
def test_streamed_body_over_the_cap_aborts_at_the_crossing_chunk():
    pulled = []
    respx.get(f"https://{PUBLIC_V4}/endless").mock(
        return_value=httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            content=_tracked([b"a" * 4096] * 100, pulled),
        )
    )
    with pytest.raises(FetchError, match="exceeds"):
        fetch_url("https://example.org/endless", _settings(), resolve=_example())
    assert len(pulled) == 3  # 8192 <= 10000 < 12288: nothing past the crossing chunk


@pytest.mark.parametrize("declared", [False, True], ids=["streamed", "content-length"])
@pytest.mark.parametrize(
    ("content_type", "limit"), [("text/html", 10_000), ("application/pdf", 50_000)]
)
def test_a_body_of_exactly_the_cap_is_accepted_and_one_more_byte_is_not(
    content_type, limit, declared
):
    """Pins both size guards at the boundary: the declared Content-Length check
    (content-length) and the running count of decoded bytes (streamed)."""

    def served(size: int) -> tuple[bytes, httpx.Response]:
        pdf = content_type == "application/pdf"
        head = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n" if pdf else b"<!DOCTYPE html>"
        body = head + b"a" * (size - len(head))
        headers = {"Content-Type": content_type}
        if declared:
            headers["Content-Length"] = str(size)
        chunks = [body[i : i + 4096] for i in range(0, size, 4096)]
        return body, httpx.Response(200, headers=headers, content=_tracked(chunks, []))

    exact_body, exact = served(limit)
    _, over = served(limit + 1)
    assert len(exact_body) == limit
    with respx.mock:
        respx.get(f"https://{PUBLIC_V4}/exact").mock(return_value=exact)
        respx.get(f"https://{PUBLIC_V4}/over").mock(return_value=over)
        result = fetch_url("https://example.org/exact", _settings(), resolve=_example())
        with pytest.raises(FetchError, match="exceeds"):
            fetch_url("https://example.org/over", _settings(), resolve=_example())
    assert result.body == exact_body


@respx.mock
def test_decompression_bomb_is_measured_in_decoded_bytes():
    bomb = gzip.compress(b"\0" * 5_000_000)
    assert len(bomb) < 10_000  # the declared (compressed) size sails under the cap
    # A streamed body: httpx.Response(content=<bytes>) would decode eagerly in
    # the test itself, and the fetcher would never see the compressed stream.
    respx.get(f"https://{PUBLIC_V4}/bomb").mock(
        return_value=httpx.Response(
            200,
            headers={
                "Content-Type": "text/html",
                "Content-Encoding": "gzip",
                "Content-Length": str(len(bomb)),
            },
            content=_tracked([bomb], []),
        )
    )
    with pytest.raises(FetchError, match="exceeds"):
        fetch_url("https://example.org/bomb", _settings(), resolve=_example())


@pytest.mark.parametrize("content_type", ["application/pdf", "application/octet-stream"])
def test_pdf_typed_body_without_the_pdf_magic_is_refused(content_type):
    with respx.mock:
        respx.get(f"https://{PUBLIC_V4}/file").mock(
            return_value=httpx.Response(
                200, headers={"Content-Type": content_type}, content=b"<html>Access denied</html>"
            )
        )
        with pytest.raises(FetchError, match="not a PDF"):
            fetch_url("https://example.org/file", _settings(), resolve=_example())


@respx.mock
def test_undecodable_content_encoding_is_refused_before_reading():
    pulled = []
    respx.get(f"https://{PUBLIC_V4}/br").mock(
        return_value=httpx.Response(
            200,
            headers={"Content-Type": "text/html", "Content-Encoding": "br"},
            content=_tracked([b"\x1b\x03\x00"], pulled),
        )
    )
    with pytest.raises(FetchError, match="unsupported Content-Encoding"):
        fetch_url("https://example.org/br", _settings(), resolve=_example())
    assert pulled == []


@respx.mock
def test_malformed_content_length_is_an_error():
    respx.get(f"https://{PUBLIC_V4}/cl").mock(
        return_value=httpx.Response(
            200,
            headers={"Content-Type": "text/html", "Content-Length": "12abc"},
            content=_tracked([b"<html></html>"], []),
        )
    )
    with pytest.raises(FetchError, match="malformed Content-Length"):
        fetch_url("https://example.org/cl", _settings(), resolve=_example())


@respx.mock
def test_corrupt_gzip_surfaces_as_a_fetch_error():
    respx.get(f"https://{PUBLIC_V4}/corrupt").mock(
        return_value=httpx.Response(
            200,
            headers={"Content-Type": "text/html", "Content-Encoding": "gzip"},
            content=_tracked([b"this is not gzip"], []),  # streamed: decoded by the fetcher
        )
    )
    with pytest.raises(FetchError, match="https://example.org/corrupt") as info:
        fetch_url("https://example.org/corrupt", _settings(), resolve=_example())
    assert isinstance(info.value.__cause__, httpx.DecodingError)


# --- transport and DNS failures ---------------------------------------------


@respx.mock
def test_transport_errors_surface_as_fetch_errors_naming_the_url():
    respx.get(f"https://{PUBLIC_V4}/down").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(FetchError, match="https://example.org/down") as info:
        fetch_url("https://example.org/down", _settings(), resolve=_example())
    assert isinstance(info.value.__cause__, httpx.ConnectError)


@respx.mock
def test_httpx_timeouts_surface_as_fetch_errors():
    respx.get(f"https://{PUBLIC_V4}/slow").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(FetchError, match="timed out"):
        fetch_url("https://example.org/slow", _settings(), resolve=_example())


def _unresolvable(host, port):
    raise socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided, or not known")


@pytest.mark.parametrize(
    ("resolve", "message"),
    [
        (_unresolvable, "could not be resolved"),
        (lambda host, port: [], "no addresses"),
        (lambda host, port: ["not-an-ip"], "unparseable"),
    ],
)
def test_dns_failures_are_fetch_errors_naming_the_url(resolve, message):
    with respx.mock:
        with pytest.raises(FetchError, match=message) as info:
            fetch_url("https://example.org/x", _settings(), resolve=resolve)
    assert not isinstance(info.value, BlockedAddressError)
    assert "https://example.org/x" in str(info.value)


def test_invalid_input_url_is_a_fetch_error():
    with pytest.raises(FetchError, match="http:// or https://"):
        fetch_url("ftp://example.org/x", _settings(), resolve=Resolver({}))


# --- the total time budget --------------------------------------------------


@respx.mock
def test_time_budget_is_checked_between_hops(clock):
    def slow_redirect(request):
        clock.now += 31
        return httpx.Response(302, headers={"Location": "/next"})

    respx.get(f"https://{PUBLIC_V4}/start").mock(side_effect=slow_redirect)
    resolver = _example()
    with pytest.raises(FetchError, match=r"timed out.*time budget") as info:
        fetch_url("https://example.org/start", _settings(), resolve=resolver)
    assert resolver.calls == [("example.org", 443)]  # the next hop never even resolved
    assert "https://example.org/start" in str(info.value)


@respx.mock
def test_time_budget_is_checked_while_streaming_the_body(clock):
    pulled = []

    def trickle():
        for i in range(50):
            pulled.append(i)
            if i == 1:
                clock.now += 31
            yield b"<p>chunk</p>"

    respx.get(f"https://{PUBLIC_V4}/trickle").mock(
        return_value=httpx.Response(200, headers={"Content-Type": "text/html"}, content=trickle())
    )
    with pytest.raises(FetchError, match=r"timed out.*time budget"):
        fetch_url("https://example.org/trickle", _settings(), resolve=_example())
    assert len(pulled) == 2


@respx.mock
def test_each_request_timeout_is_capped_by_the_remaining_budget(clock):
    def slow_redirect(request):
        clock.now += 25
        return httpx.Response(302, headers={"Location": "/next"})

    first = respx.get(f"https://{PUBLIC_V4}/start").mock(side_effect=slow_redirect)
    second = respx.get(f"https://{PUBLIC_V4}/next").mock(return_value=_html())
    fetch_url("https://example.org/start", _settings(), resolve=_example())
    full, rest = (dict.fromkeys(("connect", "read", "write", "pool"), s) for s in (30.0, 5.0))
    assert first.calls.last.request.extensions["timeout"] == full
    assert second.calls.last.request.extensions["timeout"] == rest


# --- live network (excluded in CI) ------------------------------------------


@pytest.mark.integration
def test_real_fetch_pins_the_ip_and_verifies_the_certificate_against_the_hostname():
    """Live network. Plain success proves little on its own — example.com's
    edge also completes TLS for a bare-IP connection. So the live TLS socket is
    inspected: the request went to an IP literal, and the handshake ran with
    server_hostname=example.com under a CERT_REQUIRED + check_hostname context,
    i.e. the certificate was verified against the hostname."""
    settings = Settings()
    result = fetch_url("https://example.com/", settings)  # the production path: own client
    assert result.final_url == "https://example.com/"
    assert result.content_type == "text/html" and not result.is_pdf
    assert b"Example Domain" in result.body

    seen = {}

    def inspect_tls(response):
        tls = response.extensions["network_stream"].get_extra_info("socket")
        seen.update(
            request=response.request,
            server_hostname=tls.server_hostname,
            check_hostname=tls.context.check_hostname,
            verify_mode=tls.context.verify_mode,
            dns_names=[v for k, v in tls.getpeercert().get("subjectAltName", ()) if k == "DNS"],
        )

    with httpx.Client(trust_env=False, event_hooks={"response": [inspect_tls]}) as client:
        fetch_url("https://example.com/", settings, client=client)
    ipaddress.ip_address(seen["request"].url.host)  # raises unless sent to an IP literal
    assert seen["request"].headers["host"] == "example.com"
    assert seen["server_hostname"] == "example.com"
    assert seen["check_hostname"] is True
    assert seen["verify_mode"] == ssl.CERT_REQUIRED
    assert any(name in ("example.com", "*.example.com") for name in seen["dns_names"])
