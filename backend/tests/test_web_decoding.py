"""How web._decode turns a page's bytes into text: which declaration wins, and how
a declaration nothing can decode with fails (loudly, naming the link)."""

import codecs
import logging

import pytest

from authorai.web import ThinPageError, _decode, extract_web
from tests.test_web import DIARIO_URL, _body, _diario

_SPANISH = "<p>La Organización Mundial — “agua”, €5 millones, œuvre</p>"


# --- a UTF-8 byte-order mark -------------------------------------------------


@pytest.mark.parametrize(
    "meta",
    ['<meta charset="windows-1252">', '<meta charset="iso-8859-1">', ""],
    ids=["meta-windows-1252", "meta-iso-8859-1", "no-meta"],
)
def test_a_utf8_byte_order_mark_beats_a_meta_charset_even_with_a_stray_byte(web_log, meta):
    """WHATWG: a byte-order mark decides the encoding. One invalid byte costs one
    replacement character and a warning naming the link, never a whole page read
    as windows-1252 (every accent stored as mojibake) or a failure claiming the
    page has no byte-order mark."""
    page = _diario(meta)
    raw = codecs.BOM_UTF8 + page.encode("utf-8").replace("São Paulo".encode(), b"S\xe3o Paulo")
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8-sig")

    document, metadata = extract_web(raw, url=DIARIO_URL)

    body = _body(document)
    assert "La Organización Mundial de la Salud informó" in body
    assert "S�o Paulo" in body
    assert "Ã" not in body
    assert metadata.authors == ["Lucía Fernández Ibáñez"]
    naming = [r for r in web_log.records if DIARIO_URL in r.getMessage()]
    assert naming and all(r.levelno == logging.WARNING for r in naming)
    assert "UTF-8 byte-order mark" in naming[0].getMessage()


def test_a_utf8_byte_order_mark_is_dropped_and_its_stray_byte_replaced():
    page = '<meta charset="windows-1252">' + _SPANISH
    assert _decode(codecs.BOM_UTF8 + page.encode("utf-8") + b"\x92", DIARIO_URL) == page + "�"


def test_valid_utf8_behind_a_byte_order_mark_reads_unchanged_and_quietly(web_log):
    page = _diario('<meta charset="windows-1252">')
    raw = codecs.BOM_UTF8 + page.encode("utf-8")

    assert _decode(raw, DIARIO_URL) == page
    assert extract_web(raw, url=DIARIO_URL) == extract_web(page, url=DIARIO_URL)
    assert not web_log.records


# --- a label nothing can decode with -----------------------------------------


@pytest.mark.parametrize(
    "label", ["utf-8\x00", "\x00", " \x00 "], ids=["utf8-nul", "nul", "padded"]
)
def test_a_meta_charset_holding_a_nul_byte_fails_like_an_unknown_one(label):
    """codecs.lookup raises ValueError('embedded null character') for such a label:
    a run error that named no link, matched no failure hint, and flagged no row."""
    page = _diario(f'<meta charset="{label}">')
    with pytest.raises(ValueError, match="declares an unknown charset") as excinfo:
        extract_web(page.encode("cp1252"), url=DIARIO_URL)
    assert str(excinfo.value).startswith(DIARIO_URL)
    assert repr(label.strip()) in str(excinfo.value)
    assert not isinstance(excinfo.value, ThinPageError)


@pytest.mark.parametrize("label", ["us-ascii", "ascii", "ansi_x3.4-1968"])
def test_an_ascii_meta_charset_reads_as_windows_1252(web_log, label):
    """WHATWG maps the ascii labels to windows-1252, as it does iso-8859-1: read with
    Python's strict ascii codec instead, every accent, curly quote, dash, € and œ
    would be stored as a replacement character."""
    page = f'<meta charset="{label}">' + _SPANISH
    raw = page.encode("cp1252")
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    assert _decode(raw, DIARIO_URL) == page
    assert not web_log.records
