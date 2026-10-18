"""Which characters of a PDF a reader can see.

A report's author can put text in a PDF that no reader sees but every text
extractor returns: white on white, covered by a box, invisible render mode,
clipped away, a hidden layer, a glyph too small to see, text off the page.
Such text reaches the model calls that read the report (claim extraction,
validity, the bibliography scan) and can change what they find without being
an instruction a fence could stop. This module finds it, character by
character, so the readers can leave it out.

How: for each page, pdfium's text page gives every character with its
attributes (render mode, fill colour and opacity, matrix, boxes, marks, clip
path), and the page is rendered twice — as it is, and with every text object
switched off. A character is visible only if it leaves ink in the difference,
whatever PDF feature hides it; attribute rules settle what the picture cannot
(off the page, too small, squashed, an OCR layer). A character touching
another object's character is judged again with its own object switched off
and every object touching it still drawn, so it cannot borrow its
neighbour's ink (a hidden line under a box that carries visible text would
otherwise keep the letters that touch the box's text); objects that do not
touch share a render. The rules are RULES, and the module's
judgment is named by DETECTOR_VERSION; a change to either is a change to
what every reader reads.

The OCR exception: a scanned page carries its text as an invisible layer
(render mode 3) over the scan's image. That layer is the only text such a
page has and is kept — when it lies on an image, the image under it has ink,
and images cover most of the page. A layer that disagrees with its scan, and
/ActualText (the text a reader extracts differs from the glyphs drawn), are
known gaps.

pdfium is not thread-safe and Docling uses it in the server process, so
check_pdf runs only in a child process: detect_hidden_text runs it in one
(web.in_bounded_child, with the scan's memory bound), and a reader that is
already a child (the bibliography scan) may call check_pdf directly.

A page the detector cannot check is reported with its reason, never skipped
(HiddenTextReport.unmeasured): too large to render, too many overlapping
text objects, too much text, a render that failed, or past the result's size
budget. A whole document that cannot be checked — unreadable, or out of time
or memory — is a HiddenTextCheckError. The `sample` strings are the author's
own text, for the record and the user's eyes: they never go into a prompt.
"""

from __future__ import annotations

import ctypes
import math
import os
import time
from array import array
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from authorai.log import setup_logger
from authorai.web import (
    ExtractionTimeoutError,
    ReaderExitedError,
    ResultTooLargeError,
    end_with_parent,
    handover_file,
    in_bounded_child,
    report_outcome,
)

logger = setup_logger(__name__)

DETECTOR_VERSION = "hidden-text v1"

# The rules, measured on 54 planted PDFs and 343 real pages (records in
# author_ai/report-side-research/hidden-text), each with a planted pair on
# either side in tests/fixtures/hidden_text.
RULES = {
    # share of a character's loose box inside the CropBox
    "MIN_IN_CROP": 0.5,
    # effective glyph height in points: font size x the matrix's vertical scale
    "MIN_SIZE_PT": 1.0,
    # the character matrix's horizontal scale over its vertical one (Tz, skew)
    "MIN_HSCALE": 0.2,
    # share of the tight box's pixels that change when the text is drawn ...
    "MIN_INK_SHARE": 0.02,
    # ... by at least this much in some channel (0-255)
    "MIN_INK_DIFF": 16,
    # WCAG contrast of the fill colour against the rendered background
    "MIN_CONTRAST": 1.2,
    # the OCR-layer exception: share of the loose box lying on an image ...
    "OCR_MIN_OVER_IMAGE": 0.9,
    # ... whose luminance spread under the word (its most-inked character) is at least ...
    "OCR_MIN_BG_SPREAD": 8.0,
    # ... on a page whose images cover at least this share of the CropBox
    "OCR_MIN_PAGE_IMAGE": 0.5,
    # overlap pass: two equal characters are copies of one glyph at this IoU
    "DUPLICATE_IOU": 0.6,
}

# Rendering. The rules were measured at scale 2 (1.5 and 3 read the same;
# 1.0 lost some light-grey ink), so a page too large for MAX_RENDER_PIXELS
# at 2 is rendered smaller, down to MIN_RENDER_SCALE, and past that is not
# checked: a page's renders are its main memory cost (three copies of it in
# the overlap pass), and 16 MP keeps them far under the memory bound.
RENDER_SCALE = 2.0
MIN_RENDER_SCALE = 1.5
MAX_RENDER_PIXELS = 16_000_000

# Bounds on one page's work and on what the result may carry. One render per
# group of touching text objects (see _rejudge_overlaps): a real page needs a
# few. 50,000 characters is about eight times the densest real page. The
# glyph lists (kept only for pages with something removed) pickle at 22 bytes
# a character (measured on two real pages), so 2,000,000 is about 44 MB, under
# web.RESULT_MAX_BYTES (64 MiB).
MAX_OVERLAP_RENDERS = 64
# pdfium releases what it keeps from loaded pages only when the document is
# closed: one opening per 5 pages kept a 261-page colour scan at a 276 MiB
# peak (702 MiB in one opening, 228 MiB opening per page at +27% time).
PAGES_PER_OPEN = 5
PAGE_MAX_CHARS = 50_000
RESULT_MAX_GLYPHS = 2_000_000
SAMPLE_CHARS = 120

HIDDEN_KINDS = (
    "off_page",
    "invisible_render_mode",
    "tiny",
    "squashed",
    "transparent",
    "hidden_layer",
    "same_colour_as_background",
    "clipped",
    "covered",
    "low_contrast",
)
# An earlier copy of a glyph drawn over by an identical later copy: removed
# from the reading (the reader sees one copy), counted apart from the hidden.
DUPLICATE = "duplicate"
# An invisible OCR layer over its scan: kept, counted.
OCR_LAYER_KEPT = "ocr_layer_kept"
KINDS: tuple[str | None, ...] = (None, *HIDDEN_KINDS, DUPLICATE, OCR_LAYER_KEPT)
_CODE = {kind: code for code, kind in enumerate(KINDS)}
_REMOVED = frozenset(_CODE[kind] for kind in (*HIDDEN_KINDS, DUPLICATE))

_TEXT = pdfium_c.FPDF_PAGEOBJ_TEXT
_IMAGE = pdfium_c.FPDF_PAGEOBJ_IMAGE

# The reasons a page is not checked, as the run shows them.
TOO_LARGE = "page too large to check"
TOO_MANY_OVERLAPS = "too many overlapping text objects"
TOO_MUCH_TEXT = "too much text on the page"
NOT_RENDERED = "the page could not be rendered"
RESULT_TOO_LARGE = "result too large"


class HiddenTextCheckError(RuntimeError):
    """The document could not be checked at all; the message is the user's."""


class _PageNotChecked(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Glyphs:
    """A checked page's characters, kept only for a page with something
    removed: what R2 aligns with Docling's reading and the scan cuts from
    pdfium's text. Columnar so it pickles small. `indices` are pdfium
    text-page indices; `boxes` holds four floats per glyph, the loose box
    (left, bottom, right, top) in PDF points; `kinds` one code per glyph,
    an index into KINDS (0 = visible)."""

    indices: array
    text: str
    boxes: array
    kinds: bytes

    def __len__(self) -> int:
        return len(self.text)

    def kind(self, i: int) -> str | None:
        return KINDS[self.kinds[i]]

    def removed(self, i: int) -> bool:
        return self.kinds[i] in _REMOVED

    def box(self, i: int) -> tuple[float, float, float, float]:
        return tuple(self.boxes[4 * i : 4 * i + 4])  # type: ignore[return-value]


@dataclass(frozen=True)
class PageCheck:
    """One checked page. `hidden` maps a hidden kind to (characters, sample);
    duplicates and kept OCR characters are counted apart."""

    page: int
    chars: int
    hidden: dict[str, tuple[int, str]]
    duplicates: int
    ocr_layer_kept: int
    scale: float
    glyphs: Glyphs | None
    seconds: float = field(default=0.0, compare=False)

    @property
    def hidden_chars(self) -> int:
        return sum(count for count, _ in self.hidden.values())


@dataclass(frozen=True)
class HiddenTextReport:
    version: str
    rules: dict[str, float]
    page_count: int
    pages: tuple[PageCheck, ...]
    unmeasured: tuple[tuple[int, str], ...]
    peak_memory_bytes: int | None = field(default=None, compare=False)

    @property
    def hidden_chars(self) -> int:
        return sum(check.hidden_chars for check in self.pages)

    @property
    def pages_with_hidden(self) -> tuple[int, ...]:
        return tuple(check.page for check in self.pages if check.hidden)


# ------------------------------------------------------------------ the core


def check_pdf(
    source: str | Path | bytes,
    *,
    pages: Iterable[int] | None = None,
    scale: float = RENDER_SCALE,
) -> HiddenTextReport:
    """Check every page (or the 1-based `pages`) of a PDF. Run it only in a
    child process (see the module docstring). Raises ValueError for a file
    pdfium cannot open; a page that cannot be checked is reported in
    `unmeasured` and the other pages go on.

    The document is opened afresh every PAGES_PER_OPEN pages: pdfium keeps
    document-wide data from the pages it has loaded (a scan's images among
    them) until the document is closed, so a long scanned report read in one
    opening grew past the reader's memory bound."""
    doc = _open(source)
    try:
        page_count = len(doc)
    finally:
        doc.close()
    numbers = range(1, page_count + 1) if pages is None else sorted(set(pages))
    for number in numbers:
        if not 1 <= number <= page_count:
            raise ValueError(f"page {number} is not in a PDF of {page_count} pages")
    checked: list[PageCheck] = []
    unmeasured: list[tuple[int, str]] = []
    glyph_budget = RESULT_MAX_GLYPHS
    for first in range(0, len(numbers), PAGES_PER_OPEN):
        doc = _open(source)
        try:
            for number in numbers[first : first + PAGES_PER_OPEN]:
                try:
                    check = _check_page(doc, number, scale)
                except _PageNotChecked as exc:
                    unmeasured.append((number, exc.reason))
                    continue
                except MemoryError:
                    raise
                except Exception:  # noqa: BLE001 - one page's failure is that page's, reported
                    logger.exception("page %d could not be checked for hidden text", number)
                    unmeasured.append((number, NOT_RENDERED))
                    continue
                if check.glyphs is not None:
                    if len(check.glyphs) > glyph_budget:
                        unmeasured.append((number, RESULT_TOO_LARGE))
                        continue
                    glyph_budget -= len(check.glyphs)
                checked.append(check)
        finally:
            doc.close()
    return HiddenTextReport(
        version=DETECTOR_VERSION,
        rules=dict(RULES),
        page_count=page_count,
        pages=tuple(checked),
        unmeasured=tuple(unmeasured),
    )


def _open(source: str | Path | bytes):
    try:
        return pdfium.PdfDocument(str(source) if isinstance(source, Path) else source)
    except pdfium.PdfiumError as exc:
        raise ValueError(f"the PDF could not be opened ({exc})") from exc


@dataclass
class _Char:
    index: int
    char: str
    box: tuple[float, float, float, float]  # tight
    loose: tuple[float, float, float, float]
    mode: int
    rgba: tuple[int, int, int, int] | None
    hscale: float
    vsize: float
    marks: tuple[str, ...]
    clipped: bool
    obj: int
    in_crop: float = 0.0
    over_image: float = 0.0
    ink_px: int = 0
    box_px: int = 0
    bg: tuple[int, int, int] | None = None
    bg_spread: float = 0.0
    earlier_copy: bool = False
    kind: str | None = None


def _check_page(doc, number: int, scale: float) -> PageCheck:
    started = time.perf_counter()
    page = doc[number - 1]
    try:
        scale = _fitting_scale(page, scale)
        textpage = page.get_textpage()
        try:
            chars = _characters(page, textpage)
        finally:
            textpage.close()
        if chars:
            _measure(page, chars, scale, number)
            _classify(chars, _image_cover(page, chars))
    finally:
        page.close()
    return _summary(number, chars, scale, time.perf_counter() - started)


def _fitting_scale(page, scale: float) -> float:
    width, height = page.get_size()
    if width * height * scale * scale <= MAX_RENDER_PIXELS:
        return scale
    fitted = math.floor(math.sqrt(MAX_RENDER_PIXELS / (width * height)) * 1000) / 1000
    if fitted < MIN_RENDER_SCALE:
        raise _PageNotChecked(TOO_LARGE)
    return fitted


def _characters(page, textpage) -> list[_Char]:
    """Every non-whitespace character the text page holds (pdfium's own
    generated spaces and line breaks are not the document's), with the
    attributes the rules read."""
    crop = page.get_cropbox()
    chars: list[_Char] = []
    for i in range(textpage.count_chars()):
        if pdfium_c.FPDFText_IsGenerated(textpage.raw, i) == 1:
            continue
        code = pdfium_c.FPDFText_GetUnicode(textpage.raw, i)
        char = _as_char(code)
        if not char.strip():
            continue
        if len(chars) >= PAGE_MAX_CHARS:
            raise _PageNotChecked(TOO_MUCH_TEXT)
        obj = pdfium_c.FPDFText_GetTextObject(textpage.raw, i)
        r, g, b, a = (ctypes.c_uint() for _ in range(4))
        has_fill = pdfium_c.FPDFText_GetFillColor(
            textpage.raw, i, ctypes.byref(r), ctypes.byref(g), ctypes.byref(b), ctypes.byref(a)
        )
        matrix = pdfium_c.FS_MATRIX()
        pdfium_c.FPDFText_GetMatrix(textpage.raw, i, ctypes.byref(matrix))
        size = pdfium_c.FPDFText_GetFontSize(textpage.raw, i)
        row1 = math.hypot(matrix.a, matrix.b)
        row2 = math.hypot(matrix.c, matrix.d)
        loose = textpage.get_charbox(i, loose=True)
        clip = _clip_box(obj) if obj else None
        area = _area(loose)
        chars.append(
            _Char(
                index=i,
                char=char,
                box=textpage.get_charbox(i),
                loose=loose,
                mode=pdfium_c.FPDFTextObj_GetTextRenderMode(obj) if obj else -1,
                rgba=(r.value, g.value, b.value, a.value) if has_fill else None,
                hscale=row1 / row2 if row2 else 0.0,
                vsize=size * row2,
                marks=_marks(obj) if obj else (),
                clipped=clip is not None and area > 0 and _overlap(loose, clip) / area < 0.5,
                obj=_pointer(obj),
                in_crop=_overlap(loose, crop) / area if area else 0.0,
            )
        )
    return chars


def _as_char(code: int) -> str:
    """A text-page code as one character: a lone surrogate or an impossible
    code (a broken font) becomes U+FFFD, so the result always encodes."""
    if not code or code > 0x10FFFF or 0xD800 <= code <= 0xDFFF:
        return "�" if code else ""
    return chr(code)


def _image_cover(page, chars: list[_Char]) -> float:
    """The share of the CropBox covered by image objects (at any depth), and
    each character's share lying on one — for the OCR-layer exception."""
    images = [obj.get_bounds() for obj in page.get_objects(filter=[_IMAGE])]
    for char in chars:
        area = _area(char.loose)
        char.over_image = (
            max((_overlap(char.loose, bounds) / area for bounds in images), default=0.0)
            if area
            else 0.0
        )
    left, bottom, right, top = page.get_cropbox()
    width, height = int(right - left) + 1, int(top - bottom) + 1
    mask = np.zeros((height, width), dtype=bool)  # one cell per point
    for b_left, b_bottom, b_right, b_top in images:
        x0, x1 = int(max(b_left, left) - left), int(min(b_right, right) - left)
        y0, y1 = int(max(b_bottom, bottom) - bottom), int(min(b_top, top) - bottom)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return float(mask.mean())


def _render(page, scale: float, *, number: int) -> np.ndarray:
    """The page as an RGB uint8 array. `number` names the page for a test
    that plants a failure on one page."""
    bitmap = page.render(scale=scale, may_draw_forms=False, rev_byteorder=True)
    try:
        return np.array(bitmap.to_numpy()[..., :3], dtype=np.uint8)
    finally:
        bitmap.close()


def _difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per pixel, the largest channel difference between two renders,
    computed in row blocks so no full-page temporary is made."""
    out = np.empty(a.shape[:2], dtype=np.uint8)
    for top in range(0, a.shape[0], 256):
        block_a, block_b = a[top : top + 256], b[top : top + 256]
        out[top : top + 256] = (np.maximum(block_a, block_b) - np.minimum(block_a, block_b)).max(
            axis=2
        )
    return out


def _measure(page, chars: list[_Char], scale: float, number: int) -> None:
    """Fill each character's ink, background colour and spread from the
    page rendered with and without its text; then re-judge the characters
    that overlap another object's character by switching off only their own
    object."""
    drawn = _render(page, scale, number=number)
    texts = list(page.get_objects(filter=[_TEXT]))
    for obj in texts:
        pdfium_c.FPDFPageObj_SetIsActive(obj.raw, False)
    bare = _render(page, scale, number=number)
    for obj in texts:
        pdfium_c.FPDFPageObj_SetIsActive(obj.raw, True)
    height, width = drawn.shape[:2]
    to_px = _device_map(page, width, height)
    diff = _difference(drawn, bare)
    min_diff = RULES["MIN_INK_DIFF"]
    for char in chars:
        x0, y0, x1, y1 = _pixel_box(to_px, char.box, width, height, pad=1)
        if x1 > x0 and y1 > y0:
            patch = diff[y0:y1, x0:x1]
            char.box_px = patch.size
            char.ink_px = int((patch >= min_diff).sum())
        x0, y0, x1, y1 = _pixel_box(to_px, char.loose, width, height, pad=0)
        if x1 > x0 and y1 > y0:
            behind = bare[y0:y1, x0:x1].reshape(-1, 3)
            char.bg = tuple(int(v) for v in np.median(behind, axis=0))  # type: ignore[assignment]
            char.bg_spread = float(behind.mean(axis=1, dtype=np.float32).std())
    del bare, diff
    _rejudge_overlaps(page, chars, texts, drawn, scale, number, to_px)


def _rejudge_overlaps(page, chars, texts, drawn, scale, number, to_px) -> None:
    """Re-judge every character that pass 1 called visible and whose box
    touches another text object's character: its ink may be that object's.

    A character's own ink is what changes when its object is switched off
    while every object touching it stays drawn. Objects that touch nothing in
    common are switched off together, so the touching objects are put in
    groups no two members of which touch (a greedy colouring of the graph of
    touching objects, in drawing order) and each group costs one render — a
    handful on a real page, where one render per object would leave dense
    pages unchecked.

    Copies of one glyph (equal characters overlapping at DUPLICATE_IOU) need
    their own rule, since switching one copy off leaves its twin's identical
    ink: a copy with a later twin over it is judged alone, as above, and is
    a duplicate when it leaves no ink; the topmost copy is judged with all
    its copies switched off together, so it is visible exactly when the
    glyph is. Those copy sets are grouped the same way, in renders of their
    own."""
    min_share, min_diff = RULES["MIN_INK_SHARE"], RULES["MIN_INK_DIFF"]
    pad = 1.0 / scale  # one device pixel, as the ink box is padded
    grid: dict[tuple[int, int], list[_Char]] = {}
    for char in chars:
        if char.box_px:
            for cell in _cells(char.box):
                grid.setdefault(cell, []).append(char)
    order = {_pointer(obj.raw): position for position, obj in enumerate(page.get_objects())}
    touching: dict[int, set[int]] = {}
    copies = _UnionFind()
    alone: list[_Char] = []  # judged with their own object off
    topmost: list[_Char] = []  # judged with every copy of their glyph off
    for char in chars:
        area = _area(char.box)
        if not area or not char.box_px or char.ink_px / char.box_px < min_share:
            continue
        left, bottom, right, top = char.box
        padded = (left - pad, bottom - pad, right + pad, top + pad)
        touches = has_earlier_twin = False
        for other in {id(o): o for cell in _cells(padded) for o in grid.get(cell, ())}.values():
            if other.obj == char.obj or _overlap(padded, other.box) <= 0:
                continue
            touches = True
            touching.setdefault(char.obj, set()).add(other.obj)
            touching.setdefault(other.obj, set()).add(char.obj)
            shared = _overlap(char.box, other.box)
            union = area + _area(other.box) - shared
            if other.char == char.char and union and shared / union >= RULES["DUPLICATE_IOU"]:
                copies.join(char.obj, other.obj)
                if order.get(char.obj, 0) < order.get(other.obj, 0):
                    char.earlier_copy = True
                else:
                    has_earlier_twin = True
        if not touches:
            continue
        if has_earlier_twin and not char.earlier_copy:
            topmost.append(char)
        else:
            alone.append(char)
    if not alone and not topmost:
        return
    # Copy sets as single nodes: the objects of one set go off together.
    sets = {copies.find(char.obj) for char in topmost}
    set_touching: dict[int, set[int]] = {node: set() for node in sets}
    for obj, others in touching.items():
        node = copies.find(obj)
        if node in sets:
            for other in others:
                other_node = copies.find(other)
                if other_node != node and other_node in sets:
                    set_touching[node].add(other_node)
    object_groups = _non_touching_groups(
        {obj: touching[obj] for obj in {char.obj for char in alone}} if alone else {},
        order,
        touching,
    )
    set_groups = _non_touching_groups(set_touching, order, set_touching)
    if len(object_groups) + len(set_groups) > MAX_OVERLAP_RENDERS:
        raise _PageNotChecked(TOO_MANY_OVERLAPS)
    by_pointer = {_pointer(obj.raw): obj for obj in texts}
    members = copies.members()
    passes = [(group, [c for c in alone if c.obj in set(group)]) for group in object_groups]
    for group in set_groups:
        objects = [obj for node in group for obj in members.get(node, [node])]
        nodes = set(group)
        passes.append((objects, [c for c in topmost if copies.find(c.obj) in nodes]))
    height, width = drawn.shape[:2]
    for objects, judged in passes:
        switched = [by_pointer[pointer] for pointer in objects if pointer in by_pointer]
        for obj in switched:
            pdfium_c.FPDFPageObj_SetIsActive(obj.raw, False)
        without = _render(page, scale, number=number)
        for obj in switched:
            pdfium_c.FPDFPageObj_SetIsActive(obj.raw, True)
        own = _difference(drawn, without)
        del without
        for char in judged:
            x0, y0, x1, y1 = _pixel_box(to_px, char.box, width, height, pad=1)
            if x1 > x0 and y1 > y0:
                char.ink_px = int((own[y0:y1, x0:x1] >= min_diff).sum())


def _non_touching_groups(
    wanted: dict[int, set[int]], order: dict[int, int], touching: dict[int, set[int]]
) -> list[list[int]]:
    """The nodes of `wanted` in groups whose members never touch each other
    (`touching` holds every edge): greedy colouring, in drawing order."""
    colour: dict[int, int] = {}
    for node in sorted(wanted, key=lambda pointer: order.get(pointer, 0)):
        taken = {colour[other] for other in touching.get(node, ()) if other in colour}
        colour[node] = next(c for c in range(len(taken) + 1) if c not in taken)
    groups: list[list[int]] = [[] for _ in range(max(colour.values(), default=-1) + 1)]
    for node, c in colour.items():
        groups[c].append(node)
    return groups


class _UnionFind:
    """Objects joined into sets of copies of one glyph."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, node: int) -> int:
        self.parent.setdefault(node, node)
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def join(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)

    def members(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for node in list(self.parent):
            out.setdefault(self.find(node), []).append(node)
        return out


def _cells(box) -> Iterable[tuple[int, int]]:
    left, bottom, right, top = box
    for gx in range(int(left // 16), int(right // 16) + 1):
        for gy in range(int(bottom // 16), int(top // 16) + 1):
            yield gx, gy


def _classify(chars: list[_Char], image_cover: float) -> None:
    """Apply RULES, in order, to every character (see the module docstring).
    An OCR layer is judged per text object — OCR tools write one object per
    word — by its most-inked character: some OCR engines stretch a word's
    character boxes over blank paper, and a median then set aside real words
    on visible ink ("=Constant" in a held-out scan). A word with no ink under
    any of its characters (OCR junk over a blank margin, or invisible text on
    a blank image) is still set aside."""
    spreads: dict[int, list[float]] = {}
    for char in chars:
        spreads.setdefault(char.obj, []).append(char.bg_spread)
    word_spread = {obj: max(values) for obj, values in spreads.items()}
    for char in chars:
        ink = char.ink_px / char.box_px if char.box_px else 0.0
        if char.in_crop < RULES["MIN_IN_CROP"]:
            char.kind = "off_page"
        elif char.mode in (3, 7):
            ocr = (
                char.over_image >= RULES["OCR_MIN_OVER_IMAGE"]
                and word_spread[char.obj] >= RULES["OCR_MIN_BG_SPREAD"]
                and image_cover >= RULES["OCR_MIN_PAGE_IMAGE"]
            )
            char.kind = OCR_LAYER_KEPT if ocr else "invisible_render_mode"
        elif char.vsize < RULES["MIN_SIZE_PT"]:
            char.kind = "tiny"
        elif char.hscale < RULES["MIN_HSCALE"]:
            char.kind = "squashed"
        elif ink < RULES["MIN_INK_SHARE"]:
            char.kind = _why_no_ink(char)
        elif char.rgba and char.bg and _contrast(char.rgba, char.bg) < RULES["MIN_CONTRAST"]:
            char.kind = "low_contrast"
        else:
            char.kind = None


def _why_no_ink(char: _Char) -> str:
    if char.earlier_copy:
        return DUPLICATE
    if char.rgba and char.rgba[3] == 0:
        return "transparent"
    if "OC" in char.marks:
        return "hidden_layer"
    if char.rgba and char.bg and _contrast(char.rgba, char.bg) < RULES["MIN_CONTRAST"]:
        return "same_colour_as_background"
    if char.clipped:
        return "clipped"
    return "covered"


def _summary(number: int, chars: list[_Char], scale: float, seconds: float) -> PageCheck:
    by_kind: dict[str, list[_Char]] = {}
    for char in chars:
        if char.kind is not None:
            by_kind.setdefault(char.kind, []).append(char)
    hidden = {
        kind: (len(group), "".join(char.char for char in group)[:SAMPLE_CHARS])
        for kind, group in by_kind.items()
        if kind in HIDDEN_KINDS
    }
    duplicates = len(by_kind.get(DUPLICATE, ()))
    glyphs = None
    if hidden or duplicates:
        boxes = array("f")
        for char in chars:
            boxes.extend(char.loose)
        glyphs = Glyphs(
            indices=array("i", (char.index for char in chars)),
            text="".join(char.char for char in chars),
            boxes=boxes,
            kinds=bytes(_CODE[char.kind] for char in chars),
        )
    return PageCheck(
        page=number,
        chars=len(chars),
        hidden=hidden,
        duplicates=duplicates,
        ocr_layer_kept=len(by_kind.get(OCR_LAYER_KEPT, ())),
        scale=scale,
        glyphs=glyphs,
        seconds=round(seconds, 3),
    )


# ------------------------------------------------------------------ geometry


def _pointer(handle) -> int:
    """A pdfium object handle as an int: the identity text-page characters
    and page objects are matched by."""
    return ctypes.cast(handle, ctypes.c_void_p).value or 0


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _device_map(page, width: int, height: int):
    def to_px(x: float, y: float) -> tuple[int, int]:
        dx, dy = ctypes.c_int(), ctypes.c_int()
        pdfium_c.FPDF_PageToDevice(
            page.raw, 0, 0, width, height, 0, x, y, ctypes.byref(dx), ctypes.byref(dy)
        )
        return dx.value, dy.value

    return to_px


def _pixel_box(to_px, box, width: int, height: int, *, pad: int) -> tuple[int, int, int, int]:
    left, bottom, right, top = box
    corners = [to_px(x, y) for x in (left, right) for y in (bottom, top)]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    return (
        max(0, min(xs) - pad),
        max(0, min(ys) - pad),
        min(width, max(xs) + pad),
        min(height, max(ys) + pad),
    )


def _clip_box(obj):
    """The bounding box of an object's clip path (the intersection of its
    paths' boxes), or None when it has none."""
    clip = pdfium_c.FPDFPageObj_GetClipPath(obj)
    if not clip:
        return None
    box = None
    for k in range(pdfium_c.FPDFClipPath_CountPaths(clip)):
        xs, ys = [], []
        for j in range(pdfium_c.FPDFClipPath_CountPathSegments(clip, k)):
            segment = pdfium_c.FPDFClipPath_GetPathSegment(clip, k, j)
            x, y = ctypes.c_float(), ctypes.c_float()
            if pdfium_c.FPDFPathSegment_GetPoint(segment, ctypes.byref(x), ctypes.byref(y)):
                xs.append(x.value)
                ys.append(y.value)
        if xs:
            path_box = (min(xs), min(ys), max(xs), max(ys))
            box = (
                path_box
                if box is None
                else (
                    max(box[0], path_box[0]),
                    max(box[1], path_box[1]),
                    min(box[2], path_box[2]),
                    min(box[3], path_box[3]),
                )
            )
    return box


def _marks(obj) -> tuple[str, ...]:
    names = []
    for k in range(pdfium_c.FPDFPageObj_CountMarks(obj)):
        mark = pdfium_c.FPDFPageObj_GetMark(obj, k)
        length = ctypes.c_ulong(0)
        buffer = (ctypes.c_ushort * 128)()
        if pdfium_c.FPDFPageObjMark_GetName(mark, buffer, 256, ctypes.byref(length)):
            names.append(bytes(buffer)[: max(0, length.value - 2)].decode("utf-16-le", "ignore"))
    return tuple(names)


def _luminance(rgb) -> float:
    def channel(value: float) -> float:
        value = value / 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(v) for v in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(first, second) -> float:
    a, b = _luminance(first), _luminance(second)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


# --------------------------------------------------------- the bounded child


def detect_hidden_text(
    source: BinaryIO | bytes | str | Path,
    *,
    timeout: float,
    pages: Iterable[int] | None = None,
) -> HiddenTextReport:
    """check_pdf in a child process stopped after `timeout` seconds of wall
    clock, under the bibliography scan's memory bound
    (references.bound_child_memory) — so a hostile file meets the same
    deadline, CPU limit, memory bound and parent-death watcher as every
    other reader. `source` is a path, the file's bytes or a readable binary
    handle (read from its start).

    Raises HiddenTextCheckError when the document cannot be checked: pdfium
    cannot open it, or it took longer than `timeout`, or a bound stopped
    the child. A child that ends any other way is the server's fault (an
    import that fails, a bug) and propagates as web.ReaderExitedError, as in
    references.read_pages."""
    from authorai.references import stopped_by_a_bound  # see _check_in_child

    label = "the PDF"
    if isinstance(source, (str, Path)):
        with open(source, "rb") as handle:
            payload_path = handover_file(handle, label)
    else:
        if not isinstance(source, bytes):
            source.seek(0)
        payload_path = handover_file(source, label)
    wanted = None if pages is None else tuple(pages)
    try:
        kind, payload = in_bounded_child(
            _check_in_child, (wanted,), payload_path=payload_path, url=label, timeout=timeout
        )
    except ExtractionTimeoutError as exc:
        raise HiddenTextCheckError(
            f"the PDF could not be checked for hidden text: it took longer than {timeout:g} seconds"
        ) from exc
    except ResultTooLargeError as exc:
        raise HiddenTextCheckError(
            "the PDF could not be checked for hidden text: the result was too large"
        ) from exc
    except ReaderExitedError as exc:
        why = stopped_by_a_bound(exc.exitcode)
        if why is None:
            raise
        raise HiddenTextCheckError(
            f"the PDF could not be checked for hidden text: it was too costly to check ({why})"
        ) from exc
    if kind == "result":
        return payload
    name, message, _child_traceback = payload
    if kind == "value":
        raise HiddenTextCheckError(f"the PDF could not be checked for hidden text: {message}")
    raise HiddenTextCheckError(f"the PDF could not be checked for hidden text ({name}: {message})")


def _check_in_child(
    sender, payload_path: str, pages: tuple[int, ...] | None, cpu_seconds: int
) -> None:
    """The child's side of detect_hidden_text. references is imported here,
    not at the top: the bibliography scan's reader will call check_pdf, so
    references imports this module."""
    from authorai.references import (
        READER_MEMORY_EXIT_CODE,
        bound_child_memory,
        peak_resident_bytes,
    )

    end_with_parent(cpu_seconds)
    bound_child_memory()

    def check() -> HiddenTextReport:
        try:
            report = check_pdf(payload_path, pages=pages)
        except MemoryError:  # the address-space cap, where enforced
            os._exit(READER_MEMORY_EXIT_CODE)
        finally:
            Path(payload_path).unlink(missing_ok=True)
        return replace(report, peak_memory_bytes=peak_resident_bytes())

    report_outcome(sender, check, lambda exc: "value" if isinstance(exc, ValueError) else "other")
