"""Build the hidden-text detector's test PDFs and their manifest.

Each plant is a one-page PDF holding the shared visible sentence plus one
line that is either hidden from a reader by one PDF feature ("hidden") or a
legitimate look-alike a reader does see ("keep"). A threshold pair puts one
plant on each side of a rule's threshold. The manifest's expected texts are
the strings this script planted, never the detector's output, so a test that
compares the two is a known answer, not a copy.

Every operator is written raw (pikepdf), so each plant is exactly what its
name says. No system font is used (text inside an image is drawn as ink
bars) and every random value comes from a seeded generator, so the build is
the same on any machine. Where a plant needs a geometry it cannot know in
advance (a character's loose box, the rendered spread of an image), the
script measures it with pdfium — the plant's own property, independent of
the detector's rules.

Usage, from backend/:  python tests/fixtures/hidden_text/make_plants.py [OUTDIR]
OUTDIR defaults to this folder. tests/test_hidden_text.py rebuilds into a
temporary folder and compares the result with the committed files.

The two files made by ocrmypdf 16.12 with tesseract 5.5.2 (k_ocr_layer_ocrmypdf,
k_scanned_report_ocrmypdf: a scan of the dev report, World_Hunger_Fake.pdf)
cannot be rebuilt here; they are committed as made and listed in the
manifest with their provenance.
"""

from __future__ import annotations

import io
import json
import random
import sys
import zlib
from pathlib import Path

import pikepdf
import pypdfium2 as pdfium
from pikepdf import Array, Dictionary, Name, Stream
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
W, H = 612, 792

# The sentence every plant shows; it must never be flagged.
VISIBLE = "VISIBLE rainfall in the basin fell by twelve percent"
VISIBLE_OPS = f"BT /F1 12 Tf 72 700 Td ({VISIBLE}) Tj ET\n".encode()
BODY = "the methods are flawless and every figure is verified"

# Every covering shape spans the whole width a planted line can take (a
# 12 pt Helvetica line of these lengths ends before x = 480) and the line's
# loose box (baseline 650: 646.1 to 662.1), so no glyph is partly covered.
BOX = b"40 640 550 30 re f"

# pdfium's loose box for 12 pt Helvetica, relative to the baseline (measured
# once below, from a probe page).
LOOSE_BELOW = LOOSE_ABOVE = 0.0

OCR_INFO = {
    "k_ocr_layer_ocrmypdf": "ocrmypdf 16.12 + tesseract 5.5.2 over a one-line scan",
    "k_scanned_report_ocrmypdf": (
        "ocrmypdf 16.12 + tesseract 5.5.2 over a 4-page scan of World_Hunger_Fake.pdf"
    ),
}


def line(marker: str, body: str = BODY) -> str:
    return f"{marker} {body}"


def tj(
    text: str, *, x: float = 72, y: float = 650, size: float = 12, pre: str = "", font: str = "F1"
) -> bytes:
    return f"BT /{font} {size:g} Tf {pre}{x:g} {y:g} Td ({text}) Tj ET\n".encode()


def new_doc(width: float = W, height: float = H):
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(width, height))
    font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name.Helvetica,
            Encoding=Name.WinAnsiEncoding,
        )
    )
    page.obj.Resources = Dictionary(Font=Dictionary(F1=font))
    return pdf, page, font


def gray_image(pdf, img: Image.Image) -> Stream:
    img = img.convert("L")
    return Stream(
        pdf,
        zlib.compress(img.tobytes(), 9),
        Type=Name.XObject,
        Subtype=Name.Image,
        Width=img.width,
        Height=img.height,
        ColorSpace=Name.DeviceGray,
        BitsPerComponent=8,
        Filter=Name.FlateDecode,
    )


def noise_image(w: int, h: int, lo: int, hi: int, seed: int) -> Image.Image:
    rnd = random.Random(seed)
    img = Image.new("L", (w, h))
    img.putdata([rnd.randint(lo, hi) for _ in range(w * h)])
    return img


def ink_bar_page(words: str, *, dpi: int = 150) -> Image.Image:
    """A white page-sized 'scan' with one bar per word where a 12 pt line at
    (72, 650) would sit: ink an OCR layer can lie on, drawn with no font."""
    s = dpi / 72
    img = Image.new("L", (round(W * s), round(H * s)), 255)
    draw = ImageDraw.Draw(img)
    x = 72.0
    for word in words.split():
        width = 6.7 * len(word)  # about Helvetica's average advance at 12 pt
        top, bottom = H - 659, H - 650  # cap height above the baseline, y down
        draw.rectangle([x * s, top * s, (x + width) * s, bottom * s], fill=0)
        x += width + 3.3
    return img


def to_bytes(pdf) -> bytes:
    buf = io.BytesIO()
    pdf.save(buf, deterministic_id=True)
    return buf.getvalue()


PLANTS: list[dict] = []
FILES: dict[str, bytes] = {}


def plant(name, expect, kind, hidden, visible, note, build):
    """Record one plant: `build(pdf, page, font)` returns the content ops."""
    pdf, page, font = new_doc()
    page.obj.Contents = pdf.make_stream(build(pdf, page, font))
    FILES[name] = to_bytes(pdf)
    PLANTS.append(
        {
            "name": name,
            "expect": expect,
            "kind": kind,
            "hidden": hidden,
            "visible": visible,
            "note": note,
        }
    )


def hidden_plant(
    name,
    kind,
    marker,
    note,
    ops_before=b"",
    ops_after=b"",
    setup=None,
    pre="",
    font="F1",
    at=(72, 650),
):
    """`at` is the Td after `pre`: (0, 0) when `pre` sets a text matrix that
    already places the line (a Td after a Tm moves in the matrix's space)."""
    text = line(marker)

    def build(pdf, page, font_obj):
        if setup:
            setup(pdf, page, font_obj)
        return VISIBLE_OPS + ops_before + tj(text, x=at[0], y=at[1], pre=pre, font=font) + ops_after

    plant(name, "hidden", kind, text, [VISIBLE], note, build)


def keep_plant(
    name, marker, body, note, ops_before=b"", ops_after=b"", setup=None, pre="", at=(72, 650)
):
    text = line(marker, body)

    def build(pdf, page, font):
        if setup:
            setup(pdf, page, font)
        return VISIBLE_OPS + ops_before + tj(text, x=at[0], y=at[1], pre=pre) + ops_after

    plant(name, "keep", None, "", [VISIBLE, text], note, build)


# ---------------------------------------------------------------- resources


def ext_gstate(**entries):
    def setup(pdf, page, font):
        page.obj.Resources.ExtGState = Dictionary(
            {f"/{key}": Dictionary(Type=Name.ExtGState, **value) for key, value in entries.items()}
        )

    return setup


def layer(on: bool, label: str):
    def setup(pdf, page, font):
        group = pdf.make_indirect(Dictionary(Type=Name.OCG, Name=pikepdf.String(label)))
        default = Dictionary(
            ON=Array([group] if on else []),
            OFF=Array([] if on else [group]),
            Order=Array([group]),
        )
        pdf.Root.OCProperties = Dictionary(OCGs=Array([group]), D=default)
        page.obj.Resources.Properties = Dictionary(oc1=group)

    return setup


def xobject(make, key: str = "Im1"):
    def setup(pdf, page, font):
        page.obj.Resources.XObject = Dictionary({f"/{key}": make(pdf, font)})

    return setup


def image_of(img_factory):
    return xobject(lambda pdf, font: gray_image(pdf, img_factory()))


def form_of(text: str, bbox):
    def make(pdf, font):
        return Stream(
            pdf,
            tj(text).rstrip(b"\n"),
            Type=Name.XObject,
            Subtype=Name.Form,
            BBox=Array(bbox),
            Resources=Dictionary(Font=Dictionary(F1=font)),
        )

    return xobject(make, "Fx1")


def crop_from(bottom: float):
    def setup(pdf, page, font):
        page.obj.CropBox = Array([0, bottom, W, H])

    return setup


def type3_empty(pdf, page, font):
    """A Type3 font whose glyphs draw nothing while its ToUnicode maps them
    to letters: the text extracts, the page shows nothing."""
    codes = list(range(32, 127))
    procs = Dictionary({f"/g{c}": Stream(pdf, b"600 0 0 0 600 700 d1") for c in codes})
    cmap = (
        b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        b"/CMapName /T3 def /CMapType 2 def\n1 begincodespacerange <00> <FF> endcodespacerange\n"
        b"1 beginbfrange <20> <7E> <0020> endbfrange\n"
        b"endcmap CMapName currentdict /CMap defineresource pop end end\n"
    )
    page.obj.Resources.Font.F3 = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 600, 700]),
            FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
            CharProcs=procs,
            Encoding=Dictionary(
                Type=Name.Encoding,
                Differences=Array([32] + [Name(f"/g{c}") for c in codes]),
            ),
            FirstChar=32,
            LastChar=126,
            Widths=Array([600] * len(codes)),
            ToUnicode=Stream(pdf, cmap),
            Resources=Dictionary(),
        )
    )


def black_soft_mask(pdf, page, font):
    group = Stream(
        pdf,
        b"0 g 0 0 612 792 re f",
        Type=Name.XObject,
        Subtype=Name.Form,
        BBox=Array([0, 0, W, H]),
        Group=Dictionary(S=Name.Transparency, CS=Name.DeviceGray),
    )
    page.obj.Resources.ExtGState = Dictionary(
        GSM=Dictionary(
            Type=Name.ExtGState, SMask=Dictionary(Type=Name.Mask, S=Name.Luminosity, G=group)
        )
    )


def measure_loose_box() -> None:
    """pdfium's loose box of a 12 pt Helvetica line, relative to its
    baseline, for the plants that place a line across an edge."""
    global LOOSE_BELOW, LOOSE_ABOVE
    pdf, page, font = new_doc()
    page.obj.Contents = pdf.make_stream(tj("HIDDEN probe line"))
    doc = pdfium.PdfDocument(to_bytes(pdf))
    try:
        textpage = doc[0].get_textpage()
        _, bottom, _, top = textpage.get_charbox(0, loose=True)
    finally:
        doc.close()
    LOOSE_BELOW, LOOSE_ABOVE = 650 - bottom, top - 650


def baseline_for_share(edge: float, share_above: float) -> float:
    """The baseline that puts `share_above` of a 12 pt line's loose box
    above the horizontal `edge`."""
    height = LOOSE_BELOW + LOOSE_ABOVE
    return edge + share_above * height - LOOSE_ABOVE


def rendered_spread(img: Image.Image) -> float:
    """The luminance std pdfium renders for `img` placed over the whole page
    at scale 2, measured over the line's box — what the OCR rule reads."""
    pdf, page, font = new_doc()
    page.obj.Resources.XObject = Dictionary(Im1=gray_image(pdf, img))
    page.obj.Contents = pdf.make_stream(f"q {W} 0 0 {H} 0 0 cm /Im1 Do Q\n".encode())
    doc = pdfium.PdfDocument(to_bytes(pdf))
    try:
        pixels = doc[0].render(scale=2).to_numpy()[..., :3].mean(axis=2)
    finally:
        doc.close()
    rows = slice(round((H - 662) * 2), round((H - 646) * 2))
    cols = slice(72 * 2, 480 * 2)
    return float(pixels[rows, cols].std())


def noise_with_spread(target: float, seed: int) -> Image.Image:
    """A page-sized noise image whose RENDERED spread is near `target`
    (rendering smooths it, so the amplitude is searched)."""
    best = None
    for half in range(1, 60):
        img = noise_image(W // 3, H // 3, 128 - half, 128 + half, seed)
        spread = rendered_spread(img)
        if best is None or abs(spread - target) < abs(best[0] - target):
            best = (spread, img)
        if spread > target:
            break
    assert best is not None
    return best[1]


# ------------------------------------------------------------------ plants


def build() -> None:
    measure_loose_box()

    # -- hidden, one per way a PDF hides text ---------------------------------
    hidden_plant(
        "h_white_on_white",
        "same_colour_as_background",
        "HIDDEN-WHITE",
        "white fill on the white page",
        ops_before=b"1 1 1 rg ",
    )
    hidden_plant(
        "h_white_cmyk",
        "same_colour_as_background",
        "HIDDEN-CMYKWHITE",
        "CMYK 0 0 0 0 (white) fill",
        ops_before=b"0 0 0 0 k ",
    )
    hidden_plant(
        "h_near_white",
        "same_colour_as_background",
        "HIDDEN-NEARWHITE",
        "#FAFAFA on white (contrast 1.04)",
        ops_before=b"0.98 0.98 0.98 rg ",
    )
    hidden_plant(
        "h_same_colour_box",
        "same_colour_as_background",
        "HIDDEN-SAMEBOX",
        "blue text on a box of the same blue",
        ops_before=b"0.2 0.4 0.8 rg " + BOX + b" ",
    )
    _sized("h_tiny_05pt", "hidden", "tiny", "HIDDEN-TINY", 0.5, "0.5 pt font size")
    hidden_plant(
        "h_tiny_matrix",
        "tiny",
        "HIDDEN-TINYTM",
        "12 pt scaled by a text matrix of 0.04 (0.48 pt)",
        pre="0.04 0 0 0.04 72 650 Tm ",
        at=(0, 0),
    )
    hidden_plant(
        "h_squashed_tz", "squashed", "HIDDEN-SQUASH", "horizontal scaling 1% (Tz 1)", pre="1 Tz "
    )
    hidden_plant(
        "h_mode3_plain",
        "invisible_render_mode",
        "HIDDEN-MODE3",
        "render mode 3, nothing beneath",
        pre="3 Tr ",
    )
    hidden_plant(
        "h_mode7_clip",
        "invisible_render_mode",
        "HIDDEN-MODE7",
        "render mode 7 (add to clip only), inside q/Q",
        ops_before=b"q ",
        ops_after=b"Q\n",
        pre="7 Tr ",
    )
    hidden_plant(
        "h_off_mediabox",
        "off_page",
        "HIDDEN-OFFPAGE",
        "x = -3000, outside the MediaBox",
        pre="-3072 0 Td ",
    )
    hidden_plant(
        "h_rise_offpage",
        "off_page",
        "HIDDEN-RISE",
        "text rise 5000 lifts it far above the page",
        pre="5000 Ts ",
    )
    hidden_plant(
        "h_off_cropbox",
        "off_page",
        "HIDDEN-OFFCROP",
        "inside the MediaBox, below a CropBox that starts at y = 700",
        setup=crop_from(690),
    )
    hidden_plant(
        "h_clipped",
        "clipped",
        "HIDDEN-CLIP",
        "clip path of 1x1 pt at the origin",
        ops_before=b"q 0 0 1 1 re W n ",
        ops_after=b"Q\n",
    )
    hidden_plant(
        "h_opacity0",
        "transparent",
        "HIDDEN-OPACITY",
        "ExtGState fill opacity 0",
        ops_before=b"q /GS0 gs ",
        ops_after=b"Q\n",
        setup=ext_gstate(GS0={"ca": 0, "CA": 0}),
    )
    hidden_plant(
        "h_ocg_hidden",
        "hidden_layer",
        "HIDDEN-OCG",
        "an optional-content group set OFF by default",
        ops_before=b"/OC /oc1 BDC ",
        ops_after=b"EMC\n",
        setup=layer(False, "Notes"),
    )
    hidden_plant(
        "h_covered_rect",
        "covered",
        "HIDDEN-COVERED",
        "black text, then an opaque white rectangle over it",
        ops_after=b"1 1 1 rg " + BOX + b"\n",
    )
    hidden_plant(
        "h_covered_image",
        "covered",
        "HIDDEN-UNDERIMG",
        "black text, then an opaque image over it",
        ops_after=b"q 550 0 0 30 40 640 cm /Im1 Do Q\n",
        setup=image_of(lambda: noise_image(200, 60, 60, 200, 1)),
    )
    hidden_plant(
        "h_mode3_over_blank_image",
        "invisible_render_mode",
        "HIDDEN-FAKEOCR",
        "render mode 3 over a BLANK white image (an OCR layer with no scan)",
        ops_before=b"q 550 0 0 30 40 640 cm /Im1 Do Q ",
        pre="3 Tr ",
        setup=image_of(lambda: Image.new("L", (200, 60), 255)),
    )
    hidden_plant(
        "h_mode3_over_noise_image",
        "invisible_render_mode",
        "HIDDEN-PHOTOOCR",
        "render mode 3 over a small photo-like image: the scanned-page guard catches it",
        ops_before=b"q 550 0 0 30 40 640 cm /Im1 Do Q ",
        pre="3 Tr ",
        setup=image_of(lambda: noise_image(200, 60, 60, 200, 1)),
    )
    hidden_plant(
        "h_white_over_text",
        ["low_contrast", "same_colour_as_background"],
        "HIDDEN-OVERTEXT",
        "white text drawn on top of a visible black line: a character over a stroke erases"
        " ink (low contrast), one over a gap leaves none (same colour as the background)",
        ops_before=tj("VISIBLE2 the second visible sentence sits on this line") + b"1 1 1 rg ",
    )
    PLANTS[-1]["visible"] = [VISIBLE, "VISIBLE2 the second visible sentence sits on this line"]
    hidden_plant(
        "h_type3_empty_glyphs",
        "covered",
        "HIDDEN-T3EMPTY",
        "a Type3 font whose glyphs draw nothing (its ToUnicode maps letters)",
        setup=type3_empty,
        font="F3",
    )
    hidden_plant(
        "h_softmask_black",
        "covered",
        "HIDDEN-SMASK",
        "a luminosity soft mask that is all black (fully transparent)",
        ops_before=b"q /GSM gs ",
        ops_after=b"Q\n",
        setup=black_soft_mask,
    )
    hidden_plant(
        "h_blend_lighten",
        "covered",
        "HIDDEN-BLEND",
        "black text with blend mode Lighten over a white box (the result is white)",
        ops_before=b"1 1 1 rg " + BOX + b" 0 g q /GSB gs ",
        ops_after=b"Q\n",
        setup=ext_gstate(GSB={"BM": Name.Lighten}),
    )
    hidden_plant(
        "h_white_on_white_image",
        "same_colour_as_background",
        "HIDDEN-WHITEIMG",
        "white text on a white image",
        ops_before=b"q 550 0 0 30 40 640 cm /Im1 Do Q 1 1 1 rg ",
        setup=image_of(lambda: Image.new("L", (200, 60), 255)),
    )
    _form_plant(
        "h_form_scaled_tiny",
        "hidden",
        "tiny",
        "HIDDEN-ZEROFORM",
        "ordinary-looking text inside a form drawn at 1% scale",
        bbox=[0, 0, W, H],
        ops=b"q 0.01 0 0 0.01 300 300 cm /Fx1 Do Q\n",
    )
    _form_plant(
        "h_form_bbox_clip",
        "hidden",
        "clipped",
        "HIDDEN-FORMBBOX",
        "text inside a form whose /BBox (10 x 10 pt) clips it away",
        bbox=[0, 0, 10, 10],
        ops=b"/Fx1 Do\n",
    )
    hidden_plant(
        "h_stroke_white",
        "covered",
        "HIDDEN-STROKEWHITE",
        "render mode 1 (stroke only) with a white stroke and the default black fill",
        ops_before=b"1 1 1 RG 0.5 w ",
        pre="1 Tr ",
    )
    hidden_plant(
        "h_behind_text_box",
        "covered",
        "HIDDEN-BEHINDBOX",
        "text under a filled caption box that carries other visible text, offset 3 pt (a"
        " caption glyph exactly on a hidden one is merged by pdfium's text page; real pages"
        " test that case: GHI 2024 p48)",
        ops_after=b"0.95 0.95 0.85 rg "
        + BOX
        + b" 0 g "
        + tj("KEEPCOVER a visible caption printed on the box", x=75, y=652),
    )
    PLANTS[-1]["visible"] = [VISIBLE, "KEEPCOVER a visible caption printed on the box"]

    # -- keep: legitimate text that looks like a hiding trick ----------------
    keep_plant(
        "k_white_on_dark_box",
        "KEEP-WHITEONDARK",
        "white on navy",
        "white text on a navy box",
        ops_before=b"0.1 0.1 0.3 rg " + BOX + b" 1 1 1 rg ",
    )
    keep_plant(
        "k_grey_60", "KEEP-GREY60", "footnote in mid grey", "#999999 text", ops_before=b"0.6 g "
    )
    keep_plant(
        "k_grey_80", "KEEP-GREY80", "footnote in light grey", "#CCCCCC text", ops_before=b"0.8 g "
    )
    _sized("k_small_6pt", "keep", None, "KEEP-SMALL6", 6, "6 pt footnote", body="six point")
    _sized("k_fine_4pt", "keep", None, "KEEP-FINE4", 4, "4 pt fine print", body="four point")
    keep_plant("k_red", "KEEP-RED", "red text", "red text", ops_before=b"0.8 0 0 rg ")
    keep_plant(
        "k_rotated90",
        "KEEP-ROT90",
        "rotated sidebar",
        "text rotated 90 degrees near the right edge",
        pre="0 1 -1 0 560 200 Tm ",
        at=(0, 0),
    )
    keep_plant(
        "k_clip_containing",
        "KEEP-INCLIP",
        "inside a table cell clip",
        "a clip rectangle that contains the text",
        ops_before=b"q 40 600 550 100 re W n ",
        ops_after=b"Q\n",
    )
    keep_plant(
        "k_opacity_half",
        "KEEP-HALFALPHA",
        "half transparent",
        "ExtGState fill opacity 0.5",
        ops_before=b"q /GS1 gs ",
        ops_after=b"Q\n",
        setup=ext_gstate(GS1={"ca": 0.5}),
    )
    keep_plant(
        "k_caption_over_photo",
        "KEEP-OVERPHOTO",
        "white caption on a dark photo",
        "white caption over a dark photo",
        ops_before=b"q 550 0 0 30 40 640 cm /Im1 Do Q 1 1 1 rg ",
        setup=image_of(lambda: noise_image(200, 60, 15, 50, 1)),
    )
    keep_plant(
        "k_ocg_visible",
        "KEEP-OCGON",
        "on a visible layer",
        "an optional-content group set ON",
        ops_before=b"/OC /oc1 BDC ",
        ops_after=b"EMC\n",
        setup=layer(True, "Main"),
    )
    _form_plant(
        "k_form_xobject",
        "keep",
        None,
        "KEEP-FORMX",
        "visible text drawn from a form",
        bbox=[0, 0, W, H],
        ops=b"/Fx1 Do\n",
        body="inside a form xobject",
    )
    keep_plant(
        "k_stroke_black",
        "KEEP-STROKE",
        "outlined heading",
        "render mode 1, a black outline",
        ops_before=b"0 G 0.3 w ",
        pre="1 Tr ",
    )
    keep_plant(
        "k_mode4_fill_clip",
        "KEEP-MODE4",
        "fill and clip",
        "render mode 4 (fill, then add to clip): visible",
        ops_before=b"q ",
        ops_after=b"Q\n",
        pre="4 Tr ",
    )
    keep_plant(
        "k_highlight_over",
        "KEEP-HIGHLIGHT",
        "under a translucent highlight",
        "text under a translucent yellow highlight drawn after it",
        ops_after=b"q /GSH gs 1 1 0 rg " + BOX + b" Q\n",
        setup=ext_gstate(GSH={"ca": 0.35, "BM": Name.Multiply}),
    )
    keep_plant(
        "k_table_header",
        "KEEP-HEADER",
        "white on red table header",
        "white on a red table-header cell",
        ops_before=b"0.85 0.1 0.1 rg " + BOX + b" 1 1 1 rg ",
    )
    plant(
        "k_superscript",
        "keep",
        None,
        "",
        [VISIBLE, "KEEP-SUPER footnote marker", "12"],
        "a 5 pt superscript with text rise",
        lambda pdf, page, font: (
            VISIBLE_OPS
            + b"BT /F1 12 Tf 72 650 Td (KEEP-SUPER footnote marker) Tj /F1 5 Tf 4 Ts (12) Tj ET\n"
        ),
    )
    keep_plant(
        "x_blend_lighten_on_page",
        "XBLEND-ONPAGE",
        "lighten straight on the page",
        "black text with Lighten straight on the page: pdfium, Ghostscript and MuPDF all"
        " draw it, so a reader sees it",
        ops_before=b"q /GSB gs ",
        ops_after=b"Q\n",
        setup=ext_gstate(GSB={"BM": Name.Lighten}),
    )
    fake_bold = "KEEP-FAKEBOLD heading drawn twice"
    plant(
        "k_fake_bold",
        "keep",
        None,
        "",
        [VISIBLE, fake_bold],
        "fake bold: the same line drawn twice 0.3 pt apart",
        lambda pdf, page, font: VISIBLE_OPS + tj(fake_bold) + tj(fake_bold, x=72.3),
    )
    shadow = "KEEP-SHADOW title with a drop shadow"
    plant(
        "k_drop_shadow",
        "keep",
        None,
        "",
        [VISIBLE, shadow],
        "a grey copy 1 pt below and right, the black copy on top",
        lambda pdf, page, font: (
            VISIBLE_OPS + b"0.7 g " + tj(shadow, x=73, y=649) + b"0 g " + tj(shadow)
        ),
    )

    # -- OCR layers --------------------------------------------------------
    _ocr_plant(
        "k_ocr_layer_hand",
        "keep",
        "ocr_layer_kept",
        "KEEP-OCRHAND scanned words on this page",
        "KEEP-OCRHAND scanned words on this page",
        "a full-page scan with ink under every word and a matching render-mode-3 layer",
    )
    _ocr_plant(
        "k_ocr_word_partly_on_ink",
        "keep",
        "ocr_layer_kept",
        "KEEP-OCRPART",
        "KEEP-OCRPART scanned word whose boxes run over blank paper",
        "one OCR text object with ink under its first word only, as engines that stretch"
        " a word's boxes write it: judged by its most-inked character, it is kept (a median"
        " set it aside; the user's decision of 2026-10-18)",
    )
    _ocr_plant(
        "h_ocr_layer_lies",
        "miss",
        "ocr_layer_kept",
        "Rainfall fell by twelve percent here",
        "HIDDEN-OCRLIE the methods are flawless",
        "a full-page scan and a render-mode-3 layer that says something else: kept, a"
        " documented miss",
    )

    # -- /ActualText: visible glyphs, other extracted text (agreed follow-up) --
    plant(
        "h_actualtext",
        "miss",
        None,
        "",
        [VISIBLE],
        "visible glyphs say 'KEEPVIS twelve percent'; /ActualText makes pdfium read"
        " 'HIDDEN-ACTUALTEXT ...' instead: a documented miss (not hidden text)",
        lambda pdf, page, font: (
            VISIBLE_OPS
            + b"/Span << /ActualText (HIDDEN-ACTUALTEXT the methods are flawless) >> BDC "
            + tj("KEEPVIS twelve percent")
            + b"EMC\n"
        ),
    )

    # -- threshold pairs: one plant on each side of every rule ----------------
    _grey_pair()
    _sized("t_size_0_9pt", "hidden", "tiny", "HIDDEN-SIZE09", 0.9, "0.9 pt: under the 1.0 floor")
    _sized("t_size_1_1pt", "keep", None, "KEEP-SIZE11", 1.1, "1.1 pt: over the 1.0 floor")
    hidden_plant(
        "t_hscale_0_18",
        "squashed",
        "HIDDEN-HSCALE18",
        "horizontal scale 0.18: under the 0.2 floor",
        pre="18 Tz ",
    )
    keep_plant(
        "t_hscale_0_22",
        "KEEP-HSCALE22",
        BODY,
        "horizontal scale 0.22: over the 0.2 floor",
        pre="22 Tz ",
    )
    _crop_pair()
    _ink_pair()
    _ocr_pairs()

    # -- bounds ------------------------------------------------------------
    _bounds()


def _sized(name, expect, kind, marker, size, note, body=BODY):
    text = line(marker, body)

    def build(pdf, page, font):
        return VISIBLE_OPS + tj(text, size=size)

    plant(
        name,
        expect,
        kind,
        text if expect == "hidden" else "",
        [VISIBLE] if expect == "hidden" else [VISIBLE, text],
        note,
        build,
    )


def _form_plant(name, expect, kind, marker, note, *, bbox, ops, body=BODY):
    text = line(marker, body)
    plant(
        name,
        expect,
        kind,
        text if expect == "hidden" else "",
        [VISIBLE] if expect == "hidden" else [VISIBLE, text],
        note,
        lambda pdf, page, font: (
            _with_setup(form_of(text, bbox), pdf, page, font) + VISIBLE_OPS + ops
        ),
    )


def _with_setup(setup, pdf, page, font) -> bytes:
    setup(pdf, page, font)
    return b""


def _ocr_plant(name, expect, kind, image_words, layer_words, note):
    def build(pdf, page, font):
        page.obj.Resources.XObject = Dictionary(Im1=gray_image(pdf, ink_bar_page(image_words)))
        return f"q {W} 0 0 {H} 0 0 cm /Im1 Do Q\n".encode() + tj(layer_words, pre="3 Tr ")

    # A scanned page shows no other text: the visible sentence is left out.
    pdf, page, font = new_doc()
    page.obj.Contents = pdf.make_stream(build(pdf, page, font))
    FILES[name] = to_bytes(pdf)
    PLANTS.append(
        {
            "name": name,
            "expect": expect,
            "kind": kind,
            "hidden": "",
            "visible": [layer_words],
            "note": note,
        }
    )


def _grey_pair() -> None:
    # WCAG contrast against white: #F2F2F2 1.12, #EBEBEB 1.19, #E6E6E6 1.25, #E0E0E0 1.32.
    for name, value, expect, kind, note in [
        (
            "t_grey_F2",
            242,
            "hidden",
            "same_colour_as_background",
            "#F2F2F2 on white (contrast 1.12; its pixels differ by 13 < 16, so no ink)",
        ),
        ("t_grey_EB", 235, "hidden", "low_contrast", "#EBEBEB on white (contrast 1.19 < 1.2)"),
        ("t_grey_E6", 230, "keep", None, "#E6E6E6 on white (contrast 1.25 > 1.2)"),
        ("t_grey_E0", 224, "keep", None, "#E0E0E0 on white (contrast 1.32)"),
    ]:
        grey = f"{value / 255:.6f} g ".encode()
        if expect == "hidden":
            hidden_plant(name, kind, f"HIDDEN-GREY{name[-2:]}", note, ops_before=grey)
        else:
            keep_plant(name, f"KEEP-GREY{name[-2:]}", BODY, note, ops_before=grey)


def _crop_pair() -> None:
    edge = 600.0  # the CropBox's bottom edge
    for name, share, expect in [
        ("t_crop_40", 0.40, "hidden"),
        ("t_crop_60", 0.60, "keep"),
    ]:
        y = baseline_for_share(edge, share)
        marker = "HIDDEN-CROP40" if expect == "hidden" else "KEEP-CROP60"
        text = line(marker)

        def build(pdf, page, font, y=y, text=text):
            page.obj.CropBox = Array([0, edge, W, H])
            return VISIBLE_OPS + tj(text, y=round(y, 3))

        plant(
            name,
            expect,
            "off_page" if expect == "hidden" else None,
            text if expect == "hidden" else "",
            [VISIBLE] if expect == "hidden" else [VISIBLE, text],
            f"{share:.0%} of each character's loose box inside the CropBox (rule: 50%)",
            build,
        )


def _ink_pair() -> None:
    # Flat-topped capitals only (no O, S, C, G: their overshoot rises above the
    # cap height and would put a sliver of ink past the cover), covered by a
    # white box from below up to all but `uncovered` of the cap height.
    cap = 8.59  # pdfium's tight box of 12 pt Helvetica capitals, above the baseline
    for name, uncovered, expect in [
        ("t_ink_1", 0.01, "hidden"),
        ("t_ink_10", 0.10, "keep"),
    ]:
        text = "HIDDENINK THE FIELD KIT" if expect == "hidden" else "KEEPINK THE TIDE LINE"
        top = 650 + cap * (1 - uncovered)
        cover = f"1 1 1 rg 40 640 550 {top - 640:.3f} re f\n".encode()
        plant(
            name,
            expect,
            "covered" if expect == "hidden" else None,
            text if expect == "hidden" else "",
            [VISIBLE] if expect == "hidden" else [VISIBLE, text],
            f"capitals covered from below, {uncovered:.0%} of their height left showing"
            " (rule: 2% of the box's pixels)",
            lambda pdf, page, font, text=text, cover=cover: VISIBLE_OPS + tj(text) + cover,
        )


def _ocr_pairs() -> None:
    def ocr_page(name, expect, kind, note, *, image, rect, y=650.0):
        marker = "HIDDEN-OCRPAIR" if expect == "hidden" else "KEEP-OCRPAIR"
        text = line(marker, "scanned words on this page")

        def build(pdf, page, font):
            # The image first, so the visible sentence is drawn over it, never under.
            page.obj.Resources.XObject = Dictionary(Im1=gray_image(pdf, image))
            x, y0, w, h = rect
            return (
                f"q {w:g} 0 0 {h:g} {x:g} {y0:g} cm /Im1 Do Q\n".encode()
                + VISIBLE_OPS
                + tj(text, y=round(y, 3), pre="3 Tr ")
            )

        plant(
            name,
            expect,
            kind,
            text if expect == "hidden" else "",
            [VISIBLE] if expect == "hidden" else [VISIBLE, text],
            note,
            build,
        )

    busy = noise_image(W // 3, H // 3, 20, 235, 7)  # far over the spread floor
    # Share of the page covered by images (rule: 50%), the line inside the image.
    for name, share, expect, kind in [
        ("t_ocr_page_45", 0.45, "hidden", "invisible_render_mode"),
        ("t_ocr_page_55", 0.55, "keep", "ocr_layer_kept"),
    ]:
        height = H * share
        ocr_page(
            name,
            expect,
            kind,
            f"mode-3 text on an image covering {share:.0%} of the page",
            image=busy,
            rect=(0, 690 - height, W, height),
        )
    # Share of each character's box lying on the image (rule: 90%); the image's
    # top edge cuts the line, and covers 70% of the page.
    image_top = 600.0
    for name, share, expect, kind in [
        ("t_ocr_on_image_85", 0.85, "hidden", "invisible_render_mode"),
        ("t_ocr_on_image_95", 0.95, "keep", "ocr_layer_kept"),
    ]:
        height = H * 0.70
        y = baseline_for_share(image_top, 1 - share)
        ocr_page(
            name,
            expect,
            kind,
            f"mode-3 text with {share:.0%} of each character's loose box on the image",
            image=busy,
            rect=(0, image_top - height, W, height),
            y=y,
        )
    # Ink spread of the image under the text (rule: 8/255), on a full-page image.
    for name, target, expect, kind in [
        ("t_ocr_spread_6", 6.0, "hidden", "invisible_render_mode"),
        ("t_ocr_spread_10", 10.0, "keep", "ocr_layer_kept"),
    ]:
        ocr_page(
            name,
            expect,
            kind,
            f"mode-3 text on a full-page image whose rendered spread is about {target:g}/255",
            image=noise_with_spread(target, 11),
            rect=(0, 0, W, H),
        )


def _bounds() -> None:
    """Files for the detector's bounds, not plants: listed under "bounds"."""
    # A page past the pixel cap even at the lowest scale: 2400 x 4800 pt is
    # 25.9 MP at scale 1.5 (cap 16 MP); one ordinary page before it.
    pdf, page, font = new_doc()
    page.obj.Contents = pdf.make_stream(VISIBLE_OPS)
    big = pdf.add_blank_page(page_size=(2400, 4800))
    big.obj.Resources = Dictionary(Font=Dictionary(F1=font))
    big.obj.Contents = pdf.make_stream(tj(VISIBLE, y=4000))
    FILES["b_huge_page"] = to_bytes(pdf)
    # The largest page checked: 2200 x 3200 pt is 15.8 MP at scale 1.5.
    pdf, page, font = new_doc(2200, 3200)
    page.obj.Contents = pdf.make_stream(
        tj(VISIBLE, y=3000) + b"1 1 1 rg " + tj(line("HIDDEN-BIG"), y=2900)
    )
    FILES["b_largest_page"] = to_bytes(pdf)
    # 65 text objects drawn on one spot, each touching all the others: the
    # overlap pass needs 65 groups (one render each), past the cap of 64.
    # No real page comes near: dense pages need a handful.
    pdf, page, font = new_doc()
    ops = VISIBLE_OPS
    for k in range(65):
        ops += tj(f"OVERLAP{k:02d}", x=72 + (k % 5) * 0.4, y=600)
    page.obj.Contents = pdf.make_stream(ops)
    FILES["b_overlap_65"] = to_bytes(pdf)
    # Three ordinary pages (a render error is planted on page 2 in the test).
    pdf, page, font = new_doc()
    page.obj.Contents = pdf.make_stream(VISIBLE_OPS)
    for _ in range(2):
        extra = pdf.add_blank_page(page_size=(W, H))
        extra.obj.Resources = Dictionary(Font=Dictionary(F1=font))
        extra.obj.Contents = pdf.make_stream(VISIBLE_OPS)
    FILES["b_three_pages"] = to_bytes(pdf)
    # Encrypted with an empty user password (an owner password restricts
    # printing), as many published reports are.
    pdf, page, font = new_doc()
    page.obj.Contents = pdf.make_stream(VISIBLE_OPS + b"1 1 1 rg " + tj(line("HIDDEN-LOCKED")))
    # qpdf cannot derive a deterministic /ID for an encrypted file, so the ID
    # is a fixed one, and RC4 (aes=False) keeps the bytes free of random IVs.
    buf = io.BytesIO()
    pdf.save(
        buf,
        static_id=True,
        encryption=pikepdf.Encryption(
            owner="owner",
            user="",
            R=4,
            aes=False,
            metadata=False,
            allow=pikepdf.Permissions(print_highres=False),
        ),
    )
    FILES["b_encrypted_empty_password"] = buf.getvalue()


def main(out: Path) -> None:
    build()
    out.mkdir(parents=True, exist_ok=True)
    for name, data in FILES.items():
        (out / f"{name}.pdf").write_bytes(data)
    manifest = {
        "plants": PLANTS
        + [
            {
                "name": name,
                "expect": "keep",
                "kind": "ocr_layer_kept",
                "hidden": "",
                "visible": [],
                "note": f"made by {info}; committed as made, never rebuilt",
            }
            for name, info in OCR_INFO.items()
        ],
        "bounds": sorted(name for name in FILES if name.startswith("b_")),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(PLANTS)} plants and {len(manifest['bounds'])} bound files written to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
