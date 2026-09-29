import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFile

import authorai.ingest as ingest_mod
from authorai import db as dbmod
from authorai.embeddings import FakeEmbedder
from authorai.ingest import (
    MODEL_IMAGE_MAX_EDGE,
    MODEL_IMAGE_MAX_PIXELS,
    ParsedDocument,
    ParsedFigure,
    ParsedSection,
    ParsedTable,
    encode_image,
    image_suffix,
    ingest_image,
    ingest_pdf,
    model_copy,
    parse_image,
)
from authorai.search import keyword_search
from tests.conftest import DIM


def _parsed_document() -> ParsedDocument:
    return ParsedDocument(
        title="Hunger Report",
        sections=[
            ParsedSection(
                title="Overview", page=1, text="Global hunger affected 735 million people in 2023."
            ),
            ParsedSection(
                title="Methods", page=2, text="Data was collected from national surveys."
            ),
        ],
        tables=[
            ParsedTable(
                page=3,
                markdown="| year | undernourished |\n| 2023 | 735 million |",
                caption="Hunger by year",
            )
        ],
        figures=[
            ParsedFigure(
                page=4,
                image=Image.new("RGB", (10, 10), "red"),
                caption="Trend of undernourishment worldwide",
            )
        ],
    )


def test_ingest_pdf_writes_document_chunks_and_figures(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_mod, "parse_pdf", lambda path: _parsed_document())
    run_id = dbmod.create_run(conn)

    doc_id = ingest_pdf(
        conn,
        FakeEmbedder(dim=DIM),
        run_id,
        tmp_path / "fake.pdf",
        kind="SOURCE",
        figures_dir=tmp_path / "figures",
    )

    document = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    assert document["title"] == "Hunger Report"
    assert document["run_id"] == run_id
    # The dedup donor filter matches on this stamp — a fresh ingest that
    # forgot it would produce documents that can never donate.
    assert document["embedding_model"] == "fake-embedder"

    kinds = dict(
        conn.execute(
            "SELECT kind, count(*) FROM chunks WHERE doc_id = ? GROUP BY kind", (doc_id,)
        ).fetchall()
    )
    assert kinds == {"text": 2, "table": 1, "figure": 1}

    figure = conn.execute("SELECT * FROM figures WHERE doc_id = ?", (doc_id,)).fetchone()
    assert figure["caption"] == "Trend of undernourishment worldwide"
    assert Path(figure["image_path"]).exists()

    figure_chunk = conn.execute("SELECT * FROM chunks WHERE kind = 'figure'").fetchone()
    assert figure_chunk["figure_id"] == figure["id"]

    # Every content type is reachable through search.
    assert keyword_search(conn, run_id, "undernourishment") != []
    assert keyword_search(conn, run_id, "735") != []
    assert keyword_search(conn, run_id, "surveys") != []


def test_ingest_stores_sections_and_figure_descriptions(conn, tmp_path, monkeypatch):
    import json

    from tests.conftest import FakeLLM

    monkeypatch.setattr(ingest_mod, "parse_pdf", lambda path: _parsed_document())
    run_id = dbmod.create_run(conn)
    llm = FakeLLM(image_description="A line chart showing undernourishment rising to 735 million.")

    doc_id = ingest_pdf(
        conn,
        FakeEmbedder(dim=DIM),
        run_id,
        tmp_path / "fake.pdf",
        kind="SOURCE",
        figures_dir=tmp_path / "figures",
        describe=lambda image: llm.describe_image(
            model="claude-haiku-4-5", image=image, prompt="describe"
        ),
    )

    # Sections persisted for later claim extraction.
    document = conn.execute("SELECT metadata FROM documents WHERE id = ?", (doc_id,)).fetchone()
    sections = json.loads(document["metadata"])["sections"]
    assert [section["title"] for section in sections] == ["Overview", "Methods"]

    # Figure description stored AND baked into the (immutable) chunk text.
    assert llm.image_calls == 1
    figure = conn.execute("SELECT * FROM figures WHERE doc_id = ?", (doc_id,)).fetchone()
    assert "line chart" in figure["description"]
    figure_chunk = conn.execute("SELECT text FROM chunks WHERE kind = 'figure'").fetchone()
    assert "Trend of undernourishment worldwide" in figure_chunk["text"]
    assert "line chart" in figure_chunk["text"]


def test_table_text_is_capped(conn, tmp_path, monkeypatch):
    huge = ParsedDocument(
        title=None,
        sections=[],
        tables=[ParsedTable(page=1, markdown="| cell |" * 3000, caption="Big table")],
        figures=[],
    )
    monkeypatch.setattr(ingest_mod, "parse_pdf", lambda path: huge)
    run_id = dbmod.create_run(conn)
    ingest_pdf(
        conn, FakeEmbedder(dim=DIM), run_id, tmp_path / "t.pdf", kind="SOURCE", figures_dir=tmp_path
    )
    text = conn.execute("SELECT text FROM chunks WHERE kind = 'table'").fetchone()["text"]
    assert len(text) <= ingest_mod.TABLE_TEXT_CAP + len("\n[table truncated]")
    assert text.endswith("[table truncated]")


def test_ingest_records_upload_and_absolute_image_path(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_mod, "parse_pdf", lambda path: _parsed_document())
    run_id = dbmod.create_run(conn)
    doc_id = ingest_pdf(
        conn,
        FakeEmbedder(dim=DIM),
        run_id,
        tmp_path / "fake.pdf",
        kind="SOURCE",
        figures_dir=tmp_path / "figures",
    )
    document = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    upload = conn.execute("SELECT * FROM uploads WHERE id = ?", (document["upload_id"],)).fetchone()
    assert upload is not None
    assert upload["file_name"] == "fake.pdf"
    figure = conn.execute("SELECT * FROM figures WHERE doc_id = ?", (doc_id,)).fetchone()
    assert Path(figure["image_path"]).is_absolute()


class _FailingEmbedder:
    dim = DIM

    def embed(self, texts):
        raise RuntimeError("embedding provider unavailable")


def test_ingest_failure_leaves_no_partial_state(conn, tmp_path, monkeypatch):
    # A failure in the embedding network call must not leave orphan
    # document/figure rows or PNG files behind.
    monkeypatch.setattr(ingest_mod, "parse_pdf", lambda path: _parsed_document())
    run_id = dbmod.create_run(conn)
    figures_dir = tmp_path / "figures"
    with pytest.raises(RuntimeError, match="embedding provider"):
        ingest_pdf(
            conn,
            _FailingEmbedder(),
            run_id,
            tmp_path / "fake.pdf",
            kind="SOURCE",
            figures_dir=figures_dir,
        )
    for table in ("documents", "figures", "chunks", "uploads"):
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    assert not figures_dir.exists()


def test_ingest_empty_document_fails_loudly(conn, tmp_path, monkeypatch):
    empty = ParsedDocument(title=None, sections=[], tables=[], figures=[])
    monkeypatch.setattr(ingest_mod, "parse_pdf", lambda path: empty)
    run_id = dbmod.create_run(conn)
    with pytest.raises(ValueError, match="No extractable content"):
        ingest_pdf(
            conn,
            FakeEmbedder(dim=DIM),
            run_id,
            tmp_path / "empty.pdf",
            kind="SOURCE",
            figures_dir=tmp_path,
        )


def test_ingest_chunk_ids_follow_text_then_tables_then_figures(conn, tmp_path, monkeypatch):
    # Chunk ids ARE the retrieval tie-break order and the eval's byte-identity
    # rests on them — pinned (order AND text) before ingest_pdf is split, so
    # the refactor cannot silently reorder or reword text/table/figure chunks.
    monkeypatch.setattr(ingest_mod, "parse_pdf", lambda path: _parsed_document())
    run_id = dbmod.create_run(conn)
    doc_id = ingest_pdf(
        conn,
        FakeEmbedder(dim=DIM),
        run_id,
        tmp_path / "fake.pdf",
        kind="SOURCE",
        figures_dir=tmp_path / "figures",
    )
    rows = conn.execute(
        "SELECT kind, section, page, text FROM chunks WHERE doc_id = ? ORDER BY id", (doc_id,)
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("text", "Overview", 1, "Global hunger affected 735 million people in 2023."),
        ("text", "Methods", 2, "Data was collected from national surveys."),
        ("table", None, 3, "Hunger by year\n\n| year | undernourished |\n| 2023 | 735 million |"),
        ("figure", None, 4, "Trend of undernourishment worldwide"),
    ]


# --- snapshots (URL sources) and the shared ingest_parsed path -------------

WEB_PROVENANCE = {
    "url": "https://www.who.int/news-room/fact-sheets/detail/drinking-water",
    "final_url": "https://www.who.int/news-room/fact-sheets/detail/drinking-water",
    "fetched_at": "2026-09-10T10:00:00+00:00",
    "content_type": "text/html",
    "title": "Drinking-water",
    "authors": [],
    "publisher": "World Health Organization",
    "publication_date": "2026-03-01",
    "doi": None,
    "scholarly": False,
}


def _web_document() -> ParsedDocument:
    return ParsedDocument(
        title="Drinking-water",
        sections=[
            ParsedSection(
                title="Drinking-water: key facts",
                page=None,
                text="In 2022, 2.2 billion people still lacked safely managed drinking water.",
            ),
            ParsedSection(
                title="Access to services",
                page=None,
                text="Safely managed services are free from contamination.\n\n"
                "| Service level | Population |\n|---|---|\n| Safely managed | 73% |",
            ),
        ],
        tables=[],
        figures=[],
    )


def test_snapshot_round_trips_sections_and_provenance(tmp_path):
    import json

    path = tmp_path / "u.json"
    ingest_mod.write_snapshot(path, _web_document(), WEB_PROVENANCE)
    parsed, provenance = ingest_mod.load_snapshot(path)
    assert parsed == _web_document()
    assert provenance == WEB_PROVENANCE
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == ingest_mod.SNAPSHOT_SCHEMA == 1
    assert not path.with_name(path.name + ".part").exists()


def test_snapshot_bytes_do_not_depend_on_key_order(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    ingest_mod.write_snapshot(a, _web_document(), WEB_PROVENANCE)
    ingest_mod.write_snapshot(b, _web_document(), dict(reversed(list(WEB_PROVENANCE.items()))))
    assert a.read_bytes() == b.read_bytes()


def test_snapshot_keeps_time_locators_and_non_ascii_text(tmp_path):
    path = tmp_path / "v.json"
    video = ParsedDocument(
        title="Charla",
        sections=[
            ParsedSection(
                title="",
                page=None,
                text="La Organización Mundial de la Salud informó.",
                start_seconds=75.0,
                end_seconds=150.0,
            )
        ],
        tables=[],
        figures=[],
    )
    ingest_mod.write_snapshot(path, video, {"url": "https://example.org/v"})
    parsed, _ = ingest_mod.load_snapshot(path)
    assert (parsed.sections[0].start_seconds, parsed.sections[0].end_seconds) == (75.0, 150.0)
    assert "Organización" in path.read_text(encoding="utf-8")  # stored readable, not \\u-escaped


@pytest.mark.parametrize(
    "content",
    [
        '{"schema": 2, "document": {"title": null, "sections": []}, "provenance": {}}',
        '{"document": {"title": null, "sections": []}, "provenance": {}}',
        '{"schema": 1, "provenance": {}}',
        "not json at all",
    ],
)
def test_load_snapshot_refuses_unknown_or_malformed_content(tmp_path, content):
    path = tmp_path / "s.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot"):
        ingest_mod.load_snapshot(path)


def test_failed_snapshot_write_leaves_neither_file_behind(tmp_path, monkeypatch):
    path = tmp_path / "u.json"

    def refuse(*args):
        raise OSError("disk full")

    monkeypatch.setattr(ingest_mod.os, "replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        ingest_mod.write_snapshot(path, _web_document(), WEB_PROVENANCE)
    assert not path.exists()
    assert not path.with_name(path.name + ".part").exists()


def test_ingest_parsed_carries_time_locators_into_chunks(conn, tmp_path):
    talk = ParsedDocument(
        title="Talk",
        sections=[
            ParsedSection(
                title="", page=None, text="first window text", start_seconds=0.0, end_seconds=75.0
            ),
            ParsedSection(
                title="",
                page=None,
                text="second window text",
                start_seconds=75.0,
                end_seconds=150.0,
            ),
        ],
        tables=[],
        figures=[],
    )
    run_id = dbmod.create_run(conn)
    upload_id = dbmod.add_upload(
        conn, "SOURCE", "talk", str(tmp_path / "t.json"), source_type="youtube"
    )
    doc_id = ingest_mod.ingest_parsed(
        conn,
        FakeEmbedder(dim=DIM),
        run_id,
        talk,
        path=tmp_path / "t.json",
        kind="SOURCE",
        figures_dir=tmp_path / "figures",
        upload_id=upload_id,
    )
    rows = conn.execute(
        "SELECT text, page, start_seconds, end_seconds FROM chunks WHERE doc_id = ? ORDER BY id",
        (doc_id,),
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        ("first window text", None, 0.0, 75.0),
        ("second window text", None, 75.0, 150.0),
    ]


def test_ingest_snapshot_stores_sections_and_provenance(conn, tmp_path):
    import json

    path = tmp_path / "u.json"
    ingest_mod.write_snapshot(path, _web_document(), WEB_PROVENANCE)
    run_id = dbmod.create_run(conn)
    upload_id = dbmod.add_upload(
        conn, "SOURCE", "www.who.int", str(path), source_type="web", url=WEB_PROVENANCE["url"]
    )
    doc_id = ingest_mod.ingest_snapshot(
        conn,
        FakeEmbedder(dim=DIM),
        run_id,
        path,
        kind="SOURCE",
        figures_dir=tmp_path / "figures",
        upload_id=upload_id,
        fallback_title="www.who.int",
    )
    document = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    metadata = json.loads(document["metadata"])
    assert metadata["provenance"] == WEB_PROVENANCE
    assert [s["title"] for s in metadata["sections"]] == [
        "Drinking-water: key facts",
        "Access to services",
    ]
    assert (document["title"], document["upload_id"]) == ("Drinking-water", upload_id)
    sections = [
        r["section"]
        for r in conn.execute("SELECT section FROM chunks WHERE doc_id = ? ORDER BY id", (doc_id,))
    ]
    assert sections == ["Drinking-water: key facts", "Access to services"]
    assert keyword_search(conn, run_id, "contamination") != []


def test_ingest_snapshot_refuses_an_empty_page_and_writes_nothing(conn, tmp_path):
    path = tmp_path / "e.json"
    empty = ParsedDocument(title=None, sections=[], tables=[], figures=[])
    ingest_mod.write_snapshot(path, empty, {"url": "https://example.org/e"})
    run_id = dbmod.create_run(conn)
    upload_id = dbmod.add_upload(conn, "SOURCE", "example.org", str(path), source_type="web")
    with pytest.raises(ValueError, match="No extractable content"):
        ingest_mod.ingest_snapshot(
            conn,
            FakeEmbedder(dim=DIM),
            run_id,
            path,
            kind="SOURCE",
            figures_dir=tmp_path / "figures",
            upload_id=upload_id,
            fallback_title="example.org",
        )
    assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 0


def test_pdf_document_metadata_keeps_its_exact_shape(conn, tmp_path, monkeypatch):
    """Timestamps are a transcript concept: a PDF's stored sections must stay
    exactly {title, page, text}, so new PDF ingests and the dedup copies of
    old ones describe the same shape."""
    import json

    monkeypatch.setattr(ingest_mod, "parse_pdf", lambda path: _parsed_document())
    run_id = dbmod.create_run(conn)
    doc_id = ingest_pdf(
        conn,
        FakeEmbedder(dim=DIM),
        run_id,
        tmp_path / "fake.pdf",
        kind="SOURCE",
        figures_dir=tmp_path / "figures",
    )
    stored = conn.execute("SELECT metadata FROM documents WHERE id = ?", (doc_id,)).fetchone()[0]
    assert json.loads(stored) == {
        "sections": [
            {
                "title": "Overview",
                "page": 1,
                "text": "Global hunger affected 735 million people in 2023.",
            },
            {"title": "Methods", "page": 2, "text": "Data was collected from national surveys."},
        ]
    }


# ---- Images as sources -------------------------------------------------------

# A real bar chart Docling exported from the GHI 2025 report in example_sources
# (page 19, values printed on the bars): parser output, not a drawn rectangle.
REAL_CHART = Path(__file__).parent / "fixtures" / "images" / "stunting_chart.png"


def _encoded(image, fmt, **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    return buffer.getvalue()


def _two_frames(fmt) -> bytes:
    first, second = Image.new("RGB", (20, 20), "red"), Image.new("RGB", (20, 20), "blue")
    return _encoded(first, fmt, save_all=True, append_images=[second])


def _photo_like(size=(1600, 1200)):
    noise = Image.effect_noise(size, 40).convert("RGB")
    return Image.blend(noise, Image.linear_gradient("L").resize(size).convert("RGB"), 0.6)


@pytest.fixture(scope="module")
def bomb_png() -> bytes:
    """72 megapixels in ~70 KB: the per-file byte cap alone cannot stop it."""
    return _encoded(Image.new("L", (12000, 6000)), "PNG")


@pytest.fixture()
def no_decoding(monkeypatch):
    """Any pixel decode is a test failure: the gate must refuse from the header."""
    monkeypatch.setattr(
        ImageFile.ImageFile, "load", lambda self: pytest.fail("the gate decoded pixels")
    )


@pytest.mark.parametrize(
    ("data", "suffix"),
    [
        (_encoded(Image.new("RGB", (8, 8)), "PNG"), ".png"),
        (_encoded(Image.new("RGB", (8, 8)), "JPEG"), ".jpg"),
        (_encoded(Image.new("RGB", (8, 8)), "WEBP"), ".webp"),
        # A phone photo carrying a depth map or a preview is a multi-picture
        # JPEG: Pillow reports MPO with several frames — still one photo.
        (_two_frames("MPO"), ".jpg"),
    ],
    ids=["png", "jpeg", "webp", "phone-mpo"],
)
def test_an_image_source_is_stored_under_the_suffix_its_bytes_prove(data, suffix, no_decoding):
    assert image_suffix(io.BytesIO(data)) == suffix


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>', "not a PNG"),
        (_encoded(Image.new("RGB", (8, 8)), "GIF"), "not a PNG"),
        (b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n", "not a PNG"),
        (b"", "not a PNG"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 16, "not a PNG"),
        (_two_frames("PNG"), "animated"),
        (_two_frames("WEBP"), "animated"),
    ],
    ids=["svg", "gif", "pdf-bytes", "empty", "png-magic-no-header", "apng", "animated-webp"],
)
def test_anything_but_a_still_png_jpeg_or_webp_is_refused(data, reason):
    handle = io.BytesIO(data)
    with pytest.raises(ValueError, match=reason):
        image_suffix(handle)
    handle.seek(0)  # the upload's own handle is still open for its caller


def test_a_bomb_refusal_leaves_the_callers_handle_open(bomb_png):
    handle = io.BytesIO(bomb_png)
    with pytest.raises(ValueError, match="pixels"):
        image_suffix(handle)
    handle.seek(0)


def test_a_decompression_bomb_is_refused_from_its_header_alone(bomb_png, no_decoding):
    with pytest.raises(ValueError, match=r"12000×6000 pixels; images up to 50,000,000"):
        image_suffix(io.BytesIO(bomb_png))


def test_the_pixel_limit_holds_on_both_sides_of_its_boundary(monkeypatch, no_decoding):
    monkeypatch.setattr(ingest_mod, "IMAGE_MAX_PIXELS", 100)
    assert image_suffix(io.BytesIO(_encoded(Image.new("L", (10, 10)), "PNG"))) == ".png"
    with pytest.raises(ValueError, match="images up to 100 pixels"):
        image_suffix(io.BytesIO(_encoded(Image.new("L", (10, 11)), "PNG")))


def test_pillows_own_bomb_error_reads_as_the_same_refusal(monkeypatch):
    """Past twice its MAX_IMAGE_PIXELS Pillow raises inside open(); that must
    read as our refusal, not escape as a stray exception type."""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="too large to open safely"):
        image_suffix(io.BytesIO(_encoded(Image.new("L", (10, 10)), "PNG")))


@pytest.mark.parametrize(
    "size", [(4000, 1000), (1000, 4000), (3000, 3000), (1600, 1600), (1000, 390), (900, 900)]
)
def test_the_model_copy_fits_the_standard_vision_tier_whatever_its_shape(size):
    copy, _fmt = model_copy(Image.new("RGB", size, "white"))
    width, height = copy.size
    assert max(width, height) <= MODEL_IMAGE_MAX_EDGE
    assert width * height <= MODEL_IMAGE_MAX_PIXELS
    assert abs(width / height - size[0] / size[1]) < 0.01  # the shape is kept
    # ...and the budget is used, not shrunk further than a bound requires.
    assert max(width, height) >= MODEL_IMAGE_MAX_EDGE - 2 or (
        width * height >= 0.99 * MODEL_IMAGE_MAX_PIXELS
    )


@pytest.mark.parametrize(
    ("size", "enlarged"),
    [
        ((300, 200), (600, 400)),  # twice, the most it is ever enlarged
        ((20, 20), (40, 40)),
        # The rotated-text chart the prompt was measured on: its long edge
        # reaches the tier's before twice its size does.
        ((1024, 390), (1568, 597)),
    ],
)
def test_a_small_image_is_enlarged_at_most_twice_and_never_past_the_tier(size, enlarged):
    """Measured (PR_C_MEASURE, 5 runs each): at its own 1024x390 the caption
    model misread two rotated values of a real chart in every run (12.5 read
    as 12.3, 8.9 as 8.3); enlarged within the tier it read all 28 in every
    run. Enlarging adds no information, so it stops at twice the size."""
    copy, _fmt = model_copy(Image.new("RGB", size, "white"))
    assert copy.size == enlarged


def test_a_rotated_phone_photo_reaches_the_model_upright():
    stored = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(stored).rectangle([0, 0, 99, 74], fill="red")  # top-left as stored
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90° clockwise to display"
    photo = Image.open(io.BytesIO(_encoded(stored, "JPEG", exif=exif.tobytes(), quality=95)))
    copy, _fmt = model_copy(photo)
    assert copy.size == (600, 800)  # upright (and enlarged twice)
    red, green, _blue = copy.getpixel((copy.width - 10, 10))
    assert red > 200 and green < 60  # the stored top-left corner is now top-right
    assert min(copy.getpixel((10, 10))) > 200


def test_transparency_is_flattened_onto_white():
    logo = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    ImageDraw.Draw(logo).rectangle([0, 0, 24, 49], fill=(0, 0, 255, 255))
    copy, _fmt = model_copy(logo)
    assert (copy.mode, copy.size) == ("RGB", (100, 100))
    assert copy.getpixel((80, 50)) == (255, 255, 255)
    assert copy.getpixel((20, 50)) == (0, 0, 255)


def test_a_sixteen_bit_greyscale_image_keeps_its_tones():
    """A plain conversion clips every 16-bit value above 255 to white."""
    deep = Image.new("I;16", (400, 100))
    for x in range(400):  # paste() cannot fill a 16-bit image with a number; putpixel can
        for y in range(100):
            deep.putpixel((x, y), (0, 16384, 32768, 65535)[x // 100])
    copy, _fmt = model_copy(Image.open(io.BytesIO(_encoded(deep, "PNG"))))
    assert copy.size == (800, 200)
    tones = [copy.getpixel((200 * stripe + 100, 100))[0] for stripe in range(4)]
    assert tones == pytest.approx([0, 64, 128, 255], abs=1)


def test_a_chart_is_kept_lossless_and_a_photo_is_compressed():
    _chart, chart_fmt = model_copy(Image.open(REAL_CHART))
    _photo, photo_fmt = model_copy(_photo_like())
    assert (chart_fmt, photo_fmt) == ("PNG", "JPEG")


def test_the_model_copy_stays_near_a_megabyte_even_for_pure_noise():
    """The verify batch carries a copy per claim that retrieves the image
    (256 MB per batch): the worst case has to be bounded, not typical."""
    copy, fmt = model_copy(Image.effect_noise((3000, 2000), 128).convert("RGB"))
    assert len(encode_image(copy, fmt)) < 1_200_000


def test_the_model_copy_carries_no_camera_metadata():
    exif = Image.Exif()
    exif[0x010F] = "PhoneMaker"
    exif[0x8825] = {1: "N", 2: (40.0, 26.0, 0.0)}  # GPS
    photo = Image.open(
        io.BytesIO(_encoded(Image.new("RGB", (60, 40)), "JPEG", exif=exif.tobytes()))
    )
    assert photo.getexif()  # planted
    copy, _fmt = model_copy(photo)
    for fmt in ("PNG", "JPEG"):
        reread = Image.open(io.BytesIO(encode_image(copy, fmt)))
        assert not reread.getexif()
        assert "exif" not in reread.info


@pytest.mark.parametrize(
    ("mode", "fill"), [("RGB", (200, 30, 30)), ("RGBA", (200, 30, 30, 128)), ("L", 128)]
)
def test_a_pdf_figure_is_written_byte_for_byte_as_before(tmp_path, mode, fill):
    figure = Image.new(mode, (64, 48))
    ImageDraw.Draw(figure).ellipse([4, 4, 40, 40], fill=fill)
    figure.save(tmp_path / "before.png", format="PNG")  # what ingest_parsed always wrote
    assert encode_image(figure, "PNG") == (tmp_path / "before.png").read_bytes()


def test_parse_image_makes_one_whole_page_figure_named_after_the_file():
    parsed = parse_image(REAL_CHART, name="Child stunting 2024.png")
    assert (parsed.title, parsed.sections, parsed.tables) == (None, [], [])
    [figure] = parsed.figures
    assert (figure.page, figure.caption, figure.fmt) == (None, "Child stunting 2024", "PNG")
    assert figure.image.size == (992, 1010)  # enlarged twice to read its small print


def test_parse_image_runs_the_upload_gate_again(tmp_path, bomb_png, no_decoding):
    path = tmp_path / "f00d.png"
    path.write_bytes(bomb_png)
    with pytest.raises(ValueError, match=r"^'huge.png' is 12000×6000 pixels"):
        parse_image(path, name="huge.png")


def test_a_truncated_image_fails_loudly_when_it_is_decoded_naming_the_file(tmp_path):
    data = REAL_CHART.read_bytes()
    path = tmp_path / "beef.png"
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError, match=r"^'cut.png' could not be read as an image"):
        parse_image(path, name="cut.png")


def _image_upload(conn, tmp_path, data: bytes, name: str, suffix: str):
    path = tmp_path / f"{dbmod.new_id()}{suffix}"
    path.write_bytes(data)
    run_id = dbmod.create_run(conn)
    upload_id = dbmod.add_upload(conn, "SOURCE", name, str(path), "h", source_type="image")
    return run_id, upload_id, path


def test_ingest_image_indexes_one_figure_read_by_the_caption_model(conn, tmp_path):
    original = REAL_CHART.read_bytes()
    run_id, upload_id, path = _image_upload(conn, tmp_path, original, "stunting chart.png", ".png")
    reading = "Text in the image:\nBurundi: 55.3\nDescription:\nA bar chart of child stunting."
    seen = []

    def describe(image):
        seen.append(image.size)
        return reading

    doc_id = ingest_image(
        conn,
        FakeEmbedder(dim=DIM),
        run_id,
        path,
        name="stunting chart.png",
        kind="SOURCE",
        figures_dir=tmp_path / "figures",
        upload_id=upload_id,
        describe=describe,
    )

    assert seen == [(992, 1010)]
    document = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    assert document["title"] == "stunting chart"
    assert json.loads(document["metadata"]) == {"sections": []}
    [chunk] = conn.execute("SELECT * FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
    assert (chunk["kind"], chunk["page"]) == ("figure", None)
    assert chunk["text"] == f"stunting chart\n\n{reading}"
    figure = conn.execute("SELECT * FROM figures WHERE doc_id = ?", (doc_id,)).fetchone()
    assert figure["description"] == reading
    stored = Path(figure["image_path"])
    assert stored.suffix == ".png" and stored.read_bytes().startswith(b"\x89PNG")
    assert Image.open(stored).size == (992, 1010)
    assert path.read_bytes() == original  # the original upload is never rewritten


def test_an_uploaded_photo_is_kept_for_the_judge_as_a_bounded_jpeg(conn, tmp_path):
    photo = _encoded(_photo_like((3000, 2000)), "JPEG", quality=92)
    run_id, upload_id, path = _image_upload(conn, tmp_path, photo, "field.jpg", ".jpg")
    doc_id = ingest_image(
        conn,
        FakeEmbedder(dim=DIM),
        run_id,
        path,
        name="field.jpg",
        kind="SOURCE",
        figures_dir=tmp_path / "figures",
        upload_id=upload_id,
        describe=lambda image: "Description: a field.",
    )
    figure = conn.execute("SELECT * FROM figures WHERE doc_id = ?", (doc_id,)).fetchone()
    stored = Path(figure["image_path"])
    assert stored.suffix == ".jpg" and stored.read_bytes().startswith(b"\xff\xd8\xff")
    width, height = Image.open(stored).size
    assert width * height <= MODEL_IMAGE_MAX_PIXELS and max(width, height) <= MODEL_IMAGE_MAX_EDGE


def _fake_messages_client(text: str, stop_reason: str, sent: list):
    from types import SimpleNamespace

    def create(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason=stop_reason,
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_a_reading_cut_off_at_max_tokens_is_refused_not_kept():
    """Half a transcription is not evidence: it would index as if complete."""
    from authorai.llm import AnthropicClient

    client = AnthropicClient(api_key="test-key")
    client._client = _fake_messages_client(
        "Text in the image:\nBurundi: 55.3\nNig", "max_tokens", []
    )
    with pytest.raises(RuntimeError, match="cut off at max_tokens=4096"):
        client.describe_image(
            model="m", image=Image.new("RGB", (8, 8)), prompt="p", max_tokens=4096
        )


def test_a_complete_reading_comes_back_stripped_with_the_budget_asked_for():
    from authorai.llm import AnthropicClient

    sent: list[dict] = []
    client = AnthropicClient(api_key="test-key")
    client._client = _fake_messages_client("  Description: a chart.\n", "end_turn", sent)
    reading = client.describe_image(
        model="m", image=Image.new("RGB", (8, 8)), prompt="p", max_tokens=4096
    )
    assert reading == "Description: a chart."
    assert sent[0]["max_tokens"] == 4096
    assert sent[0]["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
