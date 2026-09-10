from pathlib import Path

import pytest
from PIL import Image

import authorai.ingest as ingest_mod
from authorai import db as dbmod
from authorai.embeddings import FakeEmbedder
from authorai.ingest import (
    ParsedDocument,
    ParsedFigure,
    ParsedSection,
    ParsedTable,
    ingest_pdf,
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
