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
