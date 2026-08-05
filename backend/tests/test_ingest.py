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
