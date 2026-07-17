from types import SimpleNamespace
from pathlib import Path

from PyPDF2 import PdfWriter

from author_ai.models import Chart
from author_ai.pipelines.ingestion import IngestionPipeline
from author_ai.services.charts import chart_to_chunks
from author_ai.services.verdict_classifier import VerdictClassifier


def _blank_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)


def test_chart_chunks_include_summary_and_fact(settings):
    chart = Chart(
        id="chart-123",
        doc_id="doc-abc",
        page=2,
        figure_label="Figure 2",
        chart_type="line",
        raw_json={
            "chart_type": "line",
            "title": "Admissions over time",
            "x_axis": {"label": "Year"},
            "y_axis": {"label": "Admissions", "unit": "people"},
            "series": [
                {
                    "name": "Admissions",
                    "points": [
                        {"x": "2019", "y": 10},
                        {"x": "2020", "y": 14},
                        {"x": "2021", "y": 12},
                    ],
                }
            ],
        },
    )
    chunks = chart_to_chunks(chart, max_fact_points=3)
    summary = [c for c in chunks if c.get("chunk_type") == "chart_summary"]
    facts = [c for c in chunks if c.get("chunk_type") == "chart_fact"]
    assert summary, "expected a chart_summary chunk"
    assert facts, "expected chart_fact chunks"
    assert summary[0]["metadata"]["chart_id"] == chart.id
    assert any(fact.get("metadata", {}).get("x_value") == "2019" for fact in facts)


def test_ingestion_persists_chart_rows(monkeypatch, tmp_path, settings):
    pdf_path = tmp_path / "sample.pdf"
    _blank_pdf(pdf_path)

    fake_chart = Chart(
        id="chart-1",
        doc_id="doc-1",
        page=1,
        figure_label="Figure 1",
        chart_type="bar",
        raw_json={"chart_type": "bar", "series": []},
    )

    def fake_extract(path, doc_id):
        return [fake_chart]

    def fake_chart_chunks(chart):
        return [
            {
                "chunk_id": "chunk-1",
                "doc_id": chart.doc_id,
                "text": "Chart summary text",
                "chunk_type": "chart_summary",
                "chart_id": chart.id,
                "metadata": {
                    "chunk_type": "chart_summary",
                    "chart_id": chart.id,
                    "parent_id": "page-1",
                    "parent_title": "Page 1 - Chart",
                    "parent_page": 1,
                },
            }
        ]

    monkeypatch.setattr("author_ai.pipelines.ingestion.extract_charts_from_pdf", fake_extract)
    monkeypatch.setattr("author_ai.pipelines.ingestion.chart_to_chunks", fake_chart_chunks)
    monkeypatch.setattr("author_ai.services.ocr.should_run_ocr", lambda _path: False)

    pipeline = IngestionPipeline()
    pipeline.ingest(pdf_path, doc_id="doc-1", doc_type="SOURCE")

    charts = pipeline.repo.list_charts("doc-1")
    assert charts and charts[0]["id"] == "chart-1"
    chunks = pipeline.repo.list_chunks("doc-1")
    assert any(chunk.get("chunk_type") == "chart_summary" for chunk in chunks)


def test_verdict_prompt_marks_chart_evidence(monkeypatch, settings):
    classifier = VerdictClassifier()

    captured = {}

    class FakeCompletions:
        def create(self, model=None, messages=None, temperature=None, **_kwargs):
            captured["messages"] = messages

            class Resp:
                def __init__(self):
                    self.choices = [SimpleNamespace(message=SimpleNamespace(content='{"label":"SUPPORTED","reason":""}'))]

            return Resp()

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    classifier.client = SimpleNamespace(chat=FakeChat())

    claim = SimpleNamespace(text="Admissions peaked at 14k in 2020", metadata={})
    evidence = {
        "snippet": "Chart shows admissions rising to 14000 in 2020.",
        "parent": {"page": 3},
        "chunk_type": "chart_summary",
        "figure_label": "Figure 3",
        "metadata": {"chunk_type": "chart_summary"},
    }

    classifier.classify(claim, evidence)

    user_messages = [msg for msg in captured.get("messages", []) if msg.get("role") == "user"]
    system_messages = [msg for msg in captured.get("messages", []) if msg.get("role") == "system"]
    assert any("Evidence from chart" in msg["content"] for msg in user_messages)
    assert any("chart evidence" in msg["content"] for msg in system_messages)
