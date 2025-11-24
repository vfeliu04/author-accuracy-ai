from pathlib import Path
from types import SimpleNamespace

from author_ai.pipelines.accuracy import AccuracyPipeline
from author_ai.services.vector_store import VectorStore


def test_accuracy_pipeline_retrieval(settings, tmp_path):
    pipeline = AccuracyPipeline()
    pipeline.vector_store = VectorStore("sources_test", base_dir=settings.faiss_index_dir)

    source_payload = {
        "document": {"sections": []},
        "body_text": "",
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "doc_id": "source-doc",
                "text": "Region A recorded 30 percent food insecurity in 2024.",
            }
        ],
        "doc_id": "source-doc",
    }

    report_payload = {
        "document": {"sections": [{"text": ""}]},
        "body_text": "Region A expects 30 percent food insecurity this year.",
        "chunks": [],
        "doc_id": "report-doc",
    }

    def fake_ingest(path: Path, doc_id: str, doc_type: str = "SOURCE"):
        return source_payload if doc_type == "SOURCE" else report_payload

    pipeline.ingestion = SimpleNamespace(ingest=fake_ingest)

    pipeline.index_source({"path": "/tmp/source.pdf", "upload_id": "source-doc"})
    result = pipeline.verify_report({"path": "/tmp/report.pdf", "upload_id": "report-doc"})

    assert result["report_id"] == "report-doc"
    assert result["claims"], "claims should be extracted"
    assert any(claim["verdict"] == "SUPPORTED" for claim in result["claims"])
