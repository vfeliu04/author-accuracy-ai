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


def test_heading_prefix_filtered(settings):
    pipeline = AccuracyPipeline()
    text = (
        "3.2 Climate Extremes\n"
        "Climate change threatens agricultural stability across Africa, Asia, and Latin America.\n"
        "5.5 out of 10 people starve in Somalia."
    )
    claims = pipeline._extract_claims(text, "report-1", sections=[])
    claim_texts = [claim.text for claim in claims]
    assert any("5.5 out of 10" in entry for entry in claim_texts)
    assert all("Climate change threatens" not in entry for entry in claim_texts)


def test_heading_with_numeric_claim_survives(settings):
    pipeline = AccuracyPipeline()
    text = "5.3 out of 10 people are starving in Somalia."
    claims = pipeline._extract_claims(text, "report-2", sections=[])
    claim_texts = [claim.text for claim in claims]
    assert any("5.3 out of 10 people are starving" in entry for entry in claim_texts), claim_texts


def test_inline_heading_with_real_number(settings):
    pipeline = AccuracyPipeline()
    text = (
        "3.4 Key Findings: 20% of households face acute food insecurity.\n"
        "Additional context without numbers."
    )
    claims = pipeline._extract_claims(text, "report-3", sections=[])
    claim_texts = [claim.text for claim in claims]
    assert any("20% of households" in entry for entry in claim_texts), claim_texts


def test_pure_heading_dropped(settings):
    pipeline = AccuracyPipeline()
    text = "2.2 Climate Extremes"
    claims = pipeline._extract_claims(text, "report-4", sections=[])
    assert claims == []


def test_numbered_heading_with_text_no_digits_dropped(settings):
    pipeline = AccuracyPipeline()
    text = "2. Global hunger remains high."
    claims = pipeline._extract_claims(text, "report-5", sections=[])
    assert claims == []


def test_bullet_with_percent_survives(settings):
    pipeline = AccuracyPipeline()
    text = "- 30% of households report food insecurity."
    claims = pipeline._extract_claims(text, "report-6", sections=[])
    assert any("30% of households" in claim.text for claim in claims)


def test_year_with_quantity_survives(settings):
    pipeline = AccuracyPipeline()
    text = "2024 saw 15 million people affected by hunger."
    claims = pipeline._extract_claims(text, "report-7", sections=[])
    assert any("15 million people" in claim.text for claim in claims)


def test_section_label_with_colon_and_number_survives(settings):
    pipeline = AccuracyPipeline()
    text = "Section 7.1: 12 Million people are displaced due to conflict."
    claims = pipeline._extract_claims(text, "report-8", sections=[])
    assert any("12 Million people" in claim.text for claim in claims)
