import numpy as np

from src.hallcheck.config import settings
from src.hallcheck import cli, extract_claims, verify, chunking, tables
from src.hallcheck.models import Document, Chunk


def test_settings_defaults():
    assert settings.db_url.endswith("hallcheck.db")
    assert settings.openai_embed_model == "text-embedding-3-large"
    assert settings.openai_chat_model in {"gpt-4.1", "gpt-4o-mini"}
    assert settings.chunk_tokens == 220
    assert settings.chunk_overlap_min == 20
    assert settings.rerank_model == "gpt-4o-mini"


def test_module_exports():
    assert hasattr(cli, "main")
    assert callable(cli.main)
    assert hasattr(verify, "index_sources")
    assert hasattr(verify, "verify_report")
    claims = list(extract_claims.find_claims("In 2020, 55% of people agreed."))
    assert claims


def test_chunk_text_hybrid_defaults():
    sample = "This is a short paragraph about health. It continues to mention statistics. " \
        "Later, the topic switches to economics, which should trigger a new chunk. Economics relies on data."
    chunks = chunking.chunk_text(sample, target_tokens=40, overlap=10)
    assert chunks
    assert all(isinstance(chunk, str) and chunk.strip() for chunk in chunks)


def test_document_model_includes_author_column():
    column_names = {column.name for column in Document.__table__.columns}
    assert "author" in column_names


def test_chunk_model_includes_tables_column():
    column_names = {column.name for column in Chunk.__table__.columns}
    assert "tables" in column_names


def test_document_model_includes_pipeline_columns():
    column_names = {column.name for column in Document.__table__.columns}
    assert "router_label" in column_names
    assert "is_scanned" in column_names
    assert "content_hash" in column_names
    assert "extractor_chain" in column_names
    assert "document_json" in column_names


def test_extract_tables_from_text_handles_rank_table():
    raw = (
        "Rank1 Country 2000 2008 2016 2025 GHI scores less than 5 collectively ranked "
        "1-25.2 Armenia 20.3 10.8 6.7 <5 2 Belarus <5 <5 <5 <5"
    )
    html_tables = tables.extract_tables_from_text(raw)
    assert html_tables
    assert "Armenia" in html_tables[0]


def test_numeric_claim_captured_via_sum_matching():
    claim = verify.Claim(  # type: ignore[arg-type]
        doc_id=1,
        sentence="In total, 42 countries face serious or alarming hunger levels.",
        value=42.0,
        units=None,
        year=None,
        meta=None,
    )
    sims = np.array([0.75], dtype=np.float32)
    evidence = {
        "candidates": [
            {
                "snippet": (
                    "Furthermore, many countries are slipping backward: in 27 countries with low, moderate, "
                    "serious, or alarming 2025 GHI scores, hunger has increased since 2016. "
                    "In another 35 countries, hunger is designated as serious, and 7 countries are considered alarming."
                )
            }
        ]
        }
    status, confidence = verify._decide_verdict(claim, sims, evidence)  # type: ignore[attr-defined]
    assert status == verify.VerdictStatus.SUPPORTED
    assert confidence >= 0.4


def test_entity_guard_rejects_mismatched_subjects():
    claim_text = "A total of 42 bananas were harvested."
    evidence_text = "A total of 42 apples were harvested in 2025."
    assert not verify._entity_guard_allows_support(claim_text, evidence_text)  # type: ignore[attr-defined]


def test_entity_guard_accepts_overlapping_entities():
    claim_text = "A total of 42 bananas were harvested."
    evidence_text = "Harvest records show 42 banana crops were harvested."
    assert verify._entity_guard_allows_support(claim_text, evidence_text)  # type: ignore[attr-defined]


def test_gpt_rerank_support_can_promote_missing_numbers(monkeypatch):
    claim = verify.Claim(  # type: ignore[arg-type]
        doc_id=2,
        sentence="The initiative supported 100 participants.",
        value=100.0,
        units=None,
        year=None,
        meta=None,
    )
    sims = np.array([0.72], dtype=np.float32)
    evidence = {
        "candidates": [
            {
                "snippet": "The program backed dozens of local participants across regions.",
                "rerank_score": 0.82,
                "rerank_label": "supported",
            }
        ]
    }

    def fake_confirm(claim_text: str, evidence_text: str):
        return True, 0.8, None

    monkeypatch.setattr(verify, "_confirm_entity_alignment", fake_confirm)

    status, confidence = verify._decide_verdict(claim, sims, evidence)  # type: ignore[attr-defined]
    assert status == verify.VerdictStatus.SUPPORTED
    assert 0.3 <= confidence <= 0.75


def test_alignment_override_blocks_when_confident(monkeypatch):
    claim = verify.Claim(  # type: ignore[arg-type]
        doc_id=3,
        sentence="The report lists 20 critical hotspots.",
        value=20.0,
        units=None,
        year=None,
        meta=None,
    )
    sims = np.array([0.68], dtype=np.float32)
    evidence = {
        "candidates": [
            {
                "snippet": "The document highlights 18 hotspots across regions with critical needs.",
                "rerank_score": 0.74,
                "rerank_label": "supported",
            }
        ]
    }

    def fake_confirm(claim_text: str, evidence_text: str):
        return False, 0.85, "Different subject"

    monkeypatch.setattr(verify, "_confirm_entity_alignment", fake_confirm)

    status, confidence = verify._decide_verdict(claim, sims, evidence)  # type: ignore[attr-defined]
    assert status == verify.VerdictStatus.NOT_FOUND
    assert confidence <= 0.6
