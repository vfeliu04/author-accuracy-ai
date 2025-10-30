from src.hallcheck.config import settings
from src.hallcheck import cli, extract_claims, verify, chunking
from src.hallcheck.models import Document


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
