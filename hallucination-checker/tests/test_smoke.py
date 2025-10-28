from src.hallcheck.config import settings
from src.hallcheck import cli, extract_claims, verify


def test_settings_defaults():
    assert settings.db_url.endswith("hallcheck.db")
    assert settings.openai_embed_model == "text-embedding-3-large"


def test_module_exports():
    assert hasattr(cli, "main")
    assert callable(cli.main)
    assert hasattr(verify, "index_sources")
    assert hasattr(verify, "verify_report")
    claims = list(extract_claims.find_claims("In 2020, 55% of people agreed."))
    assert claims
