"""Application settings.

All settings come from environment variables (or backend/.env). Settings are
constructed where needed and passed explicitly — no cached global, so tests can
build their own without monkeypatching.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor to backend/.env regardless of the process working directory —
# a CWD-relative ".env" silently loads nothing when started from elsewhere.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AUTHORAI_",
        env_file=_ENV_FILE,
        extra="ignore",
    )

    db_path: Path = Path("data/authorai.db")
    figures_dir: Path = Path("data/figures")
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    # Read from the unprefixed provider names so the existing .env keeps working.
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    # Extraction is the accuracy-critical language judgment the whole score rests
    # on — it runs on the frontier model. Figure captioning is a cheap, bounded
    # description task, so it stays on Haiku (as planned in CLAUDE.md Phase 3).
    extraction_model: str = "claude-opus-5"
    caption_model: str = "claude-haiku-4-5"
