"""Application configuration handling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class Settings(BaseModel):
    """Runtime settings loaded from environment variables."""

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    db_url: str = Field(default="sqlite:///./hallcheck.db", alias="HALLCHECK_DB_URL")
    openai_embed_model: str = Field(default="text-embedding-3-large", alias="OPENAI_EMBED_MODEL")
    openai_chat_model: str = Field(default="gpt-4.1", alias="OPENAI_CHAT_MODEL")
    chunk_tokens: int = Field(default=350, alias="CHUNK_TOKENS")
    chunk_overlap: int = Field(default=60, alias="CHUNK_OVERLAP")
    project_root: Optional[Path] = None

    @classmethod
    def load(cls) -> "Settings":
        """Create a Settings instance from environment variables."""
        data = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "HALLCHECK_DB_URL": os.getenv("HALLCHECK_DB_URL", "sqlite:///./hallcheck.db"),
            "OPENAI_EMBED_MODEL": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"),
            "OPENAI_CHAT_MODEL": os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1"),
            "CHUNK_TOKENS": int(os.getenv("CHUNK_TOKENS", 350)),
            "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 60)),
        }
        root_env = os.getenv("HALLCHECK_ROOT")
        root_path = Path(root_env).resolve() if root_env else Path.cwd()
        settings = cls(**data, project_root=root_path)
        return settings


settings = Settings.load()
