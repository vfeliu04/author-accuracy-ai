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
    chunk_tokens: int = Field(default=220, alias="CHUNK_TOKENS")
    chunk_overlap: int = Field(default=60, alias="CHUNK_OVERLAP")
    chunk_overlap_min: int = Field(default=20, alias="CHUNK_OVERLAP_MIN")
    chunk_overlap_max: int = Field(default=80, alias="CHUNK_OVERLAP_MAX")
    chunk_topic_similarity: float = Field(default=0.72, alias="CHUNK_TOPIC_SIMILARITY")
    chunk_stable_similarity: float = Field(default=0.86, alias="CHUNK_STABLE_SIMILARITY")
    rerank_with_gpt: bool = Field(default=True, alias="RERANK_WITH_GPT")
    rerank_model: str = Field(default="gpt-4o-mini", alias="RERANK_MODEL")
    rerank_max_candidates: int = Field(default=5, alias="RERANK_MAX_CANDIDATES")
    pdf_pipeline_legacy: bool = Field(default=False, alias="PDF_PIPELINE_LEGACY")
    format_tables_with_gpt: bool = Field(default=True, alias="TABLES_FORMAT_WITH_GPT")
    project_root: Optional[Path] = None

    @classmethod
    def load(cls) -> "Settings":
        """Create a Settings instance from environment variables."""
        data = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "HALLCHECK_DB_URL": os.getenv("HALLCHECK_DB_URL", "sqlite:///./hallcheck.db"),
            "OPENAI_EMBED_MODEL": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"),
            "OPENAI_CHAT_MODEL": os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1"),
            "CHUNK_TOKENS": int(os.getenv("CHUNK_TOKENS", 220)),
            "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 60)),
            "CHUNK_OVERLAP_MIN": int(os.getenv("CHUNK_OVERLAP_MIN", 20)),
            "CHUNK_OVERLAP_MAX": int(os.getenv("CHUNK_OVERLAP_MAX", 80)),
            "CHUNK_TOPIC_SIMILARITY": float(os.getenv("CHUNK_TOPIC_SIMILARITY", 0.72)),
            "CHUNK_STABLE_SIMILARITY": float(os.getenv("CHUNK_STABLE_SIMILARITY", 0.86)),
            "RERANK_WITH_GPT": os.getenv("RERANK_WITH_GPT", "true").lower() not in {"0", "false", "no"},
            "RERANK_MODEL": os.getenv("RERANK_MODEL", "gpt-4o-mini"),
            "RERANK_MAX_CANDIDATES": int(os.getenv("RERANK_MAX_CANDIDATES", 5)),
            "PDF_PIPELINE_LEGACY": os.getenv("PDF_PIPELINE_LEGACY", "false").lower() in {"1", "true", "yes"},
            "TABLES_FORMAT_WITH_GPT": os.getenv("TABLES_FORMAT_WITH_GPT", "true").lower() not in {"0", "false", "no"},
        }
        root_env = os.getenv("HALLCHECK_ROOT")
        root_path = Path(root_env).resolve() if root_env else Path.cwd()
        settings = cls(**data, project_root=root_path)
        return settings


settings = Settings.load()
