"""
Configuration utilities for the Author AI backend services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path
from typing import Optional


def _bool_env(key: str, default: str = "true") -> bool:
    return os.getenv(key, default).lower() == "true"


def _path_env(key: str, default: str) -> Path:
    return Path(os.getenv(key, default)).resolve()


def _float_env(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def _int_env(key: str, default: int) -> int:
    return int(os.getenv(key, default))


@dataclass(frozen=True)
class Settings:
    """Container for environment-driven settings."""

    data_root: Path = field(default_factory=lambda: _path_env("DATA_ROOT", "./data"))
    sqlite_path: Path = field(default_factory=lambda: _path_env("SQLITE_PATH", "./data/accuracy.db"))
    cache_dir: Path = field(default_factory=lambda: _path_env("CACHE_DIR", "./data/cache"))
    faiss_index_dir: Path = field(default_factory=lambda: _path_env("FAISS_INDEX_DIR", "./data/indexes"))
    log_dir: Path = field(default_factory=lambda: _path_env("LOG_DIR", "./logs"))

    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("API_KEY"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))
    explanation_model: str = field(default_factory=lambda: os.getenv("EXPLANATION_MODEL", "gpt-4o-mini"))
    rerank_model: str = field(default_factory=lambda: os.getenv("RERANK_MODEL", "gpt-4o-mini"))
    claim_classifier_model: str = field(default_factory=lambda: os.getenv("CLAIM_CLASSIFIER_MODEL", "gpt-4o-mini"))

    rerank_with_gpt: bool = field(default_factory=lambda: _bool_env("RERANK_WITH_GPT", "true"))
    rerank_max_candidates: int = field(default_factory=lambda: _int_env("RERANK_MAX_CANDIDATES", 10))
    retrieval_support_threshold: float = field(default_factory=lambda: _float_env("RETRIEVAL_SUPPORT_THRESHOLD", 0.35))

    ocr_enabled: bool = field(default_factory=lambda: _bool_env("PDF_OCR_ENABLED", "true"))
    ocr_force_when_scanned: bool = field(default_factory=lambda: _bool_env("PDF_OCR_FORCE_WHEN_SCANNED", "true"))
    ocr_min_text_threshold: int = field(default_factory=lambda: _int_env("PDF_OCR_MIN_TEXT_THRESHOLD", 256))
    ocr_low_quality_ratio: float = field(default_factory=lambda: _float_env("PDF_OCR_LOW_QUALITY_RATIO", 0.6))
    ocr_binary_path: str = field(default_factory=lambda: os.getenv("OCR_MY_PDF_PATH", "/usr/local/bin/ocrmypdf"))

    table_engine: str = field(default_factory=lambda: os.getenv("PDF_TABLES_ENGINE", "tabula"))
    tabula_jar_path: str = field(default_factory=lambda: os.getenv("TABULA_JAR_PATH", "/usr/local/tabula/tabula.jar"))
    tabula_extra_classpath: str = field(default_factory=lambda: os.getenv("TABULA_JAI_CLASSPATH", ""))
    pdf_tables_api_key: Optional[str] = field(default_factory=lambda: os.getenv("PDF_TABLES_API_KEY"))

    claim_relevance_min: float = field(default_factory=lambda: _float_env("CLAIM_RELEVANCE_MIN", 0.35))
    claim_context_limit: int = field(default_factory=lambda: _int_env("CLAIM_CONTEXT_LIMIT", 5))
    source_context_limit: int = field(default_factory=lambda: _int_env("SOURCE_CONTEXT_LIMIT", 6))
    claim_priority_weight: float = field(default_factory=lambda: _float_env("CLAIM_PRIORITY_WEIGHT", 1.5))
    chat_history_length: int = field(default_factory=lambda: _int_env("CHAT_HISTORY_LENGTH", 6))
    llm_chat_model: str = field(default_factory=lambda: os.getenv("LLM_CHAT_MODEL", "gpt-4o-mini"))

    claim_vector_path: Path = field(default_factory=lambda: _path_env("CLAIM_VECTOR_PATH", "./data/indexes/claims"))
    source_vector_path: Path = field(default_factory=lambda: _path_env("SOURCE_VECTOR_PATH", "./data/indexes/sources"))

    credibility_title_match_threshold: float = field(default_factory=lambda: _float_env("CRED_TITLE_MATCH_THRESHOLD", 0.85))
    credibility_recency_decay_years: int = field(default_factory=lambda: _int_env("CRED_RECENCY_DECAY_YEARS", 10))

    validity_topic_threshold: float = field(default_factory=lambda: _float_env("VALIDITY_TOPIC_THRESHOLD", 0.35))
    validity_weights: str = field(
        default_factory=lambda: os.getenv(
            "VALIDITY_WEIGHTS", "coverage:0.25,consistency:0.25,methodology:0.2,context:0.2,recency:0.1"
        )
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of Settings."""

    settings = Settings()

    # Ensure key directories exist early
    for path in (
        settings.data_root,
        settings.cache_dir,
        settings.faiss_index_dir,
        settings.claim_vector_path,
        settings.log_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    return settings
