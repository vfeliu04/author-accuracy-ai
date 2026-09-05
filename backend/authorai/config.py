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
        # Without this, an aliased field ignores its FIELD NAME as an init
        # kwarg: Settings(anthropic_api_key="x") silently fell through to the
        # env/.env sources, so tests only worked where a real key existed —
        # first CI run (no .env) exposed it.
        populate_by_name=True,
    )

    db_path: Path = Path("data/authorai.db")
    figures_dir: Path = Path("data/figures")

    def run_figures_dir(self, run_id: str) -> Path:
        """The ABSOLUTE figures directory for a run — the one convention shared
        by ingest, the dedup copy, torn-ingest cleanup, and run deletion. The
        default figures_dir is relative, so an unresolved variant would store
        CWD-relative image paths that break verification (and deletion) the
        moment the server starts from a different directory."""
        return Path(self.figures_dir).resolve() / run_id

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
    # Verdicts are the product's core judgment — frontier model, like extraction.
    verdict_model: str = "claude-opus-5"
    # Bibliographic metadata extraction is a cheap bounded task, like captions.
    metadata_model: str = "claude-haiku-4-5"
    # The validity rubric is a quality judgment over the whole report.
    validity_model: str = "claude-opus-5"
    validity_weights: str = "coverage:0.25,consistency:0.25,methodology:0.2,context:0.2,recency:0.1"
    # Word-boundary matched against source publishers (see credibility.py).
    # Tier 1: intergovernmental bodies and their agencies (WMO/UNCCD added
    # 2026-08-21 — both published sources in live runs and are peers of
    # WHO/FAO, which were already listed).
    authority_tier1: str = (
        "FAO,Food and Agriculture Organization,UN,United Nations,World Bank,IMF,"
        "WHO,World Health Organization,UNICEF,OECD,Welthungerhilfe,"
        "WMO,World Meteorological Organization,UNCCD"
    )
    # Tier 2: established research institutions and publishers.
    authority_tier2: str = (
        "Reuters,Associated Press,BBC,Nature,Science,Lancet,Elsevier,"
        "National Drought Mitigation Center,NDMC,International Water Management Institute,"
        "IWMI,CGIAR,World Climate Research Programme,WCRP"
    )
    crossref_mailto: str | None = None
    job_poll_seconds: float = 2.0
    # HTTP layer. api_key unset means the app REFUSES to start (fail-closed —
    # v1 served everything openly when its key env var was missing).
    api_key: str | None = None
    uploads_dir: Path = Path("data/uploads")
    max_upload_bytes: int = 50_000_000  # per file
    # Whole-request ceiling, checked against Content-Length BEFORE the body is
    # read, so an unauthenticated attacker cannot push gigabytes at us.
    max_request_bytes: int = 220_000_000
    max_source_files: int = 20
    cors_origins: str = "http://localhost:5173"
    # Swagger/ReDoc/OpenAPI expose the full route surface; keep them for local
    # dev, disable for an exposed deployment.
    docs_enabled: bool = True
    # Chat: a Sonnet-class model with prompt caching over the static per-run
    # context. Cheaper than the Opus judgments; the answer is grounded Q&A.
    chat_model: str = "claude-sonnet-5"
    chat_max_tokens: int = 2048
    chat_history_turns: int = 12
