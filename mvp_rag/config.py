from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# This line loads environment variables from a file named `.env` in the project root.
# It's a convenient way to manage local development settings without hardcoding them.
load_dotenv()


# The `@dataclass` decorator automatically generates special methods like `__init__` and `__repr__`.
# `slots=True` is a performance optimization that uses a more memory-efficient structure for instances.
@dataclass(slots=True)
class Settings:
    """Application configuration loaded from environment variables."""

    # These are the attributes of our settings class.
    # They are defined with type hints (e.g., `str`, `Path`) for clarity and correctness.
    # `Optional[str]` means the value can be a string or `None`.
    # `field(default=...)` provides a default value if the environment variable isn't set.

    # --- API Keys and Endpoints ---
    openai_api_key: Optional[str] = field(default=None)
    qdrant_url: Optional[str] = field(default=None)
    qdrant_api_key: Optional[str] = field(default=None)

    # --- Application Behavior ---
    collection_name: str = field(default="mvp_docs")  # Name of the collection in Qdrant.
    upload_dir: Path = field(default_factory=lambda: Path("uploads"))  # Directory for temp uploads.
    log_level: str = field(default="INFO")  # Logging level (e.g., INFO, DEBUG).
    max_upload_bytes: int = field(default=20 * 1024 * 1024)  # Max file size (default: 20 MB).
    request_timeout_seconds: int = field(default=30)  # Timeout for external API calls.

    # A `@classmethod` is a method that is bound to the class and not the instance of the class.
    # This is a factory method to create a `Settings` object from environment variables.
    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        env = os.environ  # Get the dictionary of environment variables.
        # `env.get("VAR_NAME", "default_value")` safely gets a variable, providing a fallback.
        return cls(
            openai_api_key=env.get("OPENAI_API_KEY"),
            qdrant_url=env.get("QDRANT_URL"),
            qdrant_api_key=env.get("QDRANT_API_KEY"),
            collection_name=env.get("COLLECTION_NAME", "mvp_docs"),
            upload_dir=Path(env.get("UPLOAD_DIR", "uploads")),
            log_level=env.get("LOG_LEVEL", "INFO"),
            # Convert megabytes from the env var into bytes.
            max_upload_bytes=int(env.get("MAX_UPLOAD_MB", "20")) * 1024 * 1024,
            request_timeout_seconds=int(env.get("REQUEST_TIMEOUT_SECONDS", "30")),
        )

    def ensure_upload_dir(self) -> None:
        """Ensure the upload directory exists."""
        # This method creates the directory specified in `self.upload_dir`.
        # `parents=True` means it will create any necessary parent directories.
        # `exist_ok=True` means it won't raise an error if the directory already exists.
        self.upload_dir.mkdir(parents=True, exist_ok=True)
