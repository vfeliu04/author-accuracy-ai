"""Helpers for persisting uploaded files."""

from __future__ import annotations

import uuid
from pathlib import Path
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from ..config import get_settings


def save_upload(file: FileStorage, upload_id: str | None = None) -> tuple[str, Path]:
    settings = get_settings()
    upload_id = upload_id or str(uuid.uuid4())
    filename = secure_filename(file.filename or f"upload-{upload_id}.pdf")
    destination_dir = settings.data_root / "uploads" / upload_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / filename
    file.save(destination_path)
    return upload_id, destination_path


def build_file_url(upload_id: str) -> str:
    return f"/api/uploads/{upload_id}/file"
