from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


# A set of allowed MIME types for PDF files. This helps validate the file type.
PDF_MIME_TYPES = {"application/pdf"}


# A custom exception class for file-related errors.
# This makes error handling in the main app more specific.
class FileValidationError(ValueError):
    """Raised when uploaded files fail validation."""


# A helper function to check if a file is a PDF based on its name and MIME type.
def is_pdf(filename: str, content_type: str | None = None) -> bool:
    """Return True if the filename and content type represent a PDF."""
    # Check if the filename ends with ".pdf" (case-insensitive).
    # Also, check if the content type is one of the allowed PDF types.
    return filename.lower().endswith(".pdf") and (
        content_type in (None, *PDF_MIME_TYPES)
    )


# This function handles the process of saving an uploaded file to the server.
def save_upload(file: FileStorage, upload_dir: Path, max_bytes: int) -> Tuple[Path, str]:
    """Persist an uploaded file to disk and return its path and SHA256 hash."""
    # `secure_filename` is a Werkzeug utility that makes the filename safe for the filesystem.
    filename = secure_filename(file.filename or "")
    if not filename:
        raise FileValidationError("Uploaded file is missing a filename.")
    if not is_pdf(filename, file.mimetype):
        raise FileValidationError(f"Only PDF uploads are supported: {filename}")

    # Ensure the target directory exists.
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / filename

    # Read the file content into memory.
    data = file.read()
    # Check if the file size exceeds the configured limit.
    if len(data) > max_bytes:
        raise FileValidationError("Uploaded file exceeds maximum size (20 MB).")

    # Calculate the SHA256 hash of the file content for identification and logging.
    sha256 = hashlib.sha256(data).hexdigest()
    # Write the file data to the destination path.
    destination.write_bytes(data)
    return destination, sha256


# This function deletes a file from the filesystem.
def delete_file(path: Path) -> None:
    """Delete a file, ignoring missing files."""
    try:
        # `missing_ok=True` prevents an error if the file doesn't exist.
        path.unlink(missing_ok=True)
    except PermissionError as exc:
        # If we can't delete the file due to permissions, raise our custom error.
        raise FileValidationError(f"Unable to delete temporary file: {path}") from exc
