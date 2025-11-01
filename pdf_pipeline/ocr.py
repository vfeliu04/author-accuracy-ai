"""OCR integration using ocrmypdf."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from .config import pipeline_config, PDFPipelineConfig
from .router import RouteDecision


class OCRUnavailable(RuntimeError):
    pass


def ocr_if_needed(
    pdf_path: str,
    decision: RouteDecision,
    *,
    config: PDFPipelineConfig | None = None,
    force: bool = False,
) -> Tuple[Optional[str], bool]:
    cfg = config or pipeline_config
    if not cfg.ocr_enabled and not force:
        return None, False

    ocrmypdf_bin = shutil.which("ocrmypdf")
    if ocrmypdf_bin is None:
        if force:
            raise OCRUnavailable("ocrmypdf binary is not available on PATH.")
        return None, False

    if not force and not decision.features.get("is_scanned"):
        return None, False

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        output_path = handle.name

    command = [
        ocrmypdf_bin,
        "--skip-text",
        "--quiet",
        "--force-ocr",
        "--output-type",
        "pdf",
        pdf_path,
        output_path,
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        Path(output_path).unlink(missing_ok=True)
        if force:
            raise OCRUnavailable(f"OCR failed: {exc.stderr.decode('utf-8', 'ignore')}") from exc
        return None, False

    return output_path, True


__all__ = ["ocr_if_needed", "OCRUnavailable"]
