import subprocess
from pathlib import Path

import fitz
import pytest

from pdf_pipeline.ocr import ocr_if_needed, OCRUnavailable
from pdf_pipeline.router import RouteDecision


def _pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "sample text")
    doc.save(path)
    doc.close()
    return str(path)


def test_ocr_skipped_when_binary_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("pdf_pipeline.ocr.shutil.which", lambda name: None)
    decision = RouteDecision(label="report", features={"is_scanned": True})
    pdf_path = _pdf(tmp_path)
    output, used = ocr_if_needed(pdf_path, decision)
    assert output is None
    assert not used


def test_ocr_runs_when_forced(monkeypatch, tmp_path):
    tmp_output = tmp_path / "ocr.pdf"

    def fake_run(command, check, stdout, stderr):
        Path(command[-1]).write_bytes(b"%PDF-1.4")
        class Result:
            returncode = 0
        return Result()

    monkeypatch.setattr("pdf_pipeline.ocr.shutil.which", lambda name: "/usr/bin/ocrmypdf")
    monkeypatch.setattr("pdf_pipeline.ocr.subprocess.run", fake_run)
    decision = RouteDecision(label="scanned", features={"is_scanned": True})
    pdf_path = _pdf(tmp_path)
    output, used = ocr_if_needed(pdf_path, decision, force=True)
    assert used
    assert output is not None
    assert Path(output).exists()
