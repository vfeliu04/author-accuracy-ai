"""
Helpers for extracting tables using Tabula or PDFTables API.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import requests
import os

from ..config import get_settings
from .logger import setup_logger


logger = setup_logger(__name__)


def extract_tables(path: Path) -> List[Dict[str, Any]]:
    settings = get_settings()
    if settings.table_engine == "tabula":
        return _extract_with_tabula(path)
    if settings.table_engine == "pdftables":
        return _extract_with_pdf_tables(path)
    raise ValueError(f"Unsupported table engine: {settings.table_engine}")


def _extract_with_tabula(path: Path) -> List[Dict[str, Any]]:
    settings = get_settings()
    jar_path = Path(settings.tabula_jar_path)
    if not jar_path.exists():
        logger.warning("Tabula jar not found at %s; skipping table extraction.", jar_path)
        return []
    extra_cp = settings.tabula_extra_classpath.strip()
    if extra_cp:
        classpath = os.pathsep.join([segment for segment in [extra_cp, str(jar_path)] if segment])
        cmd = [
            "java",
            "-cp",
            classpath,
            "technology.tabula.CommandLineApp",
            "--pages",
            "all",
            "--format",
            "JSON",
            str(path),
        ]
    else:
        cmd = [
            "java",
            "-jar",
            str(jar_path),
            "--pages",
            "all",
            "--format",
            "JSON",
            str(path),
        ]
    output = subprocess.check_output(cmd)
    tables = json.loads(output.decode("utf-8"))
    logger.info("Tabula extracted %d tables from %s", len(tables), path)
    return tables


def _extract_with_pdf_tables(path: Path) -> List[Dict[str, Any]]:
    settings = get_settings()
    if not settings.pdf_tables_api_key:
        logger.warning("PDFTables API key missing; skipping table extraction.")
        return []

    files = {"file": path.open("rb")}
    response = requests.post(
        "https://pdftables.com/api?format=JSON",
        files=files,
        auth=(settings.pdf_tables_api_key, ""),
        timeout=float(os.getenv("PDF_TABLES_TIMEOUT", 30)),
    )
    response.raise_for_status()
    tables = response.json()
    logger.info("PDFTables returned %d tables for %s", len(tables), path)
    return tables
