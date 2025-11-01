"""Helpers for formatting table text into displayable HTML."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Optional

from .config import settings
from .gpt import ensure_openai_client, OpenAIUnavailable, _run_model

CACHE_FILE = (settings.project_root or Path(__file__).resolve().parents[2]) / "data" / "cache" / "table_format_cache.json"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _cache_key(table_text: str) -> str:
    return hashlib.sha256(table_text.strip().encode("utf-8")).hexdigest()


def _load_cache() -> dict[str, str]:
    if not CACHE_FILE.exists():
        return {}
    try:
        with CACHE_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    try:
        with CACHE_FILE.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False, indent=2)
    except Exception:
        pass


def format_table_html(table_text: str, *, caption: Optional[str] = None, force_refresh: bool = False) -> Optional[str]:
    """Format raw table text into HTML using GPT with caching."""
    normalized = table_text.strip()
    if not normalized:
        return None
    cache = _load_cache()
    key = _cache_key(normalized)
    if not force_refresh and key in cache:
        return cache[key]

    try:
        client = ensure_openai_client()
    except OpenAIUnavailable:
        return None

    system_prompt = (
        "You are a meticulous document parsing assistant. Convert the raw table text into a clean HTML table. "
        "Preserve numeric values and header labels exactly. Do not add commentary. Respond ONLY with HTML for a single "
        "<table class=\"evidence-table table table-striped table-sm\"> element. Include a <caption> if a caption is provided. "
        "Use <thead> for the header row when possible and wrap remaining rows in <tbody>. Keep the order of rows and columns."
    )
    payload = {
        "table_text": normalized,
        "caption": caption or "",
        "instructions": (
            "Return a single HTML <table> element with semantic rows and columns. "
            "Keep column order as in the source text."
        ),
    }
    try:
        html = _run_model(
            client,
            model=settings.openai_chat_model,
            system_prompt=system_prompt,
            user_payload=payload,
            max_tokens=800,
        )
    except Exception:
        html = None

    if not html:
        return None

    html = html.strip()
    if "<table" not in html.lower():
        return None

    cache[key] = html
    _save_cache(cache)
    return html


def ensure_table_classes(html: str) -> str:
    match = re.search(r"<table\b[^>]*>", html, flags=re.IGNORECASE)
    if not match:
        return html
    tag = match.group(0)
    desired = "evidence-table table table-striped table-sm"
    if "class=" in tag:
        if desired in tag:
            return html
        updated = re.sub(
            r"class=\"",
            f'class="{desired} ',
            tag,
            count=1,
        )
    else:
        updated = tag[:-1] + f' class="{desired}">' 
    return html.replace(tag, updated, 1)
