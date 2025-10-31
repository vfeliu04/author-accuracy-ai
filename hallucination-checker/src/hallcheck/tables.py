"""Utilities for detecting and rendering text-based tables."""

from __future__ import annotations

import html
import re
from typing import Iterable, List, Sequence

RANK_TABLE_HEADER = ["Rank", "Country", "2000", "2008", "2016", "2025"]
_VALUE_PATTERN = re.compile(r"^(?:<)?\d+(?:\.\d+)?$|^[—-]$")


def extract_tables_from_text(text: str) -> List[str]:
    """Return HTML tables parsed from a block of text."""
    tables: List[str] = []
    for lines in _segment_text_tables(text):
        html_table = _parse_fixed_width_table(lines)
        if html_table:
            tables.append(html_table)
            continue
        html_table = _parse_rank_style_table(" ".join(lines))
        if html_table:
            tables.append(html_table)
    return tables


def _segment_text_tables(text: str) -> List[List[str]]:
    chunks = re.split(r"\n\s*\n", text.strip())
    segments: List[List[str]] = []
    for chunk in chunks:
        lines = [line for line in chunk.splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) < 2:
            if "rank" in lines[0].lower():
                segments.append(lines)
            continue
        if not any("  " in line for line in lines) and "rank" not in lines[0].lower():
            continue
        segments.append(lines)
    return segments


def _parse_fixed_width_table(lines: Sequence[str]) -> str | None:
    header_tokens = _split_row(lines[0])
    if len(header_tokens) < 2:
        return None

    rows: List[List[str]] = []
    for line in lines[1:]:
        tokens = _split_row(line)
        if not tokens:
            continue
        aligned = _align_tokens(tokens, len(header_tokens))
        if len(aligned) != len(header_tokens):
            return None
        rows.append(aligned)

    if not rows:
        return None

    return _render_html_table(header_tokens, rows)


def _parse_rank_style_table(text: str) -> str | None:
    lowered = text.lower()
    if "rank" not in lowered or "country" not in lowered:
        return None

    body = text
    if "collectively ranked" in body:
        body = body.split("collectively ranked", 1)[1]

    body = re.sub(r"(Rank)(\d)", r"\1 \2", body)
    body = re.sub(r"(\d+)-(?=\d)", r"\1 ", body)

    tokens = body.split()
    rows: List[List[str]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.isdigit():
            rank = token
            i += 1
            country_tokens: List[str] = []
            values: List[str] = []
            while i < len(tokens) and len(values) < 4:
                tok = tokens[i]
                if _VALUE_PATTERN.match(tok):
                    values.append(tok)
                else:
                    country_tokens.append(tok)
                i += 1
            if len(values) == 4:
                country = " ".join(country_tokens).strip(",")
                rows.append([rank, country, *values])
        else:
            i += 1

    if not rows:
        return None

    return _render_html_table(RANK_TABLE_HEADER, rows)


def _split_row(line: str) -> List[str]:
    return [token.strip() for token in re.split(r"\s{2,}", line.strip()) if token.strip()]


def _align_tokens(tokens: List[str], target_len: int) -> List[str]:
    current = tokens[:]
    while len(current) > target_len:
        current[-2:] = [" ".join(current[-2:])]
    if len(current) < target_len:
        current.extend([""] * (target_len - len(current)))
    return current


def _render_html_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    head = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows)
    return (
        "<table class=\"evidence-table table table-striped table-sm\">"
        "<thead><tr>" + head + "</tr></thead>"
        "<tbody>" + body + "</tbody>"
        "</table>"
    )
