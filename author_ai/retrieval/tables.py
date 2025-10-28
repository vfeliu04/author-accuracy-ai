"""Table handling utilities used by Stage B.

We do not depend on heavy PDF extraction libraries inside this repo.  Instead
we look for simple delimiter-based structures (pipes or tabs) and treat each
line as a pseudo-sentence representing the row.  This allows us to keep the
interface consistent while still supporting table evidence in later stages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


ROW_PATTERN = re.compile(r"\|+|\t+")


@dataclass
class TableRow:
    doc_id: str
    row_id: str
    content: str


class TableExtractor:
    """Scan text for delimiter-heavy lines and expose them as table rows."""

    def extract(self, doc_id: str, content: str) -> List[TableRow]:
        rows: list[TableRow] = []
        for idx, line in enumerate(content.splitlines()):
            if not ROW_PATTERN.search(line):
                continue
            cleaned = line.strip()
            if not cleaned:
                continue
            row_id = f"{doc_id}-row-{idx}"
            rows.append(TableRow(doc_id=doc_id, row_id=row_id, content=cleaned))
        return rows


__all__ = ["TableExtractor", "TableRow"]
