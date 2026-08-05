"""Eval scoring: compare pipeline output against the golden set.

Phase 3 scores extraction (did we find the golden claims?). Phase 4 adds
verdict scoring against the same file's `expected_verdict` labels.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractionScore:
    recall: float
    precision: float
    matched: int
    golden_total: int
    extracted_total: int
    missed: list[str]

    def summary(self) -> str:
        return (
            f"extraction recall {self.recall:.2f} "
            f"({self.matched}/{self.golden_total} golden found), "
            f"precision {self.precision:.2f} "
            f"({self.matched}/{self.extracted_total} extracted matched)"
        )


def load_golden(path: Path | str) -> list[dict]:
    lines = Path(path).read_text().strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower())


def _matches(golden: dict, extracted: dict) -> bool:
    """A golden claim counts as found if value+year line up, or the texts overlap."""
    g_value, e_value = golden.get("value"), extracted.get("value")
    g_year, e_year = golden.get("year"), extracted.get("year")
    if g_value is not None and e_value is not None:
        values_match = abs(g_value - e_value) <= abs(g_value) * 0.001
        years_compatible = g_year is None or e_year is None or g_year == e_year
        if values_match and years_compatible:
            return True
    g_text, e_text = _normalize(golden["text"]), _normalize(extracted["text"])
    return bool(g_text and e_text) and (g_text in e_text or e_text in g_text)


def score_extraction(extracted: list[dict], golden: list[dict]) -> ExtractionScore:
    matched_extracted: set[int] = set()
    matched = 0
    missed: list[str] = []
    for golden_claim in golden:
        hit = next(
            (
                i
                for i, candidate in enumerate(extracted)
                if i not in matched_extracted and _matches(golden_claim, candidate)
            ),
            None,
        )
        if hit is None:
            missed.append(golden_claim["text"])
        else:
            matched_extracted.add(hit)
            matched += 1
    golden_total = len(golden)
    extracted_total = len(extracted)
    return ExtractionScore(
        recall=matched / golden_total if golden_total else 0.0,
        precision=len(matched_extracted) / extracted_total if extracted_total else 0.0,
        matched=matched,
        golden_total=golden_total,
        extracted_total=extracted_total,
        missed=missed,
    )
