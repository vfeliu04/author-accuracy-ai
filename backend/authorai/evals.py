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
    """Lowercase, with every run of non-alphanumerics collapsed to one space.

    Substituting a space rather than deleting matters: claim text extracted
    verbatim from the PDF carries line breaks mid-sentence, and deleting them
    welds words together ("hunger\\nrose" -> "hungerrose") so containment
    against the hand-written golden text silently fails and recall reads low.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# Function words say nothing about WHICH claim this is; keeping them would let
# two unrelated claims share "the/of/in" and drift over the threshold.
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or"
    " that the to was were with".split()
)

# Measured against the SHORTER claim, not the union: the extractor emits verbatim
# spans that carry surrounding context, so a golden claim sitting entirely inside
# a longer extraction should score 1.0 rather than be penalised for the extra
# words. This generalises the old substring rule, which only matched when the
# word ORDER also happened to line up.
OVERLAP_THRESHOLD = 0.6


def _tokens(text: str) -> set[str]:
    return {token for token in _normalize(text).split() if token not in _STOPWORDS}


def _overlap(golden_tokens: set[str], extracted_tokens: set[str]) -> float:
    if not golden_tokens or not extracted_tokens:
        return 0.0
    shared = len(golden_tokens & extracted_tokens)
    return shared / min(len(golden_tokens), len(extracted_tokens))


def _matches(golden: dict, extracted: dict) -> bool:
    """A golden claim counts as found if value+year line up, or the wording overlaps."""
    g_value, e_value = golden.get("value"), extracted.get("value")
    g_year, e_year = golden.get("year"), extracted.get("year")
    if g_value is not None and e_value is not None:
        values_match = abs(g_value - e_value) <= abs(g_value) * 0.001
        years_compatible = g_year is None or e_year is None or g_year == e_year
        if values_match and years_compatible:
            return True
    return _overlap(_tokens(golden["text"]), _tokens(extracted["text"])) >= OVERLAP_THRESHOLD


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
