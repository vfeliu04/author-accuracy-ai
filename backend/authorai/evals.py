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


@dataclass
class VerdictScore:
    accuracy: float
    correct: int
    matched: int
    golden_total: int
    coverage: float
    per_class: dict[str, dict]
    confusion: dict[str, dict[str, int]]
    downgraded: int

    def summary(self) -> str:
        per_class = ", ".join(
            f"{cls} {stats['correct']}/{stats['total']}" for cls, stats in self.per_class.items()
        )
        return (
            f"verdict accuracy {self.accuracy:.2f} ({self.correct}/{self.matched} matched), "
            f"coverage {self.coverage:.2f} ({self.matched}/{self.golden_total} golden claims "
            f"had a verdict), per-class: {per_class}, downgraded: {self.downgraded}"
        )


def _pair_quality(golden: dict, row: dict) -> float:
    """How specifically a matching row fits this golden claim.

    Text overlap dominates; a value+year agreement adds a small bonus. This
    exists because _matches alone is a yes/no gate: two claims sharing a bare
    value (e.g. "fewer than a dozen" and "12% of Asia's rice", both value=12)
    both pass it, and for VERDICT scoring the pairing decides which
    expected_verdict a row is compared against — first-match pairing silently
    corrupted the first holdout reference (recorded 0.89, true 0.96).
    """
    quality = _overlap(_tokens(golden["text"]), _tokens(row["text"]))
    g_value, r_value = golden.get("value"), row.get("value")
    if g_value is not None and r_value is not None:
        if abs(g_value - r_value) <= abs(g_value) * 0.001:
            quality += 0.5
    return quality


def score_verdicts(verdict_rows: list[dict], golden: list[dict]) -> VerdictScore:
    """Compare stored verdicts against golden expected_verdict labels.

    Rows are matched to golden claims one-to-one; when several unconsumed rows
    match a golden claim, the BEST-fitting one (text overlap first) is paired —
    unlike extraction scoring, the pairing here determines which label each
    verdict is compared to, so first-match is not good enough.
    """
    consumed: set[int] = set()
    correct = 0
    matched = 0
    classes = sorted({g["expected_verdict"] for g in golden})
    confusion: dict[str, dict[str, int]] = {
        c: dict.fromkeys([*classes, "OTHER"], 0) for c in classes
    }
    per_class: dict[str, dict] = {c: {"total": 0, "correct": 0} for c in classes}

    for golden_claim in golden:
        expected = golden_claim["expected_verdict"]
        candidates = [
            i
            for i, row in enumerate(verdict_rows)
            if i not in consumed and _matches(golden_claim, row)
        ]
        if not candidates:
            continue
        hit = max(candidates, key=lambda i: _pair_quality(golden_claim, verdict_rows[i]))
        consumed.add(hit)
        matched += 1
        predicted = verdict_rows[hit]["verdict"]
        per_class[expected]["total"] += 1
        confusion[expected][predicted if predicted in classes else "OTHER"] += 1
        if predicted == expected:
            correct += 1
            per_class[expected]["correct"] += 1

    golden_total = len(golden)
    return VerdictScore(
        accuracy=correct / matched if matched else 0.0,
        correct=correct,
        matched=matched,
        golden_total=golden_total,
        coverage=matched / golden_total if golden_total else 0.0,
        per_class=per_class,
        confusion=confusion,
        # A downgrade is raw != final — quote_verified==0 alone also covers
        # raw-UNVERIFIABLE verdicts whose volunteered quote failed, which were
        # never downgraded.
        downgraded=sum(
            1 for row in verdict_rows if row.get("raw_verdict") not in (None, row.get("verdict"))
        ),
    )


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
