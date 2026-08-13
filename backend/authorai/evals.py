"""Eval scoring: compare pipeline output against the golden set.

Phase 3 scores extraction (did we find the golden claims?). Phase 4 adds
verdict scoring against the same file's `expected_verdict` labels.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from authorai import db as dbmod


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


# Relative tolerance for "these are the same number" — used by BOTH the match
# gate and the pairing tie-breaker; two tolerances would let a row pass the
# gate yet score no pairing bonus, silently reordering pairs.
VALUE_TOLERANCE = 0.001


def _values_match(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) <= abs(a) * VALUE_TOLERANCE


def _tokens(text: str) -> set[str]:
    return {token for token in _normalize(text).split() if token not in _STOPWORDS}


def _overlap(golden_tokens: set[str], extracted_tokens: set[str]) -> float:
    if not golden_tokens or not extracted_tokens:
        return 0.0
    shared = len(golden_tokens & extracted_tokens)
    return shared / min(len(golden_tokens), len(extracted_tokens))


def _matches(golden: dict, extracted: dict) -> bool:
    """A golden claim counts as found if value+year line up, or the wording overlaps."""
    g_year, e_year = golden.get("year"), extracted.get("year")
    years_compatible = g_year is None or e_year is None or g_year == e_year
    if _values_match(golden.get("value"), extracted.get("value")) and years_compatible:
        return True
    return _overlap(_tokens(golden["text"]), _tokens(extracted["text"])) >= OVERLAP_THRESHOLD


def _pair(golden: list[dict], rows: list[dict], quality=None):
    """One-to-one pairing of golden claims to rows; yields (golden_claim, row_index|None).

    The consume-each-row-once rule lives HERE, once, for both scorers. With
    `quality`, ties among matching rows go to the best-fitting one (verdict
    scoring — the pairing decides which label a row is compared to); without
    it, first match wins (extraction scoring — pairing-insensitive counting,
    and the recorded baselines were produced under first-match; max() returns
    the first maximum, so a constant key reproduces it exactly).
    """
    consumed: set[int] = set()
    for golden_claim in golden:
        candidates = [
            i for i, row in enumerate(rows) if i not in consumed and _matches(golden_claim, row)
        ]
        if not candidates:
            yield golden_claim, None
            continue
        hit = (
            max(candidates, key=lambda i: quality(golden_claim, rows[i]))
            if quality
            else candidates[0]
        )
        consumed.add(hit)
        yield golden_claim, hit


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

    Text overlap dominates; a value agreement adds a small bonus. This exists
    because _matches alone is a yes/no gate: two claims sharing a bare value
    (e.g. "fewer than a dozen" and "12% of Asia's rice", both value=12) both
    pass it, and for VERDICT scoring the pairing decides which
    expected_verdict a row is compared against — first-match pairing silently
    corrupted the first holdout reference (recorded 0.89, true 0.96).
    """
    quality = _overlap(_tokens(golden["text"]), _tokens(row["text"]))
    if _values_match(golden.get("value"), row.get("value")):
        quality += 0.5
    return quality


def score_verdicts(verdict_rows: list[dict], golden: list[dict]) -> VerdictScore:
    """Compare stored verdicts against golden expected_verdict labels.

    Pairing is one-to-one, best-fit (see _pair) — unlike extraction scoring,
    the pairing here determines which label each verdict is compared to.
    """
    correct = 0
    matched = 0
    classes = sorted({g["expected_verdict"] for g in golden})
    confusion: dict[str, dict[str, int]] = {
        c: dict.fromkeys([*classes, "OTHER"], 0) for c in classes
    }
    per_class: dict[str, dict] = {c: {"total": 0, "correct": 0} for c in classes}

    for golden_claim, hit in _pair(golden, verdict_rows, quality=_pair_quality):
        if hit is None:
            continue
        expected = golden_claim["expected_verdict"]
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
        downgraded=sum(1 for row in verdict_rows if dbmod.is_downgraded(row)),
    )


@dataclass
class StanceScore:
    agreed: int
    labeled: int  # golden records carrying a stance label that paired to a row
    disagreements: list[str]

    def summary(self) -> str:
        return f"stance agreement {self.agreed}/{self.labeled} paired golden claims"


def score_stance(extracted: list[dict], golden: list[dict]) -> StanceScore:
    """Stance agreement over best-fit pairs; cannot perturb recall/precision.

    Separate from score_extraction on purpose: recall/precision keep their
    recorded first-match pairing, but stance comparison needs the best-fitting
    row — the real-vs-fabricated table twins both pass _matches, exactly the
    collision class that corrupted the first holdout verdict reference.
    Golden records without a stance label are skipped, not defaulted.
    """
    agreed = 0
    labeled = 0
    disagreements: list[str] = []
    for golden_claim, hit in _pair(golden, extracted, quality=_pair_quality):
        expected = golden_claim.get("stance")
        if expected is None or hit is None:
            continue
        labeled += 1
        actual = extracted[hit].get("stance") or "asserted"
        if actual == expected:
            agreed += 1
        else:
            disagreements.append(
                f"{golden_claim['text'][:80]!r}: expected {expected}, extracted {actual}"
            )
    return StanceScore(agreed=agreed, labeled=labeled, disagreements=disagreements)


def score_extraction(extracted: list[dict], golden: list[dict]) -> ExtractionScore:
    matched = 0
    missed: list[str] = []
    for golden_claim, hit in _pair(golden, extracted):
        if hit is None:
            missed.append(golden_claim["text"])
        else:
            matched += 1
    golden_total = len(golden)
    extracted_total = len(extracted)
    return ExtractionScore(
        recall=matched / golden_total if golden_total else 0.0,
        precision=matched / extracted_total if extracted_total else 0.0,
        matched=matched,
        golden_total=golden_total,
        extracted_total=extracted_total,
        missed=missed,
    )
