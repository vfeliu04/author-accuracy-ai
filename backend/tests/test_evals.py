import json

import pytest

from authorai.evals import load_golden, score_extraction, score_stance

GOLDEN = [
    {"text": "Hunger affected 735 million people in 2023.", "value": 735e6, "year": 2023},
    {"text": "Wheat exports fell by 12 percent.", "value": 12.0, "year": None},
    {"text": "The report claims food aid doubled.", "value": None, "year": None},
]


def test_value_year_match_wins_even_with_different_wording():
    extracted = [
        {"text": "In 2023, 735 million people faced hunger.", "value": 735e6, "year": 2023}
    ]
    score = score_extraction(extracted, GOLDEN[:1])
    assert score.matched == 1
    assert score.recall == 1.0
    assert score.precision == 1.0


def test_text_containment_fallback_for_valueless_claims():
    extracted = [{"text": "The report claims food aid doubled.", "value": None, "year": None}]
    score = score_extraction(extracted, GOLDEN[2:])
    assert score.matched == 1


def test_missed_and_precision_accounting():
    extracted = [
        {"text": "In 2023, 735 million people faced hunger.", "value": 735e6, "year": 2023},
        {"text": "Something entirely unrelated happened.", "value": 42.0, "year": 1999},
    ]
    score = score_extraction(extracted, GOLDEN)
    assert score.matched == 1
    assert score.recall == 1 / 3
    assert score.precision == 1 / 2
    assert "Wheat exports fell by 12 percent." in score.missed


def test_score_stance_agreement_and_disagreements():
    golden = [
        {
            "text": "Hunger was eradicated in 2019.",
            "value": None,
            "year": 2019,
            "stance": "disavowed",
        },
        {"text": "Conflict drives hunger.", "value": None, "year": None, "stance": "asserted"},
        {"text": "No stance label on this one.", "value": None, "year": None},  # skipped
    ]
    extracted = [
        {
            "text": "Hunger was eradicated in 2019.",
            "value": None,
            "year": 2019,
            "stance": "disavowed",
        },
        {"text": "Conflict drives hunger.", "value": None, "year": None, "stance": "disavowed"},
        {"text": "No stance label on this one.", "value": None, "year": None, "stance": "asserted"},
    ]
    score = score_stance(extracted, golden)
    assert score.labeled == 2  # the unlabeled golden record is skipped, not defaulted
    assert score.agreed == 1
    assert score.unpaired == 0
    [disagreement] = score.disagreements
    assert "expected asserted, extracted disavowed" in disagreement


def test_score_stance_reports_unpaired_labeled_records():
    # A labeled golden claim that matches nothing must be COUNTED as unpaired,
    # not silently dropped — a shrinking denominator flatters the ratio.
    golden = [
        {
            "text": "Wheat exports fell by 12 percent.",
            "value": 12.0,
            "year": None,
            "stance": "asserted",
        },
    ]
    score = score_stance([], golden)
    assert score.labeled == 0
    assert score.unpaired == 1
    assert "UNPAIRED" in score.summary()


def test_score_stance_pairs_table_twins_by_best_fit():
    # The real-vs-fabricated table twins share enough tokens to both pass
    # _matches; first-match pairing would compare stances across the wrong
    # pair. Best-fit pairing must line each golden up with its own row.
    golden = [
        {
            "text": "Global total: 733 million undernourished.",
            "value": 733e6,
            "year": None,
            "stance": "asserted",
        },
        {
            "text": "Global total: only 10 million undernourished.",
            "value": 10e6,
            "year": None,
            "stance": "disavowed",
        },
    ]
    extracted = [
        {
            "text": "Global total: only 10 million undernourished.",
            "value": 10e6,
            "year": None,
            "stance": "disavowed",
        },
        {
            "text": "Global total: 733 million undernourished.",
            "value": 733e6,
            "year": None,
            "stance": "asserted",
        },
    ]
    score = score_stance(extracted, golden)
    assert score.labeled == 2
    assert score.agreed == 2
    assert score.disagreements == []


def test_each_extracted_claim_matches_at_most_one_golden():
    # Two golden entries with the same value must not both match one extraction.
    golden = [
        {"text": "A says 10 things.", "value": 10.0, "year": None},
        {"text": "B says 10 items.", "value": 10.0, "year": None},
    ]
    extracted = [{"text": "It mentions 10.", "value": 10.0, "year": None}]
    score = score_extraction(extracted, golden)
    assert score.matched == 1
    assert len(score.missed) == 1


def test_pdf_line_breaks_do_not_defeat_text_containment():
    # Claim text is extracted VERBATIM from the PDF, so it carries line breaks
    # mid-sentence. Normalizing must turn them into spaces, not delete them —
    # deleting welds words together and the golden claim reads as MISSED.
    extracted = [{"text": "The report claims food\naid  doubled.", "value": None, "year": None}]
    score = score_extraction(extracted, GOLDEN[2:])
    assert score.matched == 1
    assert score.missed == []


def test_reordered_wording_matches():
    # Verbatim from the first live run: substring containment scored this a MISS
    # because neither text contains the other, even though it is plainly the
    # same claim. This is the case the overlap matcher exists to catch.
    golden = [
        {
            "text": "Coastal communities are immune to malnutrition according to the"
            " 'Ocean Nutrient Absorption Theory.'",
            "value": None,
            "year": None,
        }
    ]
    extracted = [
        {
            "text": "a flawed (and fictional) 'Ocean Nutrient Absorption Theory' claims"
            " that coastal communities are immune to malnutrition",
            "value": None,
            "year": None,
        }
    ]
    score = score_extraction(extracted, golden)
    assert score.matched == 1
    assert score.missed == []


def test_unrelated_claims_sharing_only_filler_words_do_not_match():
    # The guard on the looser matcher: two claims about entirely different things
    # must not match just because both are English sentences about hunger.
    golden = [{"text": "Wheat exports fell by 12 percent.", "value": None, "year": None}]
    extracted = [
        {
            "text": "Armed conflict remains the leading driver of hunger.",
            "value": None,
            "year": None,
        }
    ]
    score = score_extraction(extracted, golden)
    assert score.matched == 0


def test_score_verdicts_accuracy_confusion_and_coverage():
    from authorai.evals import score_verdicts

    golden = [
        {
            "text": "Hunger affected 735 million people.",
            "value": 735e6,
            "year": None,
            "expected_verdict": "SUPPORTED",
        },
        {
            "text": "Only 10 million are undernourished.",
            "value": 10e6,
            "year": None,
            "expected_verdict": "CONTRADICTED",
        },
        {
            "text": "The 1923 treaty ended famine.",
            "value": None,
            "year": 1923,
            "expected_verdict": "UNVERIFIABLE",
        },
    ]
    rows = [
        # Correct SUPPORTED, quote verified.
        {
            "text": "Hunger affected 735 million people.",
            "value": 735e6,
            "year": None,
            "verdict": "SUPPORTED",
            "raw_verdict": "SUPPORTED",
            "quote_verified": 1,
        },
        # Wrong: golden says CONTRADICTED, pipeline downgraded to UNVERIFIABLE
        # after a failed quote check (raw != final -> counts as downgraded).
        {
            "text": "Only 10 million are undernourished.",
            "value": 10e6,
            "year": None,
            "verdict": "UNVERIFIABLE",
            "raw_verdict": "CONTRADICTED",
            "quote_verified": 0,
        },
        # No row matches the 1923 golden claim -> coverage gap, not an error.
    ]
    score = score_verdicts(rows, golden)
    assert score.matched == 2
    assert score.correct == 1
    assert score.accuracy == 0.5
    assert score.coverage == pytest.approx(2 / 3)
    assert score.confusion["CONTRADICTED"]["UNVERIFIABLE"] == 1
    assert score.confusion["SUPPORTED"]["SUPPORTED"] == 1
    assert score.per_class["SUPPORTED"] == {"total": 1, "correct": 1}
    assert score.downgraded == 1
    assert "accuracy 0.50" in score.summary()


def test_score_verdicts_pairs_by_best_fit_not_first_match():
    from authorai.evals import score_verdicts

    # Regression for the corrupted first holdout reference: two claims share
    # value=12 with OPPOSITE expected verdicts. First-match pairing crossed
    # them (row order is uuid-lexicographic within a page, so nondeterministic)
    # and scored a fully-correct pipeline 0/2.
    golden = [
        {
            "text": "Fewer than a dozen countries worldwide currently show serious"
            " or alarming hunger.",
            "value": 12,
            "year": None,
            "expected_verdict": "CONTRADICTED",
        },
        {
            "text": "A Pacific 'floating farm corridor' supplies 12% of Asia's rice.",
            "value": 12,
            "year": None,
            "expected_verdict": "UNVERIFIABLE",
        },
    ]
    rows = [
        # Deliberately listed corridor-first — the order that broke pairing.
        {
            "text": "Industry newsletters add that a Pacific 'floating farm corridor' now"
            " supplies 12% of Asia's rice.",
            "value": 12,
            "year": None,
            "verdict": "UNVERIFIABLE",
            "raw_verdict": "UNVERIFIABLE",
            "quote_verified": None,
        },
        {
            "text": "Contrary press coverage has claimed that fewer than a dozen countries"
            " worldwide currently show serious or alarming hunger.",
            "value": 12,
            "year": None,
            "verdict": "CONTRADICTED",
            "raw_verdict": "CONTRADICTED",
            "quote_verified": 1,
        },
    ]
    score = score_verdicts(rows, golden)
    assert score.correct == 2
    assert score.accuracy == 1.0
    assert score.downgraded == 0


def test_score_verdicts_consumes_each_row_once():
    from authorai.evals import score_verdicts

    golden = [
        {"text": "A says 10 things.", "value": 10.0, "year": None, "expected_verdict": "SUPPORTED"},
        {"text": "B says 10 items.", "value": 10.0, "year": None, "expected_verdict": "SUPPORTED"},
    ]
    rows = [
        {
            "text": "It mentions 10.",
            "value": 10.0,
            "year": None,
            "verdict": "SUPPORTED",
            "quote_verified": 1,
        }
    ]
    score = score_verdicts(rows, golden)
    assert score.matched == 1  # one row cannot satisfy two golden claims


def test_load_golden_parses_jsonl(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(entry) for entry in GOLDEN))
    assert load_golden(path) == GOLDEN
