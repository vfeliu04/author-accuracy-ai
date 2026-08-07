import json

from authorai.evals import load_golden, score_extraction

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


def test_load_golden_parses_jsonl(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(entry) for entry in GOLDEN))
    assert load_golden(path) == GOLDEN
