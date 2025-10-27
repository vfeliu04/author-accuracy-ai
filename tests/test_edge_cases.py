from author_ai.claims.extract import extract_claims


def test_locale_decimal_parsing() -> None:
    text = "The rate was 1,2 per 100k in 2021."
    claims = extract_claims(text)
    assert len(claims) == 1
    claim = claims[0]
    assert claim.type == "statistic"
    assert claim.quantity == 1.2
    assert claim.unit == "per 100k"
    assert claim.time == "2021"


def test_en_dash_range_detection() -> None:
    text = "Admissions fell to 10–12 million."
    claims = extract_claims(text)
    assert len(claims) == 1
    claim = claims[0]
    assert claim.type == "range"
    assert claim.range == (10.0, 12.0)
    assert claim.unit == "million"


def test_ignore_phone_numbers() -> None:
    text = "Call us at 555-1234 for more information."
    claims = extract_claims(text)
    assert claims == []
