import pytest

from veritas.claims.extract import extract_claims


def _get_claim(claims, claim_type):
    for claim in claims:
        if claim.type == claim_type:
            return claim
    return None


def test_statistic_and_delta_extraction() -> None:
    text = (
        "About 23.5% of UK A&E attendances in Q2 2024 breached the four-hour standard, "
        "up 2pp vs 2023."
    )
    claims = extract_claims(text)
    assert len(claims) >= 2

    statistic = _get_claim(claims, "statistic")
    assert statistic is not None
    assert pytest.approx(statistic.quantity, rel=1e-3) == 23.5
    assert statistic.unit == "%"
    assert statistic.qualifier == "about"
    assert statistic.time == "2024-Q2"
    assert text[statistic.span.start : statistic.span.end] == statistic.text

    delta = _get_claim(claims, "delta")
    assert delta is not None
    assert pytest.approx(delta.delta, rel=1e-3) == 2.0
    assert delta.unit == "pp"
    assert delta.delta_direction == "up"
    assert delta.baseline_time == "2023"
    assert text[delta.span.start : delta.span.end] == delta.text


def test_ratio_extraction() -> None:
    text = "Roughly one in five discharges were delayed in 2022."
    claims = extract_claims(text)
    ratio = _get_claim(claims, "ratio")
    assert ratio is not None
    assert ratio.ratio == (1.0, 5.0)
    assert ratio.qualifier == "roughly"
    assert ratio.time == "2022"
    assert text[ratio.span.start : ratio.span.end] == ratio.text


def test_range_extraction() -> None:
    text = "Between 20 and 25% of referrals were triaged."
    claims = extract_claims(text)
    range_claim = _get_claim(claims, "range")
    assert range_claim is not None
    assert range_claim.range == (20.0, 25.0)
    assert range_claim.unit == "%"
    assert text[range_claim.span.start : range_claim.span.end] == range_claim.text
