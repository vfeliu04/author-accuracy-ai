import pytest
from pydantic import ValidationError

from veritas.claims.schema import Claim, Span


def _base_span() -> Span:
    return Span(start=0, end=1)


def test_claim_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Claim(type="statistic", text="1%", span=_base_span(), quantity=1.0, extra="nope")


def test_statistic_claim_instantiation() -> None:
    claim = Claim(
        type="statistic",
        text="50%",
        span=_base_span(),
        quantity=50.0,
        unit="%",
    )
    assert claim.quantity == 50.0


def test_ratio_claim_instantiation() -> None:
    claim = Claim(
        type="ratio",
        text="one in five",
        span=_base_span(),
        ratio=(1.0, 5.0),
    )
    assert claim.ratio == (1.0, 5.0)


def test_range_claim_instantiation() -> None:
    claim = Claim(
        type="range",
        text="between 10 and 12%",
        span=_base_span(),
        range=(10.0, 12.0),
    )
    assert claim.range == (10.0, 12.0)


def test_delta_claim_instantiation() -> None:
    claim = Claim(
        type="delta",
        text="up 2pp vs 2023",
        span=_base_span(),
        delta=2.0,
        delta_direction="up",
        baseline_time="2023",
        unit="pp",
    )
    assert claim.delta == 2.0
