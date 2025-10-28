"""Stage A tests: conversion of legacy claims to new contract."""

from __future__ import annotations

from author_ai.pipeline import VerificationPipeline, _convert_legacy_claim, _enumerate_sentences
from author_ai.claims.schema import Claim as LegacyClaim, Span


def _legacy_statistic() -> LegacyClaim:
    return LegacyClaim(
        type="statistic",
        text="Inflation was 6% in 2023.",
        span=Span(start=0, end=26),
        quantity=6.0,
        unit="%",
        time="2023",
        subject="Inflation",
        location="UK",
    )


def test_convert_legacy_claim_populates_new_fields() -> None:
    legacy = _legacy_statistic()
    claim = _convert_legacy_claim(legacy, claim_index=1, sentence_id="sent-1")
    assert claim.id == "claim-1"
    assert claim.kind == "statistic"
    assert claim.canonical["unit"] == "%"
    assert claim.span == {"start": 0, "end": 26}


def test_stage_a_sentence_mapping_handles_single_sentence() -> None:
    pipeline = VerificationPipeline()
    sentences = _enumerate_sentences("Just one sentence.")
    claim = _convert_legacy_claim(_legacy_statistic(), 1, sentence_id=sentences[0][0])
    assert claim.sentence_id == "sent-1"
