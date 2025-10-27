"""Strict Pydantic schema definitions for claim extraction."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

ClaimType = Literal["statistic", "ratio", "range", "delta"]


class Span(BaseModel):
    """Inclusive-exclusive span in the original text."""

    model_config = ConfigDict(extra="forbid")

    start: int
    end: int


class Claim(BaseModel):
    """Canonical representation of a quantitative claim."""

    model_config = ConfigDict(extra="forbid")

    type: ClaimType
    text: str
    span: Span
    quantity: Optional[float] = None
    unit: Optional[str] = None
    subject: Optional[str] = None
    population: Optional[str] = None
    time: Optional[str] = None
    location: Optional[str] = None
    qualifier: Optional[str] = None
    ratio: Optional[Tuple[float, float]] = None
    range: Optional[Tuple[float, float]] = None
    delta: Optional[float] = None
    delta_direction: Optional[Literal["up", "down"]] = None
    baseline_time: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


__all__ = ["Claim", "Span", "ClaimType"]
