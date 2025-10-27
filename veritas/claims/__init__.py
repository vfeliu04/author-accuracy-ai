"""Public API for claim extraction utilities."""

from veritas.claims.extract import extract_claims
from veritas.claims.schema import Claim, ClaimType, Span

__all__ = ["Claim", "ClaimType", "Span", "extract_claims"]
