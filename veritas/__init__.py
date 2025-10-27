"""Top-level package for Veritas claim extraction."""

from veritas.claims import Claim, ClaimType, Span, extract_claims

__all__ = ["Claim", "ClaimType", "Span", "extract_claims"]
__version__ = "0.1.0"
