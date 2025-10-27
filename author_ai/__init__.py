"""Top-level package for Author AI claim extraction."""

from author_ai.claims import Claim, ClaimType, Span, extract_claims

__all__ = ["Claim", "ClaimType", "Span", "extract_claims"]
__version__ = "0.1.0"
