"""Top-level package for Author AI claim extraction and verification."""

from author_ai.claims import Claim, ClaimType, Span, extract_claims
from author_ai.pipeline import PipelineOutput, VerificationPipeline

__all__ = [
    "Claim",
    "ClaimType",
    "Span",
    "extract_claims",
    "VerificationPipeline",
    "PipelineOutput",
]
__version__ = "0.1.0"
