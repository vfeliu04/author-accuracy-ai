"""
Domain models shared across services.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class SourceMetadata:
    source_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[str] = None
    publisher: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    confidence: str = "LOW"
    user_verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimEvidence:
    evidence_id: str
    source_id: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: Optional[str] = None
    table_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Claim:
    claim_id: str
    text: str
    verdict: str
    confidence: float
    confidence_band: str
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence: List[ClaimEvidence] = field(default_factory=list)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "confidence_band": self.confidence_band,
            "explanation": self.explanation,
        }


@dataclass
class ChatTurn:
    session_id: str
    role: str
    message: str
    timestamp: str = field(default_factory=_now_iso)
    context_ids: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class CredibilityScore:
    source_id: str
    score: float
    metadata_confidence: str
    components: Dict[str, float]
    last_refreshed_at: str = field(default_factory=_now_iso)


@dataclass
class ValidityScores:
    report_id: str
    overall: float
    coverage: float
    consistency: float
    methodology: float
    context: float
    recency: float
    diagnostics: Dict[str, Any]
    last_calculated_at: str = field(default_factory=_now_iso)
