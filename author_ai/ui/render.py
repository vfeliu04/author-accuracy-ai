"""HTML + JSON rendering utilities for Stage E."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List

from author_ai.config import StageEConfig
from author_ai.models import Claim, EvidenceSpan, Score, VerificationResult


@dataclass
class RenderedReport:
    """Bundle returned by Stage E."""

    html: str
    jsonl: str


class ReportBuilder:
    """Generate a Turnitin-style report with inline highlights + evidence cards."""

    def __init__(self, config: StageEConfig) -> None:
        self.config = config

    def build(
        self,
        original_text: str,
        claims: Iterable[Claim],
        verifications: Iterable[VerificationResult],
        scores: Iterable[Score],
        evidence_map: dict[str, list[EvidenceSpan]],
    ) -> RenderedReport:
        """Render the HTML + JSONL artefacts required by the CLI."""

        claim_list = list(claims)
        verification_by_id = {item.claim_id: item for item in verifications}
        score_by_id = {item.claim_id: item for item in scores}

        highlighted = self._highlight_text(original_text, claim_list)
        cards = [
            self._render_card(
                claim,
                verification_by_id.get(claim.id),
                score_by_id.get(claim.id),
                evidence_map.get(claim.id, []),
            )
            for claim in claim_list
        ]

        html = "\n".join(
            [
                "<html>",
                "<head><meta charset='utf-8'><title>author.ai Verification Report</title></head>",
                "<body>",
                "<h1>Verification report</h1>",
                "<section>",
                "<h2>Text with highlights</h2>",
                f"<p>{highlighted}</p>",
                "</section>",
                "<section>",
                "<h2>Evidence cards</h2>",
                *cards,
                "</section>",
                "</body>",
                "</html>",
            ]
        )

        json_lines = []
        for claim in claim_list:
            payload = {
                "claim": claim.model_dump(),
                "verification": verification_by_id.get(claim.id).model_dump()
                if claim.id in verification_by_id
                else None,
                "score": score_by_id.get(claim.id).model_dump() if claim.id in score_by_id else None,
                "evidence": [span.model_dump() for span in evidence_map.get(claim.id, [])],
            }
            json_lines.append(json.dumps(payload, ensure_ascii=False))

        return RenderedReport(html=html, jsonl="\n".join(json_lines))

    def _highlight_text(self, text: str, claims: List[Claim]) -> str:
        """Wrap the claim spans with the configured highlight tag."""

        if not claims:
            return text

        tag = self.config.highlight_tag
        pieces: list[str] = []
        last_index = 0

        for claim in sorted(claims, key=lambda c: c.span.get("start", 0)):
            start = int(claim.span.get("start") or 0)
            end = int(claim.span.get("end") or start)
            start = max(start, last_index)
            end = max(end, start)
            start = min(start, len(text))
            end = min(end, len(text))
            pieces.append(text[last_index:start])
            snippet = text[start:end]
            aria = f" data-claim-id='{claim.id}' data-label='{claim.kind}'"
            pieces.append(f"<{tag}{aria}>{snippet}</{tag}>")
            last_index = end
        pieces.append(text[last_index:])
        return "".join(pieces)

    def _render_card(
        self,
        claim: Claim,
        verification: VerificationResult | None,
        score: Score | None,
        evidence: Iterable[EvidenceSpan],
    ) -> str:
        """Render one evidence card containing status + snippets."""

        evidence_items = [
            f"<li><strong>{span.doc_id}</strong>: {span.content}</li>" for span in evidence
        ]
        evidence_html = "<ul>" + "".join(evidence_items) + "</ul>" if evidence_items else "<p>No evidence</p>"
        label = verification.label if verification else "unverified"
        score_text = f"{score.score_0_100}" if score else "N/A"
        return (
            "<article class='card'>"
            f"<h3>{claim.text}</h3>"
            f"<p><strong>Label:</strong> {label}</p>"
            f"<p><strong>Score:</strong> {score_text}</p>"
            f"<h4>Evidence</h4>{evidence_html}"
            "</article>"
        )


__all__ = ["ReportBuilder", "RenderedReport"]
