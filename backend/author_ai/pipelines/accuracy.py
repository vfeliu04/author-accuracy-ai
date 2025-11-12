"""
High-level accuracy pipeline coordinator.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import List, Dict, Any

from ..models import Claim, ClaimEvidence
from ..storage.database import Repository
from ..config import get_settings
from ..services.logger import setup_logger
from ..services.embedding import embed_texts
from ..services.vector_store import VectorStore
from .ingestion import IngestionPipeline


logger = setup_logger(__name__)


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class AccuracyPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.ingestion = IngestionPipeline()
        self.repo = Repository()
        self.vector_store = VectorStore("sources")

    def index_source(self, upload: dict) -> Dict[str, Any]:
        pdf_path = Path(upload["path"]).resolve()
        payload = self.ingestion.ingest(pdf_path, doc_id=upload["upload_id"], doc_type="SOURCE")
        self._index_chunks(payload["chunks"])
        logger.info("Indexed source %s with %d chunks", pdf_path, len(payload["chunks"]))
        return payload

    def verify_report(self, upload: dict) -> Dict[str, Any]:
        pdf_path = Path(upload["path"]).resolve()
        report_id = upload["upload_id"]
        payload = self.ingestion.ingest(pdf_path, doc_id=report_id, doc_type="REPORT")
        claims = self._extract_claims(payload["body_text"], report_id)
        evidence_rows = []
        for claim in claims:
            evidence_rows.extend(self._retrieve_evidence(claim))

        claim_dicts = [
            {
                "claim_id": claim.claim_id,
                "report_id": report_id,
                "text": claim.text,
                "verdict": claim.verdict,
                "confidence": claim.confidence,
                "confidence_band": claim.confidence_band,
                "explanation": claim.explanation,
                "metadata": claim.metadata,
            }
            for claim in claims
        ]
        self.repo.insert_claims(claim_dicts)
        if evidence_rows:
            self.repo.insert_evidence(evidence_rows)

        self._persist_claim_index(report_id, claims)
        self._persist_source_index(report_id, evidence_rows)

        supported = sum(1 for claim in claims if claim.verdict == "SUPPORTED")
        contradicted = sum(1 for claim in claims if claim.verdict == "CONTRADICTED")
        not_found = len(claims) - supported - contradicted
        accuracy_pct = (supported / len(claims) * 100) if claims else 0.0
        summary_text = (
            f"Accuracy is {accuracy_pct:.1f}% ({supported} of {len(claims)} claims supported, "
            f"{contradicted} contradicted, {not_found} not found)."
        )
        self.repo.update_document_metadata(
            report_id,
            {
                "accuracy_summary": summary_text,
                "accuracy_supported": supported,
                "accuracy_contradicted": contradicted,
                "accuracy_not_found": not_found,
            },
        )
        logger.info("Extracted %d claims from %s", len(claims), pdf_path)
        return {"report_id": report_id, "claims": [claim.to_summary() for claim in claims]}

    def _extract_claims(self, text: str, report_id: str) -> List[Claim]:
        sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
        claims: List[Claim] = []
        for sentence in sentences:
            if any(char.isdigit() for char in sentence):
                claim_id = str(uuid.uuid4())
                number_match = re.search(r"(-?\d[\d,\.]*)(?:\s?(percent|%|million|billion))?", sentence, re.IGNORECASE)
                year_match = re.search(r"(20\d{2}|19\d{2})", sentence)
                metadata = {
                    "primary_value": number_match.group(1) if number_match else None,
                    "units": (number_match.group(2) or "count") if number_match else None,
                    "year": year_match.group(1) if year_match else None,
                }
                claim = Claim(
                    claim_id=claim_id,
                    text=sentence,
                    verdict="NOT_EVALUATED",
                    confidence=0.4,
                    confidence_band="LOW",
                    explanation="Awaiting retrieval alignment.",
                    metadata=metadata,
                    evidence=[],
                )
                claims.append(claim)
        return claims

    def _index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        texts = [chunk["text"] for chunk in chunks]
        if not texts:
            return
        vectors = embed_texts(texts)
        metadata = [
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "snippet": chunk["text"][:400],
            }
            for chunk in chunks
        ]
        self.vector_store.add(vectors, metadata)

    def _retrieve_evidence(self, claim: Claim) -> List[Dict[str, Any]]:
        vectors = embed_texts([claim.text])
        if not vectors:
            return []
        hits = self.vector_store.search(vectors[0], top_k=5)
        evidence_rows = []
        if not hits:
            claim.verdict = "NOT_FOUND"
            claim.explanation = "No supporting evidence retrieved."
            return evidence_rows

        best = hits[0]
        score = best["score"]
        threshold = self.settings.retrieval_support_threshold
        verdict = "SUPPORTED" if score >= threshold else "NOT_FOUND"
        claim.verdict = verdict
        claim.confidence = float(min(0.99, max(0.05, score)))
        claim.confidence_band = self._band_from_confidence(claim.confidence)
        claim.explanation = (
            f"Retrieved evidence from {best['doc_id']} with similarity {score:.2f}."
            if verdict == "SUPPORTED"
            else "Similarity below support threshold."
        )

        for hit in hits:
            evidence_rows.append(
                {
                    "evidence_id": str(uuid.uuid4()),
                    "claim_id": claim.claim_id,
                    "source_id": hit["doc_id"],
                    "chunk_id": hit.get("chunk_id"),
                    "verdict_label": verdict if hit is best else "ALTERNATIVE",
                    "metadata": {"snippet": hit.get("snippet"), "score": hit["score"]},
                }
            )
        return evidence_rows

    @staticmethod
    def _band_from_confidence(value: float) -> str:
        if value >= 0.7:
            return "HIGH"
        if value >= 0.4:
            return "MEDIUM"
        return "LOW"

    def _persist_claim_index(self, report_id: str, claims: List[Claim]) -> None:
        if not claims:
            VectorStore(report_id, base_dir=self.settings.claim_vector_path).overwrite([], [])
            return
        texts = [
            f"Claim: {claim.text}\nVerdict: {claim.verdict}\nExplanation: {claim.explanation}"
            for claim in claims
        ]
        vectors = embed_texts(texts)
        metadata = [
            {
                "claim_id": claim.claim_id,
                "verdict": claim.verdict,
                "confidence": claim.confidence,
            }
            for claim in claims
        ]
        VectorStore(report_id, base_dir=self.settings.claim_vector_path).overwrite(vectors, metadata)

    def _persist_source_index(self, report_id: str, evidence_rows: List[Dict[str, Any]]) -> None:
        snippets: List[str] = []
        metadata: List[Dict[str, Any]] = []
        for row in evidence_rows:
            snippet = (row.get("metadata") or {}).get("snippet")
            if not snippet:
                continue
            snippets.append(snippet)
            metadata.append(
                {
                    "claim_id": row["claim_id"],
                    "source_id": row.get("source_id"),
                    "chunk_id": row.get("chunk_id"),
                    "verdict_label": row.get("verdict_label"),
                    "score": (row.get("metadata") or {}).get("score"),
                    "snippet": snippet[:600],
                }
            )
        store = VectorStore(report_id, base_dir=self.settings.source_vector_path)
        if not snippets:
            store.overwrite([], [])
            return
        vectors = embed_texts(snippets)
        store.overwrite(vectors, metadata)
