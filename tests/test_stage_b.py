"""Stage B tests covering the hybrid retrieval pipeline."""

from __future__ import annotations

from author_ai.config import StageBConfig
from author_ai.models import Claim
from author_ai.retrieval import HybridRetrievalPipeline


def _sample_claim() -> Claim:
    return Claim(
        id="claim-1",
        sentence_id="sent-1",
        text="Unemployment was 4.2% in March 2024.",
        is_statistic=True,
        kind="statistic",
        values=[{"value": 4.2, "unit": "%"}],
        time="March 2024",
        span={"start": 0, "end": 36},
        verbatim="Unemployment was 4.2% in March 2024.",
        canonical={"unit": "%", "value_norm": 4.2, "time_norm": "March 2024", "geo_norm": None, "population_norm": None},
    )


def test_hybrid_retrieval_returns_evidence() -> None:
    retriever = HybridRetrievalPipeline(StageBConfig())
    retriever.index_corpus({"doc-1": "Official data shows unemployment was 4.2% in March 2024."})
    evidence = retriever.retrieve(_sample_claim())
    assert evidence, "Hybrid retriever should return at least one evidence span"
    assert evidence[0].doc_id == "doc-1"
