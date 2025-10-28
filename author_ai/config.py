"""Centralised configuration for the five-stage pipeline.

Every knob lives here so the CLI, tests, and future notebooks can tweak
behaviour without hunting through the implementation.  The defaults mirror
the tolerances from the project brief and can be serialised to JSON/YAML
if we ever need persisted configs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StageAConfig(BaseModel):
    """Candidate extraction (LLM) settings."""

    batch_size: int = Field(default=4, description="How many sentences to batch per fake LLM call.")
    max_claims: int | None = Field(default=None, description="Hard limit on emitted claims.")


class StageBConfig(BaseModel):
    """Hybrid retrieval settings."""

    top_k_text: int = 2
    top_k_tables: int = 1
    max_corpus_chars: int = Field(default=200_000, description="Trim huge documents for predictability.")
    bm25_weight: float = 0.6
    dense_weight: float = 0.3
    number_weight: float = 0.1
    min_numeric_overlap: int = 1
    allowed_sources: list[str] | None = None


class StageCConfig(BaseModel):
    """Judge configuration (LLM surrogate)."""

    max_spans: int = Field(default=3, description="Two text spans + one table row by default.")
    temperature: float = 0.0
    model_name: str = "stub-judge"
    fail_on_invalid_json: bool = True


class StageDTolerances(BaseModel):
    """Numerical tolerances controlling disagreement thresholds."""

    percent_abs: float = Field(default=0.5, description="Absolute tolerance for percentage points.")
    rate_per_rel: float = Field(default=0.02, description="Relative tolerance for per-capita rates.")
    ratio_nominal: float = Field(default=0.20, description="Nominal expected ratio (e.g. 1 in 5).")
    ratio_tolerance: float = Field(default=0.01, description="Allowed deviation around the ratio nominal.")
    disputed_disagreement: float = Field(default=0.1, description="Gap before we flag evidence as disputed.")


class StageDWeights(BaseModel):
    """Weights used when combining feature scores."""

    evidence_strength: float = 0.35
    unit_time_geo_checks: float = 0.2
    numeric_tolerance: float = 0.2
    source_quality: float = 0.15
    judge_label_weight: float = 0.1


class StageDConfig(BaseModel):
    """Stage D: scoring + calibration."""

    tolerances: StageDTolerances = StageDTolerances()
    weights: StageDWeights = StageDWeights()
    calibration: Literal["temperature", "isotonic", "none"] = "temperature"
    temperature: float = Field(default=1.5, description="Default temperature for logit scaling.")


class StageEConfig(BaseModel):
    """Stage E: reporting."""

    highlight_tag: str = Field(default="mark", description="HTML tag used to wrap inline highlights.")
    include_json: bool = Field(default=True, description="Emit JSONL alongside the HTML report.")


class PipelineConfig(BaseModel):
    """Convenience wrapper bundling all stage configurations."""

    stage_a: StageAConfig = StageAConfig()
    stage_b: StageBConfig = StageBConfig()
    stage_c: StageCConfig = StageCConfig()
    stage_d: StageDConfig = StageDConfig()
    stage_e: StageEConfig = StageEConfig()


def load_default_config() -> PipelineConfig:
    """Helper to mirror the existing `config.py` import pattern in tests."""

    return PipelineConfig()


__all__ = [
    "PipelineConfig",
    "StageAConfig",
    "StageBConfig",
    "StageCConfig",
    "StageDConfig",
    "StageEConfig",
    "StageDTolerances",
    "StageDWeights",
    "load_default_config",
]
