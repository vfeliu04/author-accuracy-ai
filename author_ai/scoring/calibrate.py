"""Calibration helpers for turning logits into probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass
class TemperatureCalibrator:
    """Standard logistic temperature scaling."""

    temperature: float = 1.0

    def calibrate(self, logit: float) -> float:
        scaled = logit / max(self.temperature, 1e-6)
        return 1.0 / (1.0 + math.exp(-scaled))


@dataclass
class IdentityCalibrator:
    """No-op calibrator used for 'none' or stub isotonic."""

    def calibrate(self, value: float) -> float:
        return max(0.0, min(1.0, value))


def build_calibrator(kind: Literal["temperature", "isotonic", "none"], temperature: float) -> object:
    """Factory returning the appropriate calibrator instance."""

    if kind == "temperature":
        return TemperatureCalibrator(temperature=temperature)
    if kind == "none":
        return IdentityCalibrator()
    # Placeholder: isotonic would need training data; fall back to identity.
    return IdentityCalibrator()


__all__ = ["TemperatureCalibrator", "IdentityCalibrator", "build_calibrator"]
