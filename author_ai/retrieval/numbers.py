"""Numeric index used to nudge retrieval toward matching magnitudes.

The implementation is deliberately simple: we scan documents for numbers and
store them in memory.  Queries supply the target numbers we care about and we
return documents where the distance is within the configured tolerances.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence


NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _to_float(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


@dataclass
class NumberMatch:
    doc_id: str
    value: float
    distance: float
    count: int


class NumberIndex:
    """Keeps a lookup of numeric mentions per document."""

    def __init__(self, absolute_tolerance: float = 0.5, relative_tolerance: float = 0.03) -> None:
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self._doc_numbers: dict[str, list[float]] = {}

    def index(self, docs: Iterable[tuple[str, str]]) -> None:
        """Extract every number from the provided documents."""

        self._doc_numbers.clear()
        for doc_id, content in docs:
            numbers: list[float] = []
            for match in NUMBER_PATTERN.findall(content):
                value = _to_float(match)
                if value is None:
                    continue
                numbers.append(value)
            if numbers:
                self._doc_numbers[doc_id] = numbers

    def search(self, targets: Sequence[float], top_k: int = 3) -> List[NumberMatch]:
        """Find documents with numerics close to any of the target values."""

        if not targets:
            return []

        matches: list[NumberMatch] = []
        for doc_id, numbers in self._doc_numbers.items():
            match_count = 0
            best_distance = float("inf")
            best_value = None
            for candidate in numbers:
                for target in targets:
                    distance = self._distance(candidate, target)
                    if distance is None:
                        continue
                    match_count += 1
                    if distance < best_distance:
                        best_distance = distance
                        best_value = candidate
            if match_count and best_value is not None:
                matches.append(
                    NumberMatch(doc_id=doc_id, value=best_value, distance=best_distance, count=match_count)
                )

        matches.sort(key=lambda item: (item.distance, -item.count))
        return matches[:top_k]

    def _distance(self, candidate: float, target: float) -> float | None:
        """Return an absolute or relative distance if tolerances are met."""

        absolute_diff = abs(candidate - target)
        if absolute_diff <= self.absolute_tolerance:
            return absolute_diff

        if target == 0:
            return None

        relative_diff = abs(candidate - target) / abs(target)
        if relative_diff <= self.relative_tolerance:
            return relative_diff

        return None


__all__ = ["NumberIndex", "NumberMatch"]
