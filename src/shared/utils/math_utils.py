"""Pure deterministic math utilities for metrics, loss calculation, and feature normalization."""

from __future__ import annotations
import math
from typing import Sequence


def compute_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_val = compute_mean(values)
    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def z_score_normalize(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        return ()
    mean_val = compute_mean(values)
    std_val = compute_standard_deviation(values)
    if std_val == 0.0:
        return tuple(0.0 for _ in values)
    return tuple((x - mean_val) / std_val for x in values)
