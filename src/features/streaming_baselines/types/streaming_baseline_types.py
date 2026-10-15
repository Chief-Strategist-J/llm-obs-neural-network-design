from dataclasses import dataclass
from typing import Sequence
from enum import Enum


class FastPathDecision(str, Enum):
    BENIGN = "benign"
    ANOMALOUS = "anomalous"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class EWMAConfigDTO:
    alpha: float = 0.2
    target_metric: str = "latency_ms"
    warmup_samples: int = 10


@dataclass(frozen=True)
class CUSUMConfigDTO:
    k_slack: float = 0.5
    h_threshold: float = 4.0
    drift_allowance: float = 0.1


@dataclass(frozen=True)
class StreamingFilterStateModel:
    metric_name: str
    sample_count: int
    mean_estimate: float
    variance_estimate: float
    ewma_value: float
    cusum_pos: float
    cusum_neg: float
    last_updated_epoch_ms: int


@dataclass(frozen=True)
class FastPathTriageResultDTO:
    metric_name: str
    raw_value: float
    decision: FastPathDecision
    ewma: float
    z_score: float
    requires_neural_inference: bool
