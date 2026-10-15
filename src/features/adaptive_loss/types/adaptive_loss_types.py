from dataclasses import dataclass
from typing import Sequence
from enum import Enum


class LossType(str, Enum):
    CROSS_ENTROPY = "cross_entropy"
    FOCAL_LOSS = "focal_loss"
    CLASS_BALANCED = "class_balanced"


@dataclass(frozen=True)
class AdaptiveLossConfigDTO:
    loss_type: LossType = LossType.FOCAL_LOSS
    gamma: float = 2.0
    alpha: Sequence[float] | None = None
    beta: float = 0.999
    reduction: str = "mean"


@dataclass(frozen=True)
class ClassBalanceProfileModel:
    profile_id: str
    class_counts: Sequence[int]
    calculated_weights: Sequence[float]
    beta_used: float
    created_at_iso: str
