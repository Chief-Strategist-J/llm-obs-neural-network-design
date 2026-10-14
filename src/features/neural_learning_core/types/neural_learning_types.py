from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TinyBatchSampleDTO:
    features: Sequence[float]
    label: int


@dataclass(frozen=True)
class OverfitVerificationConfigDTO:
    model_name: str
    input_dim: int
    hidden_dim: int
    num_classes: int
    samples: Sequence[TinyBatchSampleDTO]
    learning_rate: float = 0.01
    max_steps: int = 50
    target_loss_threshold: float = 0.05


@dataclass(frozen=True)
class OverfitVerificationResult:
    initial_loss: float
    final_loss: float
    steps_taken: int
    is_overfitted: bool
    status: str
