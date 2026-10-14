from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TelemetrySampleDTO:
    features: Sequence[float]
    label: int


@dataclass(frozen=True)
class NetworkSpecCreateDTO:
    name: str
    architecture_type: str
    input_dim: int
    hidden_dim: int
    num_classes: int
    layer_count: int
    parameter_budget: int
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class NetworkSpecModel:
    id: str
    name: str
    architecture_type: str
    input_dim: int
    hidden_dim: int
    num_classes: int
    layer_count: int
    parameter_budget: int
    status: str
    initial_loss: float = 0.0
    final_loss: float = 0.0
    steps_taken: int = 0
    is_overfit_verified: bool = False


@dataclass(frozen=True)
class TinyBatchOverfitCommand:
    spec_id: str
    samples: Sequence[TelemetrySampleDTO]
    max_steps: int = 50
    target_loss_threshold: float = 0.08
