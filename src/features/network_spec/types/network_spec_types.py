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


@dataclass(frozen=True)
class DatasetTrainingCommand:
    spec_id: str
    train_samples: Sequence[TelemetrySampleDTO]
    val_samples: Sequence[TelemetrySampleDTO]
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    target_checkpoint_dir: str = "checkpoints"


@dataclass(frozen=True)
class DatasetTrainingResult:
    spec_id: str
    trained_epochs: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    checkpoint_path: str
    checkpoint_sha256: str
