from dataclasses import dataclass
from typing import Sequence, Mapping, Any


@dataclass(frozen=True)
class TelemetryRecordDTO:
    record_id: str
    trace_id: str
    timestamp_epoch_ms: int
    features: Sequence[float]
    target_label: int


@dataclass(frozen=True)
class PipelineTaskSpecDTO:
    task_name: str
    modality: str
    metric_target: str
    records: Sequence[TelemetryRecordDTO]
    split_strategy: str
    train_ratio: float


@dataclass(frozen=True)
class DatasetSplitResult:
    train_records: Sequence[TelemetryRecordDTO]
    val_records: Sequence[TelemetryRecordDTO]
    train_traces: Sequence[str]
    val_traces: Sequence[str]


@dataclass(frozen=True)
class BaselineEvaluationResult:
    majority_class: int
    accuracy: float
    f1_score: float
    sample_count: int


@dataclass(frozen=True)
class TelemetryPipelineModel:
    pipeline_id: str
    task_name: str
    status: str
    baseline_result: BaselineEvaluationResult
    created_at_iso: str
