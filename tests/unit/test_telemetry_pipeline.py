import pytest
from src.features.system_telemetry_pipeline.contracts.telemetry_pipeline_types import (
    TelemetryRecordDTO,
    PipelineTaskSpecDTO,
)
from src.features.system_telemetry_pipeline.domain.rules.telemetry_split_rules import TelemetrySplitRules
from src.features.system_telemetry_pipeline.service.telemetry_pipeline_service import (
    TelemetryPipelineService,
)
from src.features.system_telemetry_pipeline.infrastructure.adapters.telemetry_pipeline_repository_adapter import (
    InMemoryTelemetryPipelineRepository,
)
from src.features.system_telemetry_pipeline.domain.machines.telemetry_pipeline_machine import (
    STATUS_BASELINE_ESTABLISHED,
)
from src.shared.types.envelope_types import TraceContext
from src.shared.errors.app_errors import ValidationError


def test_split_prevents_trace_leakage_and_temporal_lookahead():
    records = (
        TelemetryRecordDTO("r1", "trace-alpha", 1000, (0.1, 0.2), 0),
        TelemetryRecordDTO("r2", "trace-alpha", 1050, (0.3, 0.4), 0),
        TelemetryRecordDTO("r3", "trace-beta", 2000, (0.5, 0.6), 1),
        TelemetryRecordDTO("r4", "trace-gamma", 3000, (0.7, 0.8), 0),
    )
    split = TelemetrySplitRules.execute_leak_free_split(records, train_ratio=0.5)

    assert len(set(split.train_traces).intersection(set(split.val_traces))) == 0
    max_train_ts = max(r.timestamp_epoch_ms for r in split.train_records)
    min_val_ts = min(r.timestamp_epoch_ms for r in split.val_records)
    assert max_train_ts <= min_val_ts


def test_leakage_detection_raises():
    from src.features.system_telemetry_pipeline.contracts.telemetry_pipeline_types import DatasetSplitResult

    leaky_split = DatasetSplitResult(
        train_records=(TelemetryRecordDTO("r1", "trace-shared", 1000, (0.1,), 0),),
        val_records=(TelemetryRecordDTO("r2", "trace-shared", 1100, (0.2,), 1),),
        train_traces=("trace-shared",),
        val_traces=("trace-shared",),
    )
    with pytest.raises(ValidationError):
        TelemetrySplitRules.verify_split_leakage(leaky_split)


@pytest.mark.asyncio
async def test_service_evaluates_trivial_baseline():
    repo = InMemoryTelemetryPipelineRepository()
    service = TelemetryPipelineService(repository=repo)
    ctx = TraceContext(trace_id="test-trace-foundation")

    records = (
        TelemetryRecordDTO("r1", "trace-1", 1000, (0.1,), 0),
        TelemetryRecordDTO("r2", "trace-1", 1050, (0.2,), 0),
        TelemetryRecordDTO("r3", "trace-2", 2000, (0.3,), 0),
        TelemetryRecordDTO("r4", "trace-3", 3000, (0.4,), 1),
        TelemetryRecordDTO("r5", "trace-3", 3100, (0.5,), 0),
    )
    spec = PipelineTaskSpecDTO(
        task_name="distributed_anomaly_detection",
        modality="distributed_trace_sequence",
        metric_target="f1_macro",
        records=records,
        split_strategy="temporal_group",
        train_ratio=0.6,
    )

    result = await service.initialize_pipeline_and_baseline(ctx, spec)
    assert result.is_ok is True
    assert result.value is not None
    assert result.value.status == STATUS_BASELINE_ESTABLISHED
    assert result.value.baseline_result.majority_class == 0
    assert result.value.baseline_result.sample_count > 0
