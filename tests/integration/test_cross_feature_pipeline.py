import pytest
from src.features.system_telemetry_pipeline.index import (
    PipelineTaskSpecDTO,
    TelemetryRecordDTO,
    TelemetryPipelineService,
    InMemoryTelemetryPipelineRepository,
)
from src.features.system_telemetry_pipeline.domain.rules.telemetry_split_rules import TelemetrySplitRules
from src.features.data_sanitization.index import (
    DataSanitizationService,
    InMemoryDataSanitizationRepository,
    SanitizationConfigDTO,
)
from src.features.network_spec.index import (
    NetworkSpecService,
    InMemoryNetworkSpecRepository,
    PyTorchNeuralComputeAdapter,
    NetworkSpecCreateDTO,
    TelemetrySampleDTO,
    DatasetTrainingCommand,
)
from src.shared.types.envelope_types import TraceContext


@pytest.mark.asyncio
async def test_end_to_end_cross_feature_pipeline_flow():
    ctx = TraceContext(trace_id="integration-cross-feature-001")

    raw_records = (
        TelemetryRecordDTO(
            record_id="rec_1",
            trace_id="trace_001",
            timestamp_epoch_ms=1000,
            features=(100.0, 1.0, 0.0),
            target_label=0,
        ),
        TelemetryRecordDTO(
            record_id="rec_2",
            trace_id="trace_001",
            timestamp_epoch_ms=1100,
            features=(150.0, 1.0, 1.0),
            target_label=0,
        ),
        TelemetryRecordDTO(
            record_id="rec_3",
            trace_id="trace_002",
            timestamp_epoch_ms=1300,
            features=(200.0, 2.0, 2.0),
            target_label=1,
        ),
        TelemetryRecordDTO(
            record_id="rec_4",
            trace_id="trace_003",
            timestamp_epoch_ms=1600,
            features=(200.0, 1.0, 0.0),
            target_label=0,
        ),
        TelemetryRecordDTO(
            record_id="rec_5",
            trace_id="trace_004",
            timestamp_epoch_ms=2000,
            features=(300.0, 2.0, 3.0),
            target_label=1,
        ),
    )

    split = TelemetrySplitRules.execute_leak_free_split(raw_records, train_ratio=0.75)
    assert len(split.train_records) > 0
    assert len(split.val_records) > 0

    pipeline_repo = InMemoryTelemetryPipelineRepository()
    pipeline_service = TelemetryPipelineService(repository=pipeline_repo)
    task_spec = PipelineTaskSpecDTO(
        task_name="integration_classification_task",
        modality="distributed_trace_sequence",
        metric_target="f1_macro",
        records=raw_records,
        split_strategy="temporal_group",
        train_ratio=0.75,
    )
    init_res = await pipeline_service.initialize_pipeline_and_baseline(ctx, task_spec)
    assert init_res.is_ok is True
    pipeline_model = init_res.value
    assert pipeline_model is not None
    assert pipeline_model.status == "BASELINE_ESTABLISHED"

    train_feature_matrix = [list(r.features) for r in split.train_records]

    sanitization_repo = InMemoryDataSanitizationRepository()
    sanitization_service = DataSanitizationService(repository=sanitization_repo)

    fit_res = await sanitization_service.fit_profile(
        ctx,
        train_feature_matrix,
        SanitizationConfigDTO(clip_outliers=True, normalize_zscore=True),
    )
    assert fit_res.is_ok is True
    profile = fit_res.value
    assert profile is not None
    assert profile.feature_count == 3

    train_samples_raw = [(r.record_id, list(r.features), r.target_label) for r in split.train_records]
    val_samples_raw = [(r.record_id, list(r.features), r.target_label) for r in split.val_records]

    cleaned_train_res = await sanitization_service.transform_samples(ctx, profile.profile_id, train_samples_raw)
    cleaned_val_res = await sanitization_service.transform_samples(ctx, profile.profile_id, val_samples_raw)
    assert cleaned_train_res.is_ok is True
    assert cleaned_val_res.is_ok is True

    cleaned_train = cleaned_train_res.value
    cleaned_val = cleaned_val_res.value
    assert cleaned_train is not None
    assert cleaned_val is not None

    net_repo = InMemoryNetworkSpecRepository()
    compute = PyTorchNeuralComputeAdapter()
    net_service = NetworkSpecService(repository=net_repo, compute=compute)

    create_dto = NetworkSpecCreateDTO(
        name="integration_neural_net",
        architecture_type="mlp",
        input_dim=3,
        hidden_dim=16,
        num_classes=2,
        layer_count=2,
        parameter_budget=50_000,
    )

    create_res = await net_service.create_network_spec(ctx, create_dto)
    assert create_res.is_ok is True
    spec_model = create_res.value
    assert spec_model is not None

    train_cmd = DatasetTrainingCommand(
        spec_id=spec_model.id,
        train_samples=tuple(TelemetrySampleDTO(features=s.cleaned_features, label=s.target_label) for s in cleaned_train),
        val_samples=tuple(TelemetrySampleDTO(features=s.cleaned_features, label=s.target_label) for s in cleaned_val),
        epochs=10,
        batch_size=2,
        learning_rate=0.01,
        weight_decay=1e-4,
        target_checkpoint_dir="checkpoints/integration_test",
    )

    train_res = await net_service.train_dataset(ctx, train_cmd)
    assert train_res.is_ok is True
    training_res = train_res.value
    assert training_res is not None
    assert training_res.trained_epochs == 10
    assert training_res.checkpoint_path != ""
