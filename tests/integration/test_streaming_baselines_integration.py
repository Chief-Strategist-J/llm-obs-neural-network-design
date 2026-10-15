import pytest
from src.features.streaming_baselines.index import (
    StreamingBaselineService,
    InMemoryStreamingBaselineRepository,
    FastPathDecision,
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
async def test_streaming_fast_path_gating_into_neural_inference():
    ctx = TraceContext(trace_id="integration-fastpath-001")

    stream_repo = InMemoryStreamingBaselineRepository()
    stream_service = StreamingBaselineService(repository=stream_repo)

    net_repo = InMemoryNetworkSpecRepository()
    compute = PyTorchNeuralComputeAdapter()
    net_service = NetworkSpecService(repository=net_repo, compute=compute)

    create_res = await net_service.create_network_spec(
        ctx,
        NetworkSpecCreateDTO(
            name="uncertainty_classifier",
            architecture_type="mlp",
            input_dim=1,
            hidden_dim=8,
            num_classes=2,
            layer_count=2,
            parameter_budget=10000,
        ),
    )
    assert create_res.is_ok is True
    model_spec = create_res.value

    for i in range(20):
        val = 100.0 + (i % 3) * 5.0
        res = await stream_service.triage_span_metric(ctx, "latency_ms", val)
        assert res.is_ok is True
        assert res.value.requires_neural_inference is False

    triage_ambiguous = await stream_service.triage_span_metric(ctx, "latency_ms", 118.0)
    assert triage_ambiguous.is_ok is True
    assert triage_ambiguous.value.requires_neural_inference is True


    train_cmd = DatasetTrainingCommand(
        spec_id=model_spec.id,
        train_samples=(
            TelemetrySampleDTO(features=[100.0], label=0),
            TelemetrySampleDTO(features=[130.0], label=1),
        ),
        val_samples=(
            TelemetrySampleDTO(features=[100.0], label=0),
            TelemetrySampleDTO(features=[130.0], label=1),
        ),
        epochs=5,
        batch_size=2,
        learning_rate=0.01,
        weight_decay=1e-4,
        target_checkpoint_dir="checkpoints/fastpath_test",
    )
    train_res = await net_service.train_dataset(ctx, train_cmd)
    assert train_res.is_ok is True
    assert train_res.value.trained_epochs == 5
