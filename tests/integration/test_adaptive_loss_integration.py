import pytest
from src.features.network_spec.index import (
    NetworkSpecService,
    InMemoryNetworkSpecRepository,
    PyTorchNeuralComputeAdapter,
    NetworkSpecCreateDTO,
    TelemetrySampleDTO,
    DatasetTrainingCommand,
)
from src.features.adaptive_loss.index import (
    AdaptiveLossService,
    InMemoryAdaptiveLossRepository,
)
from src.shared.types.envelope_types import TraceContext


@pytest.mark.asyncio
async def test_adaptive_loss_neural_training_imbalanced_dataset():
    ctx = TraceContext(trace_id="integration-adaptive-loss-001")

    loss_repo = InMemoryAdaptiveLossRepository()
    loss_service = AdaptiveLossService(repository=loss_repo)

    class_counts = [95, 5]
    profile_res = await loss_service.calculate_class_balance_profile(ctx, class_counts, beta=0.99)
    assert profile_res.is_ok is True
    weights = profile_res.value.calculated_weights

    train_samples = []
    for _ in range(95):
        train_samples.append(TelemetrySampleDTO(features=[10.0, 1.0, 0.0], label=0))
    for _ in range(5):
        train_samples.append(TelemetrySampleDTO(features=[500.0, 2.0, 5.0], label=1))

    val_samples = [
        TelemetrySampleDTO(features=[10.0, 1.0, 0.0], label=0),
        TelemetrySampleDTO(features=[500.0, 2.0, 5.0], label=1),
    ]

    net_repo = InMemoryNetworkSpecRepository()
    compute = PyTorchNeuralComputeAdapter()
    net_service = NetworkSpecService(repository=net_repo, compute=compute)

    create_dto = NetworkSpecCreateDTO(
        name="imbalanced_adaptive_loss_net",
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

    train_cmd = DatasetTrainingCommand(
        spec_id=spec_model.id,
        train_samples=tuple(train_samples),
        val_samples=tuple(val_samples),
        epochs=15,
        batch_size=10,
        learning_rate=0.01,
        weight_decay=1e-4,
        target_checkpoint_dir="checkpoints/adaptive_loss_test",
        loss_type="focal_loss",
        focal_gamma=2.0,
        class_weights=weights,
    )

    train_res = await net_service.train_dataset(ctx, train_cmd)
    assert train_res.is_ok is True
    training_res = train_res.value
    assert training_res is not None
    assert training_res.trained_epochs == 15
    assert training_res.checkpoint_path != ""
    assert training_res.train_loss > 0.0
