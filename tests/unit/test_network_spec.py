import pytest
from src.features.network_spec.types.network_spec_types import (
    NetworkSpecCreateDTO,
    TelemetrySampleDTO,
    TinyBatchOverfitCommand,
)
from src.features.network_spec.schema.network_spec_schema import NetworkSpecSchema
from src.features.network_spec.rules.network_spec_rules import NetworkSpecRules
from src.features.network_spec.machines.network_spec_machine import (
    NetworkSpecMachine,
    STATUS_PENDING,
    STATUS_ACTIVE,
    STATUS_TINY_BATCH_OVERFIT_VERIFIED,
    STATUS_ARCHIVED,
)
from src.features.network_spec.service.network_spec_service import NetworkSpecService
from src.features.network_spec.repository.network_spec_repository import (
    InMemoryNetworkSpecRepository,
)
from src.features.network_spec.repository.neural_compute_adapter import (
    PyTorchNeuralComputeAdapter,
)
from src.shared.types.envelope_types import TraceContext
from src.shared.errors.app_errors import ValidationError


def test_schema_validation_valid():
    payload = {
        "name": "Transformer-Decoder-7B",
        "architecture_type": "transformer",
        "input_dim": 32,
        "hidden_dim": 128,
        "num_classes": 2,
        "layer_count": 32,
        "parameter_budget": 7_000_000_000,
    }
    dto = NetworkSpecSchema.from_api(payload)
    assert dto.name == "Transformer-Decoder-7B"
    assert dto.architecture_type == "transformer"
    assert dto.input_dim == 32
    assert dto.hidden_dim == 128
    assert dto.layer_count == 32
    assert dto.parameter_budget == 7_000_000_000


def test_schema_validation_invalid_arch():
    payload = {
        "name": "InvalidNet",
        "architecture_type": "quantum_unsupported",
        "layer_count": 10,
        "parameter_budget": 1000,
    }
    with pytest.raises(ValidationError):
        NetworkSpecSchema.from_api(payload)


def test_rules_evaluation():
    valid_dto = NetworkSpecCreateDTO(
        name="ValidNet",
        architecture_type="cnn",
        input_dim=16,
        hidden_dim=32,
        num_classes=2,
        layer_count=50,
        parameter_budget=500_000,
    )
    assert NetworkSpecRules.can_create_spec(valid_dto) is True

    dense_invalid_dto = NetworkSpecCreateDTO(
        name="InvalidNet",
        architecture_type="cnn",
        input_dim=16,
        hidden_dim=32,
        num_classes=2,
        layer_count=600,
        parameter_budget=10_000,
    )
    assert NetworkSpecRules.can_create_spec(dense_invalid_dto) is False


def test_state_machine_transitions():
    assert NetworkSpecMachine.transition(STATUS_PENDING, STATUS_ACTIVE) == STATUS_ACTIVE
    assert NetworkSpecMachine.transition(STATUS_ACTIVE, STATUS_ARCHIVED) == STATUS_ARCHIVED
    assert (
        NetworkSpecMachine.transition(STATUS_PENDING, STATUS_TINY_BATCH_OVERFIT_VERIFIED)
        == STATUS_TINY_BATCH_OVERFIT_VERIFIED
    )

    with pytest.raises(ValidationError):
        NetworkSpecMachine.transition(STATUS_ARCHIVED, STATUS_ACTIVE)


@pytest.mark.asyncio
async def test_service_creation_and_pytorch_overfit():
    repo = InMemoryNetworkSpecRepository()
    compute = PyTorchNeuralComputeAdapter()
    service = NetworkSpecService(repository=repo, compute=compute)
    ctx = TraceContext(trace_id="test-trace-123")

    dto = NetworkSpecCreateDTO(
        name="TelemetryResidualNet",
        architecture_type="transformer",
        input_dim=4,
        hidden_dim=32,
        num_classes=2,
        layer_count=2,
        parameter_budget=100_000,
    )

    create_res = await service.create_network_spec(ctx, dto)
    assert create_res.is_ok is True
    assert create_res.value is not None
    spec_id = create_res.value.id

    samples = (
        TelemetrySampleDTO(features=(1.0, 0.0, -1.0, 0.5), label=0),
        TelemetrySampleDTO(features=(0.0, 1.0, 1.0, -0.5), label=1),
        TelemetrySampleDTO(features=(1.2, -0.1, -0.8, 0.4), label=0),
        TelemetrySampleDTO(features=(-0.2, 0.9, 1.1, -0.4), label=1),
    )
    cmd = TinyBatchOverfitCommand(
        spec_id=spec_id,
        samples=samples,
        max_steps=50,
        target_loss_threshold=0.08,
    )

    overfit_res = await service.verify_tiny_batch_overfit(ctx, cmd)
    assert overfit_res.is_ok is True
    assert overfit_res.value is not None
    assert overfit_res.value.is_overfit_verified is True
    assert overfit_res.value.final_loss <= 0.08
    assert overfit_res.value.status == STATUS_TINY_BATCH_OVERFIT_VERIFIED
