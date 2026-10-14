import pytest
from src.features.network_spec.types.network_spec_types import NetworkSpecCreateDTO
from src.features.network_spec.schema.network_spec_schema import NetworkSpecSchema
from src.features.network_spec.rules.network_spec_rules import NetworkSpecRules
from src.features.network_spec.machines.network_spec_machine import NetworkSpecMachine
from src.features.network_spec.service.network_spec_service import NetworkSpecService
from src.features.network_spec.repository.network_spec_repository import InMemoryNetworkSpecRepository
from src.shared.types.envelope_types import TraceContext
from src.shared.constants.system_constants import STATUS_PENDING, STATUS_ACTIVE, STATUS_ARCHIVED
from src.shared.errors.app_errors import ValidationError


def test_schema_validation_valid():
    payload = {
        "name": "Transformer-Decoder-7B",
        "architecture_type": "transformer",
        "layer_count": 32,
        "parameter_budget": 7_000_000_000,
    }
    dto = NetworkSpecSchema.from_api(payload)
    assert dto.name == "Transformer-Decoder-7B"
    assert dto.architecture_type == "transformer"
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
        layer_count=50,
        parameter_budget=500_000,
    )
    assert NetworkSpecRules.can_create_spec(valid_dto) is True

    dense_invalid_dto = NetworkSpecCreateDTO(
        name="InvalidNet",
        architecture_type="cnn",
        layer_count=600,
        parameter_budget=10_000,
    )
    assert NetworkSpecRules.can_create_spec(dense_invalid_dto) is False


def test_state_machine_transitions():
    assert NetworkSpecMachine.transition(STATUS_PENDING, STATUS_ACTIVE) == STATUS_ACTIVE
    assert NetworkSpecMachine.transition(STATUS_ACTIVE, STATUS_ARCHIVED) == STATUS_ARCHIVED

    with pytest.raises(ValidationError):
        NetworkSpecMachine.transition(STATUS_ARCHIVED, STATUS_ACTIVE)


@pytest.mark.asyncio
async def test_service_creation_flow():
    repo = InMemoryNetworkSpecRepository()
    service = NetworkSpecService(repository=repo)
    ctx = TraceContext(trace_id="test-trace-123")
    dto = NetworkSpecCreateDTO(
        name="GPT-Lite",
        architecture_type="transformer",
        layer_count=12,
        parameter_budget=125_000_000,
    )

    result = await service.create_network_spec(ctx, dto)
    assert result.is_ok is True
    assert result.value is not None
    assert result.value.name == "GPT-Lite"
    assert result.value.status == STATUS_PENDING
