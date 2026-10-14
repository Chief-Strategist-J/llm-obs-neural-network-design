"""Multi-step workflow DAG for network specification validation and persistence."""

from src.features.network_spec.types.network_spec_types import NetworkSpecCreateDTO, NetworkSpecModel
from src.features.network_spec.rules.network_spec_rules import NetworkSpecRules
from src.features.network_spec.repository.network_spec_repository_port import NetworkSpecRepositoryPort
from src.shared.types.envelope_types import TraceContext
from src.shared.constants.system_constants import STATUS_PENDING
from src.shared.errors.app_errors import ValidationError
from src.shared.utils.date_utils import generate_uuidv7_str


class NetworkSpecWorkflow:
    def __init__(self, repo: NetworkSpecRepositoryPort) -> None:
        self._repo = repo

    async def execute_creation_flow(self, ctx: TraceContext, dto: NetworkSpecCreateDTO) -> NetworkSpecModel:
        if not NetworkSpecRules.can_create_spec(dto):
            raise ValidationError("Network specification violates parameter density rules.")

        model = NetworkSpecModel(
            id=generate_uuidv7_str(),
            name=dto.name,
            architecture_type=dto.architecture_type,
            layer_count=dto.layer_count,
            parameter_budget=dto.parameter_budget,
            status=STATUS_PENDING,
        )
        return await self._repo.save(ctx, model)
