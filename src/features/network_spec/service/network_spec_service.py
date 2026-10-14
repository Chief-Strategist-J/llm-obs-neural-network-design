"""Domain service coordinating business invariants and outbound ports."""

from src.features.network_spec.types.network_spec_types import NetworkSpecCreateDTO, NetworkSpecModel
from src.features.network_spec.workflows.network_spec_workflow import NetworkSpecWorkflow
from src.features.network_spec.repository.network_spec_repository_port import NetworkSpecRepositoryPort
from src.shared.types.envelope_types import TraceContext, Result
from src.shared.errors.app_errors import ApplicationError
from src.infra.observability.tracer import Tracer


class NetworkSpecService:
    def __init__(self, repository: NetworkSpecRepositoryPort) -> None:
        self._repository = repository
        self._workflow = NetworkSpecWorkflow(repo=repository)

    async def create_network_spec(
        self, ctx: TraceContext, dto: NetworkSpecCreateDTO
    ) -> Result[NetworkSpecModel, ApplicationError]:
        with Tracer.start_span("network_spec.create", ctx=ctx) as child_ctx:
            try:
                model = await self._workflow.execute_creation_flow(child_ctx, dto)
                return Result.ok(model)
            except ApplicationError as err:
                return Result.err(err)
            except Exception as ex:
                return Result.err(ApplicationError(str(ex)))
