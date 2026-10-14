from src.features.network_spec.types.network_spec_types import (
    NetworkSpecCreateDTO,
    NetworkSpecModel,
    TinyBatchOverfitCommand,
)
from src.features.network_spec.workflows.network_spec_workflow import NetworkSpecWorkflow
from src.features.network_spec.repository.neural_compute_adapter import (
    NetworkSpecRepositoryPort,
    NeuralComputePort,
    PyTorchNeuralComputeAdapter,
)
from src.shared.types.envelope_types import TraceContext, Result
from src.shared.errors.app_errors import ApplicationError
from src.infra.observability.tracer import Tracer


class NetworkSpecService:
    def __init__(
        self,
        repository: NetworkSpecRepositoryPort,
        compute: NeuralComputePort | None = None,
    ) -> None:
        self._repository = repository
        self._compute = compute or PyTorchNeuralComputeAdapter()
        self._workflow = NetworkSpecWorkflow(repo=repository, compute=self._compute)

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

    async def verify_tiny_batch_overfit(
        self, ctx: TraceContext, command: TinyBatchOverfitCommand
    ) -> Result[NetworkSpecModel, ApplicationError]:
        with Tracer.start_span("network_spec.verify_overfit", ctx=ctx) as child_ctx:
            try:
                model = await self._workflow.execute_overfit_verification_flow(child_ctx, command)
                return Result.ok(model)
            except ApplicationError as err:
                return Result.err(err)
            except Exception as ex:
                return Result.err(ApplicationError(str(ex)))
