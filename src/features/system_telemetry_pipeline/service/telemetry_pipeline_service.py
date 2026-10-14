from src.features.system_telemetry_pipeline.types.telemetry_pipeline_types import (
    PipelineTaskSpecDTO,
    TelemetryPipelineModel,
)
from src.features.system_telemetry_pipeline.workflows.trivial_baseline_workflow import (
    TrivialBaselineWorkflow,
)
from src.features.system_telemetry_pipeline.repository.telemetry_pipeline_repository_port import (
    TelemetryPipelineRepositoryPort,
)
from src.shared.types.envelope_types import TraceContext, Result
from src.shared.errors.app_errors import ApplicationError
from src.infra.observability.tracer import Tracer


class TelemetryPipelineService:
    def __init__(self, repository: TelemetryPipelineRepositoryPort) -> None:
        self._repository = repository
        self._workflow = TrivialBaselineWorkflow(repo=repository)

    async def initialize_pipeline_and_baseline(
        self, ctx: TraceContext, dto: PipelineTaskSpecDTO
    ) -> Result[TelemetryPipelineModel, ApplicationError]:
        with Tracer.start_span("telemetry_pipeline.initialize", ctx=ctx) as child_ctx:
            try:
                model = await self._workflow.execute_foundation_flow(child_ctx, dto)
                return Result.ok(model)
            except ApplicationError as err:
                return Result.err(err)
            except Exception as ex:
                return Result.err(ApplicationError(str(ex)))
