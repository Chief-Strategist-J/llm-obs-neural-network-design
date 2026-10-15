from src.features.streaming_baselines.types.streaming_baseline_types import (
    EWMAConfigDTO,
    CUSUMConfigDTO,
    StreamingFilterStateModel,
    FastPathTriageResultDTO,
)
from src.features.streaming_baselines.workflows.streaming_filter_workflow import (
    StreamingFilterWorkflow,
)
from src.features.streaming_baselines.repository.streaming_baseline_repository import (
    StreamingBaselineRepositoryPort,
)
from src.shared.types.envelope_types import TraceContext, Result
from src.shared.errors.app_errors import ApplicationError
from src.infra.observability.tracer import Tracer


class StreamingBaselineService:
    def __init__(self, repository: StreamingBaselineRepositoryPort) -> None:
        self._repository = repository
        self._workflow = StreamingFilterWorkflow(repo=repository)

    async def triage_span_metric(
        self,
        ctx: TraceContext,
        metric_name: str,
        value: float,
        ewma_cfg: EWMAConfigDTO | None = None,
        cusum_cfg: CUSUMConfigDTO | None = None,
    ) -> Result[FastPathTriageResultDTO, ApplicationError]:
        with Tracer.start_span("streaming_baseline.triage", ctx=ctx) as child_ctx:
            try:
                result = await self._workflow.execute_filter_step(
                    child_ctx, metric_name, value, ewma_cfg, cusum_cfg
                )
                return Result.ok(result)
            except ApplicationError as err:
                return Result.err(err)
            except Exception as ex:
                return Result.err(ApplicationError(str(ex)))

    async def get_state(
        self,
        ctx: TraceContext,
        metric_name: str,
    ) -> Result[StreamingFilterStateModel | None, ApplicationError]:
        with Tracer.start_span("streaming_baseline.get_state", ctx=ctx) as child_ctx:
            try:
                state = await self._repository.get_state(child_ctx, metric_name)
                return Result.ok(state)
            except ApplicationError as err:
                return Result.err(err)
            except Exception as ex:
                return Result.err(ApplicationError(str(ex)))
