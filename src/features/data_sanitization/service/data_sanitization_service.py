from typing import Sequence
from src.features.data_sanitization.types.data_sanitization_types import (
    SanitizationProfileModel,
    SanitizationConfigDTO,
    CleanedSampleDTO,
)
from src.features.data_sanitization.workflows.data_sanitization_workflow import (
    DataSanitizationWorkflow,
)
from src.features.data_sanitization.repository.data_sanitization_repository import (
    DataSanitizationRepositoryPort,
)
from src.shared.types.envelope_types import TraceContext, Result
from src.shared.errors.app_errors import ApplicationError
from src.infra.observability.tracer import Tracer


class DataSanitizationService:
    def __init__(self, repository: DataSanitizationRepositoryPort) -> None:
        self._repository = repository
        self._workflow = DataSanitizationWorkflow(repo=repository)

    async def fit_profile(
        self,
        ctx: TraceContext,
        train_features: Sequence[Sequence[float]],
        config: SanitizationConfigDTO | None = None,
    ) -> Result[SanitizationProfileModel, ApplicationError]:
        with Tracer.start_span("data_sanitization.fit", ctx=ctx) as child_ctx:
            try:
                profile = await self._workflow.execute_fit_flow(child_ctx, train_features, config)
                return Result.ok(profile)
            except ApplicationError as err:
                return Result.err(err)
            except Exception as ex:
                return Result.err(ApplicationError(str(ex)))

    async def transform_samples(
        self,
        ctx: TraceContext,
        profile_id: str,
        samples: Sequence[tuple[str, Sequence[float], int]],
    ) -> Result[tuple[CleanedSampleDTO, ...], ApplicationError]:
        with Tracer.start_span("data_sanitization.transform", ctx=ctx) as child_ctx:
            try:
                cleaned = await self._workflow.execute_transform_flow(child_ctx, profile_id, samples)
                return Result.ok(cleaned)
            except ApplicationError as err:
                return Result.err(err)
            except Exception as ex:
                return Result.err(ApplicationError(str(ex)))
