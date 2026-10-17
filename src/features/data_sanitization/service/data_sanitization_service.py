from typing import Sequence
from src.features.data_sanitization.contracts.data_sanitization_types import (
    SanitizationProfileModel,
    SanitizationConfigDTO,
    CleanedSampleDTO,
)
from src.features.data_sanitization.domain.workflows.data_sanitization_workflow import (
    DataSanitizationWorkflow,
)
from src.features.data_sanitization.infrastructure.adapters.data_sanitization_repository_adapter import (
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

    async def fit_profile_polars(
        self,
        ctx: TraceContext,
        df: "polars.DataFrame",
        feature_columns: Sequence[str],
        config: SanitizationConfigDTO | None = None,
    ) -> Result[SanitizationProfileModel, ApplicationError]:
        from src.features.data_sanitization.domain.rules.data_sanitization_rules import DataSanitizationRules
        from src.features.data_sanitization.domain.machines.data_sanitization_machine import STATUS_FITTED
        from src.shared.utils.date_utils import generate_uuidv7_str, get_utc_iso_timestamp

        with Tracer.start_span("data_sanitization.fit_polars", ctx=ctx) as child_ctx:
            try:
                cfg = config or SanitizationConfigDTO()
                stats = DataSanitizationRules.compute_feature_statistics_polars(df, feature_columns)
                profile = SanitizationProfileModel(
                    profile_id=generate_uuidv7_str(),
                    feature_count=len(stats),
                    statistics=stats,
                    config=cfg,
                    status=STATUS_FITTED,
                    created_at_iso=get_utc_iso_timestamp(),
                )
                saved = await self._repository.save_profile(child_ctx, profile)
                return Result.ok(saved)
            except ApplicationError as err:
                return Result.err(err)
            except Exception as ex:
                return Result.err(ApplicationError(str(ex)))

    async def transform_dataframe_polars(
        self,
        ctx: TraceContext,
        profile_id: str,
        df: "polars.DataFrame",
        feature_columns: Sequence[str],
    ) -> Result["polars.DataFrame", ApplicationError]:
        with Tracer.start_span("data_sanitization.transform_polars", ctx=ctx) as child_ctx:
            try:
                cleaned_df = await self._workflow.execute_transform_polars_flow(
                    child_ctx, profile_id, df, feature_columns
                )
                return Result.ok(cleaned_df)
            except ApplicationError as err:
                return Result.err(err)
            except Exception as ex:
                return Result.err(ApplicationError(str(ex)))

