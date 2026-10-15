from typing import Sequence
from src.features.adaptive_loss.types.adaptive_loss_types import ClassBalanceProfileModel
from src.features.adaptive_loss.workflows.adaptive_loss_workflow import AdaptiveLossWorkflow
from src.features.adaptive_loss.repository.adaptive_loss_repository import AdaptiveLossRepositoryPort
from src.shared.types.envelope_types import TraceContext, Result
from src.shared.errors.app_errors import ApplicationError
from src.infra.observability.tracer import Tracer


class AdaptiveLossService:
    def __init__(self, repository: AdaptiveLossRepositoryPort) -> None:
        self._repository = repository
        self._workflow = AdaptiveLossWorkflow(repo=repository)

    async def calculate_class_balance_profile(
        self,
        ctx: TraceContext,
        class_counts: Sequence[int],
        beta: float = 0.999,
    ) -> Result[ClassBalanceProfileModel, ApplicationError]:
        with Tracer.start_span("adaptive_loss.calculate_profile", ctx=ctx) as child_ctx:
            try:
                profile = await self._workflow.execute_calculate_profile_flow(
                    child_ctx, class_counts, beta=beta
                )
                return Result.ok(profile)
            except ApplicationError as err:
                return Result.err(err)
            except Exception as ex:
                return Result.err(ApplicationError(str(ex)))

    async def get_profile(
        self,
        ctx: TraceContext,
        profile_id: str,
    ) -> Result[ClassBalanceProfileModel | None, ApplicationError]:
        with Tracer.start_span("adaptive_loss.get_profile", ctx=ctx) as child_ctx:
            try:
                profile = await self._repository.find_profile_by_id(child_ctx, profile_id)
                return Result.ok(profile)
            except ApplicationError as err:
                return Result.err(err)
            except Exception as ex:
                return Result.err(ApplicationError(str(ex)))
