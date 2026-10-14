from src.features.neural_learning_core.types.neural_learning_types import (
    OverfitVerificationConfigDTO,
    OverfitVerificationResult,
)
from src.features.neural_learning_core.workflows.tiny_batch_overfit_workflow import (
    TinyBatchOverfitWorkflow,
)
from src.shared.types.envelope_types import TraceContext, Result
from src.shared.errors.app_errors import ApplicationError
from src.infra.observability.tracer import Tracer


class NeuralLearningService:
    @staticmethod
    async def verify_tiny_batch_overfit(
        ctx: TraceContext, config: OverfitVerificationConfigDTO
    ) -> Result[OverfitVerificationResult, ApplicationError]:
        with Tracer.start_span("neural_learning.verify_overfit", ctx=ctx):
            try:
                res = TinyBatchOverfitWorkflow.execute_overfit_verification(config)
                return Result.ok(res)
            except ApplicationError as err:
                return Result.err(err)
            except Exception as ex:
                return Result.err(ApplicationError(str(ex)))
