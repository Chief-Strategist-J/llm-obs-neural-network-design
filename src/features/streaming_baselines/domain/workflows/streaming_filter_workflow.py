import math
import time
from src.features.streaming_baselines.contracts.streaming_baseline_types import (
    EWMAConfigDTO,
    CUSUMConfigDTO,
    StreamingFilterStateModel,
    FastPathTriageResultDTO,
)
from src.features.streaming_baselines.domain.rules.streaming_baseline_rules import StreamingBaselineRules
from src.features.streaming_baselines.infrastructure.adapters.streaming_baseline_repository_adapter import (
    StreamingBaselineRepositoryPort,
)
from src.shared.types.envelope_types import TraceContext


class StreamingFilterWorkflow:
    def __init__(self, repo: StreamingBaselineRepositoryPort) -> None:
        self._repo = repo

    async def execute_filter_step(
        self,
        ctx: TraceContext,
        metric_name: str,
        value: float,
        ewma_cfg: EWMAConfigDTO | None = None,
        cusum_cfg: CUSUMConfigDTO | None = None,
    ) -> FastPathTriageResultDTO:
        e_cfg = ewma_cfg or EWMAConfigDTO()
        c_cfg = cusum_cfg or CUSUMConfigDTO()

        state = await self._repo.get_state(ctx, metric_name)
        if not state:
            state = StreamingFilterStateModel(
                metric_name=metric_name,
                sample_count=0,
                mean_estimate=value,
                variance_estimate=1.0,
                ewma_value=value,
                cusum_pos=0.0,
                cusum_neg=0.0,
                last_updated_epoch_ms=int(time.time() * 1000),
            )

        m2 = state.variance_estimate * max(1, state.sample_count - 1)
        new_count, new_mean, new_m2 = StreamingBaselineRules.update_welford_stats(
            state.sample_count, state.mean_estimate, m2, value
        )
        new_var = new_m2 / max(1, new_count - 1) if new_count > 1 else 1.0

        new_ewma = StreamingBaselineRules.update_ewma(state.ewma_value, value, e_cfg.alpha)

        std = math.sqrt(new_var) if new_var > 1e-8 else 1.0
        standardized = (value - new_mean) / std
        new_pos, new_neg = StreamingBaselineRules.update_cusum(
            state.cusum_pos, state.cusum_neg, standardized, c_cfg.k_slack
        )

        updated_state = StreamingFilterStateModel(
            metric_name=metric_name,
            sample_count=new_count,
            mean_estimate=new_mean,
            variance_estimate=new_var,
            ewma_value=new_ewma,
            cusum_pos=new_pos,
            cusum_neg=new_neg,
            last_updated_epoch_ms=int(time.time() * 1000),
        )

        triage_result = StreamingBaselineRules.evaluate_triage(
            val=value,
            state=updated_state,
            ewma_cfg=e_cfg,
            cusum_cfg=c_cfg,
        )

        await self._repo.save_state(ctx, updated_state)
        return triage_result
