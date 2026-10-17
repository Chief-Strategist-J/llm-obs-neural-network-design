import math
from src.features.streaming_baselines.contracts.streaming_baseline_types import (
    EWMAConfigDTO,
    CUSUMConfigDTO,
    StreamingFilterStateModel,
    FastPathDecision,
    FastPathTriageResultDTO,
)
from src.shared.errors.app_errors import ValidationError


class StreamingBaselineRules:
    @staticmethod
    def update_welford_stats(
        count: int, mean: float, m2: float, new_val: float
    ) -> tuple[int, float, float]:
        new_count = count + 1
        delta = new_val - mean
        new_mean = mean + delta / new_count
        delta2 = new_val - new_mean
        new_m2 = m2 + delta * delta2
        return new_count, new_mean, new_m2

    @staticmethod
    def update_ewma(current_ewma: float, new_val: float, alpha: float) -> float:
        if not (0.01 <= alpha <= 1.0):
            raise ValidationError(f"EWMA alpha must be between 0.01 and 1.0, got {alpha}")
        return alpha * new_val + (1.0 - alpha) * current_ewma

    @staticmethod
    def update_cusum(
        pos: float, neg: float, standardized_val: float, k_slack: float
    ) -> tuple[float, float]:
        if k_slack <= 0.0:
            raise ValidationError(f"CUSUM slack parameter k must be positive, got {k_slack}")
        new_pos = max(0.0, pos + standardized_val - k_slack)
        new_neg = max(0.0, neg - standardized_val - k_slack)
        return new_pos, new_neg

    @staticmethod
    def evaluate_triage(
        val: float,
        state: StreamingFilterStateModel,
        ewma_cfg: EWMAConfigDTO,
        cusum_cfg: CUSUMConfigDTO,
    ) -> FastPathTriageResultDTO:
        if state.sample_count < ewma_cfg.warmup_samples:
            return FastPathTriageResultDTO(
                metric_name=state.metric_name,
                raw_value=val,
                decision=FastPathDecision.BENIGN,
                ewma=state.ewma_value,
                z_score=0.0,
                requires_neural_inference=False,
            )

        std = math.sqrt(state.variance_estimate) if state.variance_estimate > 1e-8 else 1.0
        z_score = (val - state.mean_estimate) / std

        is_cusum_anomaly = (
            state.cusum_pos > cusum_cfg.h_threshold or state.cusum_neg > cusum_cfg.h_threshold
        )
        is_extreme_z = abs(z_score) > 4.0

        if is_extreme_z or is_cusum_anomaly:
            decision = FastPathDecision.ANOMALOUS
            requires_nn = False
        elif 1.5 < abs(z_score) <= 4.0:
            decision = FastPathDecision.AMBIGUOUS
            requires_nn = True
        else:
            decision = FastPathDecision.BENIGN
            requires_nn = False

        return FastPathTriageResultDTO(
            metric_name=state.metric_name,
            raw_value=val,
            decision=decision,
            ewma=state.ewma_value,
            z_score=z_score,
            requires_neural_inference=requires_nn,
        )
