import time
from src.features.streaming_baselines.rules.streaming_baseline_rules import (
    StreamingBaselineRules,
)
from src.features.streaming_baselines.types.streaming_baseline_types import (
    EWMAConfigDTO,
    CUSUMConfigDTO,
    StreamingFilterStateModel,
)


def test_streaming_baseline_microsecond_latency_benchmark():
    state = StreamingFilterStateModel(
        metric_name="latency_ms",
        sample_count=1000,
        mean_estimate=150.0,
        variance_estimate=400.0,
        ewma_value=150.0,
        cusum_pos=0.0,
        cusum_neg=0.0,
        last_updated_epoch_ms=0,
    )
    ewma_cfg = EWMAConfigDTO(alpha=0.2)
    cusum_cfg = CUSUMConfigDTO(k_slack=0.5, h_threshold=4.0)

    for _ in range(500):
        _ = StreamingBaselineRules.evaluate_triage(165.0, state, ewma_cfg, cusum_cfg)

    iterations = 20_000
    t0 = time.perf_counter()
    for _ in range(iterations):
        count, mean, m2 = StreamingBaselineRules.update_welford_stats(1000, 150.0, 400.0, 165.0)
        ewma = StreamingBaselineRules.update_ewma(150.0, 165.0, 0.2)
        pos, neg = StreamingBaselineRules.update_cusum(0.0, 0.0, 0.75, 0.5)
        _ = StreamingBaselineRules.evaluate_triage(165.0, state, ewma_cfg, cusum_cfg)
    total_time = time.perf_counter() - t0

    latency_us_per_op = (total_time / iterations) * 1_000_000.0
    assert latency_us_per_op < 15.0
