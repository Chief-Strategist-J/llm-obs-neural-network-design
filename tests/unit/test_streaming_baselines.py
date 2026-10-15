import math
import pytest
from src.features.streaming_baselines.types.streaming_baseline_types import (
    EWMAConfigDTO,
    CUSUMConfigDTO,
    StreamingFilterStateModel,
    FastPathDecision,
)
from src.features.streaming_baselines.rules.streaming_baseline_rules import (
    StreamingBaselineRules,
)
from src.features.streaming_baselines.repository.streaming_baseline_repository import (
    InMemoryStreamingBaselineRepository,
)
from src.features.streaming_baselines.service.streaming_baseline_service import (
    StreamingBaselineService,
)
from src.shared.types.envelope_types import TraceContext
from src.shared.errors.app_errors import ValidationError


def test_welford_stats_and_ewma_rules():
    count, mean, m2 = 0, 0.0, 0.0
    for v in [10.0, 20.0, 30.0]:
        count, mean, m2 = StreamingBaselineRules.update_welford_stats(count, mean, m2, v)

    assert count == 3
    assert mean == 20.0
    variance = m2 / (count - 1)
    assert variance == 100.0

    ewma = 10.0
    ewma = StreamingBaselineRules.update_ewma(ewma, 20.0, alpha=0.5)
    assert ewma == 15.0

    with pytest.raises(ValidationError):
        StreamingBaselineRules.update_ewma(ewma, 20.0, alpha=1.5)


def test_cusum_slack_validation():
    with pytest.raises(ValidationError):
        StreamingBaselineRules.update_cusum(0.0, 0.0, 1.0, k_slack=-0.1)


def test_triage_decision_branches():
    state_warmup = StreamingFilterStateModel(
        metric_name="latency_ms",
        sample_count=5,
        mean_estimate=100.0,
        variance_estimate=25.0,
        ewma_value=100.0,
        cusum_pos=0.0,
        cusum_neg=0.0,
        last_updated_epoch_ms=0,
    )
    triage_warmup = StreamingBaselineRules.evaluate_triage(
        val=500.0,
        state=state_warmup,
        ewma_cfg=EWMAConfigDTO(warmup_samples=10),
        cusum_cfg=CUSUMConfigDTO(),
    )
    assert triage_warmup.decision == FastPathDecision.BENIGN
    assert triage_warmup.requires_neural_inference is False

    state_ready = StreamingFilterStateModel(
        metric_name="latency_ms",
        sample_count=50,
        mean_estimate=100.0,
        variance_estimate=100.0,
        ewma_value=100.0,
        cusum_pos=0.0,
        cusum_neg=0.0,
        last_updated_epoch_ms=0,
    )
    triage_benign = StreamingBaselineRules.evaluate_triage(
        val=105.0,
        state=state_ready,
        ewma_cfg=EWMAConfigDTO(),
        cusum_cfg=CUSUMConfigDTO(),
    )
    assert triage_benign.decision == FastPathDecision.BENIGN
    assert triage_benign.requires_neural_inference is False

    triage_ambiguous = StreamingBaselineRules.evaluate_triage(
        val=125.0,
        state=state_ready,
        ewma_cfg=EWMAConfigDTO(),
        cusum_cfg=CUSUMConfigDTO(),
    )
    assert triage_ambiguous.decision == FastPathDecision.AMBIGUOUS
    assert triage_ambiguous.requires_neural_inference is True

    triage_anomaly = StreamingBaselineRules.evaluate_triage(
        val=500.0,
        state=state_ready,
        ewma_cfg=EWMAConfigDTO(),
        cusum_cfg=CUSUMConfigDTO(),
    )
    assert triage_anomaly.decision == FastPathDecision.ANOMALOUS
    assert triage_anomaly.requires_neural_inference is False


@pytest.mark.asyncio
async def test_streaming_baseline_service_lifecycle():
    repo = InMemoryStreamingBaselineRepository()
    service = StreamingBaselineService(repository=repo)
    ctx = TraceContext(trace_id="trace-stream-001")

    for _ in range(15):
        res = await service.triage_span_metric(ctx, "duration_ms", 50.0)
        assert res.is_ok is True
        assert res.value.decision == FastPathDecision.BENIGN

    res_ambiguous = await service.triage_span_metric(ctx, "duration_ms", 150.0)
    assert res_ambiguous.is_ok is True
    assert res_ambiguous.value.requires_neural_inference is True
