import math
import pytest
from src.features.data_sanitization.types.data_sanitization_types import (
    SanitizationConfigDTO,
)
from src.features.data_sanitization.rules.data_sanitization_rules import (
    DataSanitizationRules,
)
from src.features.data_sanitization.machines.data_sanitization_machine import (
    DataSanitizationMachine,
    STATUS_RAW,
    STATUS_FITTED,
    STATUS_ACTIVE,
    STATUS_DEPRECATED,
)
from src.features.data_sanitization.repository.data_sanitization_repository import (
    InMemoryDataSanitizationRepository,
)
from src.features.data_sanitization.service.data_sanitization_service import (
    DataSanitizationService,
)
from src.shared.types.envelope_types import TraceContext
from src.shared.errors.app_errors import ValidationError


def test_sanitization_rules_nan_and_inf_handling():
    raw_matrix = [
        [10.0, 1.0, 100.0],
        [20.0, 2.0, 200.0],
        [30.0, 3.0, 300.0],
    ]
    stats = DataSanitizationRules.compute_feature_statistics(raw_matrix)
    assert len(stats) == 3
    assert stats[0].mean == 20.0
    assert stats[1].mean == 2.0

    dirty_sample = [float("nan"), float("inf"), -float("inf")]
    cfg = SanitizationConfigDTO(clip_outliers=True, normalize_zscore=True)
    cleaned = DataSanitizationRules.sanitize_sample(
        sample_id="dirty_1",
        features=dirty_sample,
        label=1,
        statistics=stats,
        config=cfg,
    )
    assert cleaned.imputed_count == 3
    for v in cleaned.cleaned_features:
        assert not math.isnan(v)
        assert not math.isinf(v)


def test_sanitization_rules_outlier_clipping_and_zscore():
    raw_matrix = [
        [10.0, 5.0],
        [20.0, 5.0],
        [30.0, 5.0],
    ]
    stats = DataSanitizationRules.compute_feature_statistics(raw_matrix)
    assert stats[1].is_constant is True

    cfg = SanitizationConfigDTO(clip_outliers=True, outlier_std_threshold=2.0, normalize_zscore=True)
    outlier_sample = [1000.0, 5.0]
    cleaned = DataSanitizationRules.sanitize_sample(
        sample_id="outlier_1",
        features=outlier_sample,
        label=0,
        statistics=stats,
        config=cfg,
    )
    assert cleaned.cleaned_features[0] == 2.0
    assert cleaned.cleaned_features[1] == 0.0


def test_sanitization_machine_transitions():
    assert DataSanitizationMachine.transition(STATUS_RAW, STATUS_FITTED) == STATUS_FITTED
    assert DataSanitizationMachine.transition(STATUS_FITTED, STATUS_ACTIVE) == STATUS_ACTIVE
    assert DataSanitizationMachine.transition(STATUS_ACTIVE, STATUS_DEPRECATED) == STATUS_DEPRECATED

    with pytest.raises(ValidationError):
        DataSanitizationMachine.transition(STATUS_DEPRECATED, STATUS_ACTIVE)


@pytest.mark.asyncio
async def test_sanitization_service_fit_and_transform():
    repo = InMemoryDataSanitizationRepository()
    service = DataSanitizationService(repository=repo)
    ctx = TraceContext(trace_id="sanitization-trace-001")

    train_matrix = [
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ]
    fit_res = await service.fit_profile(ctx, train_matrix)
    assert fit_res.is_ok is True
    profile = fit_res.value
    assert profile is not None
    assert profile.feature_count == 2
    assert profile.status == STATUS_FITTED

    samples_to_clean = [
        ("s1", [float("nan"), 15.0], 0),
        ("s2", [2.0, float("inf")], 1),
    ]
    transform_res = await service.transform_samples(ctx, profile.profile_id, samples_to_clean)
    assert transform_res.is_ok is True
    cleaned_tuples = transform_res.value
    assert len(cleaned_tuples) == 2
    assert cleaned_tuples[0].sample_id == "s1"
    assert cleaned_tuples[1].sample_id == "s2"
