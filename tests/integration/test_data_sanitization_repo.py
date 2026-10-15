import pytest
import polars as pl
from src.features.data_sanitization.index import (
    InMemoryDataSanitizationRepository,
    SanitizationProfileModel,
    SanitizationConfigDTO,
    FeatureStatisticsDTO,
    STATUS_RAW,
    STATUS_FITTED,
    STATUS_ACTIVE,
    STATUS_DEPRECATED,
    DataSanitizationMachine,
    DataSanitizationRules,
)
from src.shared.types.envelope_types import TraceContext
from src.shared.utils.date_utils import generate_uuidv7_str, get_utc_iso_timestamp


@pytest.mark.asyncio
async def test_data_sanitization_repository_and_machine_lifecycle():
    ctx = TraceContext(trace_id="integration-sanitization-repo-001")
    repo = InMemoryDataSanitizationRepository()

    stats = (
        FeatureStatisticsDTO(feature_index=0, mean=100.0, std=15.0, min_val=50.0, max_val=150.0, is_constant=False),
        FeatureStatisticsDTO(feature_index=1, mean=0.0, std=1.0, min_val=0.0, max_val=0.0, is_constant=True),
    )
    profile_id = generate_uuidv7_str()
    profile = SanitizationProfileModel(
        profile_id=profile_id,
        feature_count=2,
        statistics=stats,
        config=SanitizationConfigDTO(),
        status=STATUS_FITTED,
        created_at_iso=get_utc_iso_timestamp(),
    )

    saved = await repo.save_profile(ctx, profile)
    assert saved.profile_id == profile_id

    fetched = await repo.find_profile_by_id(ctx, profile_id)
    assert fetched is not None
    assert fetched.feature_count == 2
    assert fetched.status == STATUS_FITTED
    assert len(fetched.statistics) == 2

    new_status = DataSanitizationMachine.transition(fetched.status, STATUS_ACTIVE)
    assert new_status == STATUS_ACTIVE

    active_profile = SanitizationProfileModel(
        profile_id=fetched.profile_id,
        feature_count=fetched.feature_count,
        statistics=fetched.statistics,
        config=fetched.config,
        status=new_status,
        created_at_iso=fetched.created_at_iso,
    )
    await repo.save_profile(ctx, active_profile)

    refetched = await repo.find_profile_by_id(ctx, profile_id)
    assert refetched is not None
    assert refetched.status == STATUS_ACTIVE


def test_polars_streaming_batch_pipeline_integration():
    data = {
        "span_duration": [12.0, 24.0, 36.0, 48.0, 60.0, None, float("nan"), 10000.0],
        "request_size": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
        "zero_variance": [42.0] * 8,
    }
    df = pl.DataFrame(data)

    stats = DataSanitizationRules.compute_feature_statistics_polars(
        df=df,
        feature_columns=["span_duration", "request_size", "zero_variance"],
    )
    assert len(stats) == 3
    assert stats[2].is_constant is True

    cfg = SanitizationConfigDTO(
        impute_nan_value=0.0,
        impute_inf_value=1e6,
        clip_outliers=True,
        outlier_std_threshold=3.0,
        normalize_zscore=True,
    )

    cleaned_df = DataSanitizationRules.sanitize_dataframe_polars(
        df=df,
        feature_columns=["span_duration", "request_size", "zero_variance"],
        statistics=stats,
        config=cfg,
    )

    assert cleaned_df["span_duration"].null_count() == 0
    assert cleaned_df["request_size"].null_count() == 0
    assert cleaned_df["zero_variance"].null_count() == 0
    assert all(abs(v) <= 3.01 for v in cleaned_df["span_duration"].to_list())
    assert cleaned_df["zero_variance"].to_list() == [0.0] * 8
