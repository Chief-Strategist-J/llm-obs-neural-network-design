import time
import random
import polars as pl
from src.features.data_sanitization.domain.rules.data_sanitization_rules import DataSanitizationRules
from src.features.data_sanitization.contracts.data_sanitization_types import SanitizationConfigDTO


def test_polars_scale_benchmark_100k_rows():
    row_count = 100_000
    random.seed(42)

    latency_data = [random.gauss(150.0, 30.0) if random.random() > 0.05 else None for _ in range(row_count)]
    latency_data[10] = 50000.0
    latency_data[20] = float("nan")

    tokens_data = [float(random.randint(50, 2000)) for _ in range(row_count)]
    cost_data = [0.002 * t for t in tokens_data]

    df = pl.DataFrame({
        "latency_ms": latency_data,
        "token_count": tokens_data,
        "estimated_cost": cost_data,
    })

    t0 = time.perf_counter()
    stats = DataSanitizationRules.compute_feature_statistics_polars(
        df=df,
        feature_columns=["latency_ms", "token_count", "estimated_cost"],
    )
    t_fit = time.perf_counter() - t0

    assert len(stats) == 3
    assert t_fit < 1.0

    cfg = SanitizationConfigDTO(
        impute_nan_value=0.0,
        impute_inf_value=1e6,
        clip_outliers=True,
        outlier_std_threshold=3.0,
        normalize_zscore=True,
    )

    t1 = time.perf_counter()
    cleaned_df = DataSanitizationRules.sanitize_dataframe_polars(
        df=df,
        feature_columns=["latency_ms", "token_count", "estimated_cost"],
        statistics=stats,
        config=cfg,
    )
    t_transform = time.perf_counter() - t1

    assert cleaned_df.shape == (row_count, 3)
    assert cleaned_df["latency_ms"].null_count() == 0
    assert cleaned_df["token_count"].null_count() == 0
    assert cleaned_df["estimated_cost"].null_count() == 0
    assert t_transform < 1.0
