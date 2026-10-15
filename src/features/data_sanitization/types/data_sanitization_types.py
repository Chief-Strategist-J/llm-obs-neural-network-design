from dataclasses import dataclass
from typing import Sequence, Mapping, Any


@dataclass(frozen=True)
class FeatureStatisticsDTO:
    feature_index: int
    mean: float
    std: float
    min_val: float
    max_val: float
    is_constant: bool


@dataclass(frozen=True)
class SanitizationConfigDTO:
    impute_nan_value: float = 0.0
    impute_inf_value: float = 1e6
    clip_outliers: bool = True
    outlier_std_threshold: float = 5.0
    normalize_zscore: bool = True


@dataclass(frozen=True)
class CleanedSampleDTO:
    sample_id: str
    cleaned_features: Sequence[float]
    target_label: int
    imputed_count: int


@dataclass(frozen=True)
class SanitizationProfileModel:
    profile_id: str
    feature_count: int
    statistics: Sequence[FeatureStatisticsDTO]
    config: SanitizationConfigDTO
    status: str
    created_at_iso: str
