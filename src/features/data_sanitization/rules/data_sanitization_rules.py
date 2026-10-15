import math
from typing import Sequence
from src.features.data_sanitization.types.data_sanitization_types import (
    FeatureStatisticsDTO,
    SanitizationConfigDTO,
    CleanedSampleDTO,
)
from src.shared.errors.app_errors import ValidationError


class DataSanitizationRules:
    @staticmethod
    def compute_feature_statistics(
        raw_samples: Sequence[Sequence[float]],
    ) -> tuple[FeatureStatisticsDTO, ...]:
        if not raw_samples or not raw_samples[0]:
            raise ValidationError("Cannot compute feature statistics on empty sample matrix.")

        dim = len(raw_samples[0])
        stats: list[FeatureStatisticsDTO] = []

        for idx in range(dim):
            col_vals: list[float] = []
            for row in raw_samples:
                val = row[idx]
                if math.isnan(val) or math.isinf(val):
                    continue
                col_vals.append(float(val))

            if not col_vals:
                stats.append(
                    FeatureStatisticsDTO(
                        feature_index=idx,
                        mean=0.0,
                        std=1.0,
                        min_val=0.0,
                        max_val=0.0,
                        is_constant=True,
                    )
                )
                continue

            mean = sum(col_vals) / len(col_vals)
            variance = sum((x - mean) ** 2 for x in col_vals) / max(1, len(col_vals) - 1)
            std = math.sqrt(variance)
            min_val = min(col_vals)
            max_val = max(col_vals)
            is_constant = std < 1e-8 or (max_val - min_val) < 1e-8

            stats.append(
                FeatureStatisticsDTO(
                    feature_index=idx,
                    mean=mean,
                    std=std if std > 1e-8 else 1.0,
                    min_val=min_val,
                    max_val=max_val,
                    is_constant=is_constant,
                )
            )

        return tuple(stats)

    @staticmethod
    def sanitize_sample(
        sample_id: str,
        features: Sequence[float],
        label: int,
        statistics: Sequence[FeatureStatisticsDTO],
        config: SanitizationConfigDTO,
    ) -> CleanedSampleDTO:
        if len(features) != len(statistics):
            raise ValidationError(
                f"Feature dimension mismatch: expected {len(statistics)}, got {len(features)}"
            )
        if label < 0:
            raise ValidationError(f"Classification target label must be non-negative, got {label}")

        cleaned_feats: list[float] = []
        imputed_count = 0

        for idx, val in enumerate(features):
            stat = statistics[idx]
            v = float(val)

            if math.isnan(v):
                v = stat.mean if not stat.is_constant else config.impute_nan_value
                imputed_count += 1
            elif math.isinf(v):
                v = math.copysign(config.impute_inf_value, v)
                imputed_count += 1

            if config.clip_outliers and not stat.is_constant:
                lower = stat.mean - config.outlier_std_threshold * stat.std
                upper = stat.mean + config.outlier_std_threshold * stat.std
                if v < lower:
                    v = lower
                elif v > upper:
                    v = upper

            if config.normalize_zscore and not stat.is_constant:
                v = (v - stat.mean) / stat.std
            elif stat.is_constant:
                v = 0.0

            cleaned_feats.append(v)

        return CleanedSampleDTO(
            sample_id=sample_id,
            cleaned_features=tuple(cleaned_feats),
            target_label=int(label),
            imputed_count=imputed_count,
        )
