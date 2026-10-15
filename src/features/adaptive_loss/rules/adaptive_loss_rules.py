import math
from typing import Sequence
from src.shared.errors.app_errors import ValidationError


class AdaptiveLossRules:
    @staticmethod
    def calculate_effective_sample_weights(
        class_counts: Sequence[int],
        beta: float = 0.999,
    ) -> tuple[float, ...]:
        if not class_counts:
            raise ValidationError("Class counts sequence cannot be empty.")
        if not (0.0 < beta < 1.0):
            raise ValidationError(f"Beta must be in range (0.0, 1.0), got {beta}")

        num_classes = len(class_counts)
        effective_num = []
        for count in class_counts:
            if count < 0:
                raise ValidationError(f"Class count cannot be negative: {count}")
            if count == 0:
                effective_num.append(1e-8)
            else:
                eff = (1.0 - math.pow(beta, count)) / (1.0 - beta)
                effective_num.append(max(eff, 1e-8))

        raw_weights = [1.0 / eff for eff in effective_num]
        total_weight = sum(raw_weights)
        if total_weight <= 0.0:
            return tuple(1.0 for _ in range(num_classes))

        normalized_weights = [(w / total_weight) * float(num_classes) for w in raw_weights]
        return tuple(normalized_weights)

    @staticmethod
    def validate_focal_parameters(gamma: float, alpha: Sequence[float] | None, num_classes: int) -> None:
        if gamma < 0.0:
            raise ValidationError(f"Focal loss gamma must be non-negative, got {gamma}")
        if alpha is not None:
            if len(alpha) != num_classes:
                raise ValidationError(
                    f"Alpha sequence length ({len(alpha)}) must match number of classes ({num_classes})"
                )
            for idx, a in enumerate(alpha):
                if a < 0.0:
                    raise ValidationError(f"Alpha weight for class {idx} cannot be negative: {a}")
