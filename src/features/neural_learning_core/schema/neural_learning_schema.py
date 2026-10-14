from typing import Any, Mapping
from src.shared.errors.app_errors import ValidationError
from src.features.neural_learning_core.types.neural_learning_types import (
    TinyBatchSampleDTO,
    OverfitVerificationConfigDTO,
    OverfitVerificationResult,
)


class NeuralLearningSchema:
    @staticmethod
    def from_api(payload: Mapping[str, Any]) -> OverfitVerificationConfigDTO:
        model_name = payload.get("model_name", "residual_telemetry_net")
        input_dim = payload.get("input_dim")
        hidden_dim = payload.get("hidden_dim", 64)
        num_classes = payload.get("num_classes", 2)
        samples_raw = payload.get("samples")
        lr = payload.get("learning_rate", 0.01)
        steps = payload.get("max_steps", 50)
        threshold = payload.get("target_loss_threshold", 0.05)

        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValidationError("Field 'input_dim' must be a positive integer.")
        if not isinstance(samples_raw, (list, tuple)) or len(samples_raw) == 0:
            raise ValidationError("Field 'samples' must contain at least 1 sample.")
        if len(samples_raw) > 100:
            raise ValidationError("Tiny-batch overfit requires <= 100 samples.")

        parsed_samples: list[TinyBatchSampleDTO] = []
        for s in samples_raw:
            if not isinstance(s, dict):
                raise ValidationError("Each sample must be an object.")
            feats = s.get("features", [])
            label = s.get("label", 0)
            if len(feats) != input_dim:
                raise ValidationError(f"Feature length {len(feats)} does not match input_dim {input_dim}.")
            parsed_samples.append(
                TinyBatchSampleDTO(
                    features=tuple(float(x) for x in feats),
                    label=int(label),
                )
            )

        return OverfitVerificationConfigDTO(
            model_name=str(model_name),
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            num_classes=int(num_classes),
            samples=tuple(parsed_samples),
            learning_rate=float(lr),
            max_steps=int(steps),
            target_loss_threshold=float(threshold),
        )

    @staticmethod
    def to_api(res: OverfitVerificationResult) -> Mapping[str, Any]:
        return {
            "initial_loss": res.initial_loss,
            "final_loss": res.final_loss,
            "steps_taken": res.steps_taken,
            "is_overfitted": res.is_overfitted,
            "status": res.status,
        }
