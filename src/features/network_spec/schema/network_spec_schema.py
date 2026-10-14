from typing import Any, Mapping
from src.shared.errors.app_errors import ValidationError
from src.features.network_spec.types.network_spec_types import (
    NetworkSpecCreateDTO,
    NetworkSpecModel,
    TelemetrySampleDTO,
    TinyBatchOverfitCommand,
)


class NetworkSpecSchema:
    @staticmethod
    def from_api(payload: Mapping[str, Any]) -> NetworkSpecCreateDTO:
        name = payload.get("name")
        arch = payload.get("architecture_type", "transformer")
        input_dim = payload.get("input_dim", 16)
        hidden_dim = payload.get("hidden_dim", 64)
        num_classes = payload.get("num_classes", 2)
        layers = payload.get("layer_count", 4)
        budget = payload.get("parameter_budget", 1_000_000)

        if not name or not isinstance(name, str):
            raise ValidationError("Field 'name' must be a non-empty string.")
        if not arch or arch not in ["transformer", "cnn", "mlp", "rnn", "hybrid"]:
            raise ValidationError("Field 'architecture_type' must be one of transformer, cnn, mlp, rnn, hybrid.")
        if not isinstance(input_dim, int) or input_dim < 1:
            raise ValidationError("Field 'input_dim' must be a positive integer.")
        if not isinstance(hidden_dim, int) or hidden_dim < 4:
            raise ValidationError("Field 'hidden_dim' must be >= 4.")
        if not isinstance(layers, int) or layers < 1 or layers > 1000:
            raise ValidationError("Field 'layer_count' must be an integer between 1 and 1000.")
        if not isinstance(budget, int) or budget < 1:
            raise ValidationError("Field 'parameter_budget' must be a positive integer.")

        return NetworkSpecCreateDTO(
            name=name,
            architecture_type=arch,
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            num_classes=int(num_classes),
            layer_count=layers,
            parameter_budget=budget,
        )

    @staticmethod
    def parse_overfit_command(payload: Mapping[str, Any]) -> TinyBatchOverfitCommand:
        spec_id = payload.get("spec_id")
        raw_samples = payload.get("samples")
        steps = payload.get("max_steps", 50)
        threshold = payload.get("target_loss_threshold", 0.08)

        if not spec_id:
            raise ValidationError("Field 'spec_id' is required.")
        if not isinstance(raw_samples, (list, tuple)) or len(raw_samples) == 0:
            raise ValidationError("Field 'samples' must contain at least 1 sample.")

        parsed_samples: list[TelemetrySampleDTO] = []
        for s in raw_samples:
            if not isinstance(s, dict):
                raise ValidationError("Each sample must be an object.")
            feats = s.get("features", [])
            label = s.get("label", 0)
            parsed_samples.append(
                TelemetrySampleDTO(
                    features=tuple(float(x) for x in feats),
                    label=int(label),
                )
            )

        return TinyBatchOverfitCommand(
            spec_id=str(spec_id),
            samples=tuple(parsed_samples),
            max_steps=int(steps),
            target_loss_threshold=float(threshold),
        )

    @staticmethod
    def to_api(model: NetworkSpecModel) -> Mapping[str, Any]:
        return {
            "id": model.id,
            "name": model.name,
            "architecture_type": model.architecture_type,
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "num_classes": model.num_classes,
            "layer_count": model.layer_count,
            "parameter_budget": model.parameter_budget,
            "status": model.status,
            "initial_loss": model.initial_loss,
            "final_loss": model.final_loss,
            "steps_taken": model.steps_taken,
            "is_overfit_verified": model.is_overfit_verified,
        }
