from typing import Any, Mapping, Sequence
from src.shared.errors.app_errors import ValidationError
from src.features.network_spec.contracts.network_spec_types import (
    NetworkSpecCreateDTO,
    NetworkSpecModel,
    TelemetrySampleDTO,
    TinyBatchOverfitCommand,
    DatasetTrainingCommand,
    DatasetTrainingResult,
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
    def parse_training_command(payload: Mapping[str, Any]) -> DatasetTrainingCommand:
        spec_id = payload.get("spec_id")
        raw_train = payload.get("train_samples")
        raw_val = payload.get("val_samples")
        epochs = payload.get("epochs", 10)
        batch_size = payload.get("batch_size", 32)
        lr = payload.get("learning_rate", 0.001)
        decay = payload.get("weight_decay", 1e-4)
        ckpt_dir = payload.get("target_checkpoint_dir", "checkpoints")
        loss_type = payload.get("loss_type", "cross_entropy")
        focal_gamma = payload.get("focal_gamma", 2.0)
        raw_weights = payload.get("class_weights")
        class_weights = tuple(float(w) for w in raw_weights) if raw_weights else None

        if not spec_id:
            raise ValidationError("Field 'spec_id' is required.")
        if not isinstance(raw_train, (list, tuple)) or len(raw_train) == 0:
            raise ValidationError("Field 'train_samples' must contain at least 1 sample.")
        if not isinstance(raw_val, (list, tuple)) or len(raw_val) == 0:
            raise ValidationError("Field 'val_samples' must contain at least 1 sample.")

        def _parse_samples(raw_list: Sequence[Any]) -> list[TelemetrySampleDTO]:
            parsed: list[TelemetrySampleDTO] = []
            for item in raw_list:
                if not isinstance(item, dict):
                    raise ValidationError("Each sample must be an object.")
                parsed.append(
                    TelemetrySampleDTO(
                        features=tuple(float(x) for x in item.get("features", [])),
                        label=int(item.get("label", 0)),
                    )
                )
            return parsed

        return DatasetTrainingCommand(
            spec_id=str(spec_id),
            train_samples=tuple(_parse_samples(raw_train)),
            val_samples=tuple(_parse_samples(raw_val)),
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(lr),
            weight_decay=float(decay),
            target_checkpoint_dir=str(ckpt_dir),
            loss_type=str(loss_type),
            focal_gamma=float(focal_gamma),
            class_weights=class_weights,
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

    @staticmethod
    def training_result_to_api(res: DatasetTrainingResult) -> Mapping[str, Any]:
        return {
            "spec_id": res.spec_id,
            "trained_epochs": res.trained_epochs,
            "train_loss": res.train_loss,
            "val_loss": res.val_loss,
            "val_accuracy": res.val_accuracy,
            "checkpoint_path": res.checkpoint_path,
            "checkpoint_sha256": res.checkpoint_sha256,
        }
