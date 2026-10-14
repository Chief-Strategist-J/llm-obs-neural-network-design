"""Anti-corruption layer and schema validation for network_spec feature."""

from typing import Any, Mapping
from src.shared.errors.app_errors import ValidationError
from src.features.network_spec.types.network_spec_types import NetworkSpecCreateDTO, NetworkSpecModel


class NetworkSpecSchema:
    @staticmethod
    def from_api(payload: Mapping[str, Any]) -> NetworkSpecCreateDTO:
        name = payload.get("name")
        arch = payload.get("architecture_type")
        layers = payload.get("layer_count")
        budget = payload.get("parameter_budget")

        if not name or not isinstance(name, str):
            raise ValidationError("Field 'name' must be a non-empty string.")
        if not arch or arch not in ["transformer", "cnn", "mlp", "rnn", "hybrid"]:
            raise ValidationError("Field 'architecture_type' must be one of transformer, cnn, mlp, rnn, hybrid.")
        if not isinstance(layers, int) or layers < 1 or layers > 1000:
            raise ValidationError("Field 'layer_count' must be an integer between 1 and 1000.")
        if not isinstance(budget, int) or budget < 1:
            raise ValidationError("Field 'parameter_budget' must be a positive integer.")

        return NetworkSpecCreateDTO(
            name=name,
            architecture_type=arch,
            layer_count=layers,
            parameter_budget=budget,
        )

    @staticmethod
    def to_api(model: NetworkSpecModel) -> Mapping[str, Any]:
        return {
            "id": model.id,
            "name": model.name,
            "architecture_type": model.architecture_type,
            "layer_count": model.layer_count,
            "parameter_budget": model.parameter_budget,
            "status": model.status,
        }
