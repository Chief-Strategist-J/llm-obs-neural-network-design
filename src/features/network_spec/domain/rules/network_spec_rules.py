"""Pure deterministic decision rules for network_spec feature."""

from src.features.network_spec.contracts.network_spec_types import NetworkSpecCreateDTO


class NetworkSpecRules:
    @staticmethod
    def can_create_spec(dto: NetworkSpecCreateDTO) -> bool:
        if dto.layer_count > 500 and dto.parameter_budget < 1_000_000:
            return False
        return True
