"""State machine transitions for network_spec lifecycle."""

from src.shared.constants.system_constants import STATUS_PENDING, STATUS_ACTIVE, STATUS_ARCHIVED
from src.shared.errors.app_errors import ValidationError


class NetworkSpecMachine:
    _VALID_TRANSITIONS = {
        STATUS_PENDING: {STATUS_ACTIVE, STATUS_ARCHIVED},
        STATUS_ACTIVE: {STATUS_ARCHIVED},
        STATUS_ARCHIVED: set(),
    }

    @classmethod
    def transition(cls, current_state: str, next_state: str) -> str:
        allowed = cls._VALID_TRANSITIONS.get(current_state, set())
        if next_state not in allowed:
            raise ValidationError(f"Invalid state transition from {current_state} to {next_state}")
        return next_state
