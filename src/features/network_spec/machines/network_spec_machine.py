from src.shared.errors.app_errors import ValidationError

STATUS_PENDING: str = "PENDING"
STATUS_ACTIVE: str = "ACTIVE"
STATUS_TINY_BATCH_OVERFIT_VERIFIED: str = "TINY_BATCH_OVERFIT_VERIFIED"
STATUS_ARCHIVED: str = "ARCHIVED"


class NetworkSpecMachine:
    _VALID_TRANSITIONS = {
        STATUS_PENDING: {STATUS_ACTIVE, STATUS_TINY_BATCH_OVERFIT_VERIFIED, STATUS_ARCHIVED},
        STATUS_TINY_BATCH_OVERFIT_VERIFIED: {STATUS_ACTIVE, STATUS_ARCHIVED},
        STATUS_ACTIVE: {STATUS_ARCHIVED},
        STATUS_ARCHIVED: set(),
    }

    @classmethod
    def transition(cls, current_state: str, next_state: str) -> str:
        allowed = cls._VALID_TRANSITIONS.get(current_state, set())
        if next_state not in allowed:
            raise ValidationError(f"Invalid state transition from {current_state} to {next_state}")
        return next_state
