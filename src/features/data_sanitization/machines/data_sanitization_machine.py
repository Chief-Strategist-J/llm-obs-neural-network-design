from src.shared.errors.app_errors import ValidationError

STATUS_RAW: str = "RAW"
STATUS_FITTED: str = "FITTED"
STATUS_ACTIVE: str = "ACTIVE"
STATUS_DEPRECATED: str = "DEPRECATED"


class DataSanitizationMachine:
    _VALID_TRANSITIONS = {
        STATUS_RAW: {STATUS_FITTED, STATUS_DEPRECATED},
        STATUS_FITTED: {STATUS_ACTIVE, STATUS_DEPRECATED},
        STATUS_ACTIVE: {STATUS_DEPRECATED},
        STATUS_DEPRECATED: set(),
    }

    @classmethod
    def transition(cls, current_state: str, next_state: str) -> str:
        allowed = cls._VALID_TRANSITIONS.get(current_state, set())
        if next_state not in allowed:
            raise ValidationError(
                f"Invalid sanitization state transition from {current_state} to {next_state}"
            )
        return next_state
