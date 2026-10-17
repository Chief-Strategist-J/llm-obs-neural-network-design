from src.shared.errors.app_errors import ValidationError

STATUS_SPECIFIED: str = "SPECIFIED"
STATUS_SPLIT_VALIDATED: str = "SPLIT_VALIDATED"
STATUS_BASELINE_ESTABLISHED: str = "BASELINE_ESTABLISHED"


class TelemetryPipelineMachine:
    _VALID_TRANSITIONS = {
        STATUS_SPECIFIED: {STATUS_SPLIT_VALIDATED},
        STATUS_SPLIT_VALIDATED: {STATUS_BASELINE_ESTABLISHED},
        STATUS_BASELINE_ESTABLISHED: set(),
    }

    @classmethod
    def transition(cls, current_state: str, next_state: str) -> str:
        allowed = cls._VALID_TRANSITIONS.get(current_state, set())
        if next_state not in allowed:
            raise ValidationError(
                f"Invalid pipeline state transition from {current_state} to {next_state}"
            )
        return next_state
