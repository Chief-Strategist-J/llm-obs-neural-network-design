from src.shared.errors.app_errors import ValidationError

STATUS_INITIALIZED: str = "INITIALIZED"
STATUS_TINY_BATCH_OVERFIT_PASSED: str = "TINY_BATCH_OVERFIT_PASSED"
STATUS_TINY_BATCH_OVERFIT_FAILED: str = "TINY_BATCH_OVERFIT_FAILED"


class NeuralTrainingMachine:
    _VALID_TRANSITIONS = {
        STATUS_INITIALIZED: {STATUS_TINY_BATCH_OVERFIT_PASSED, STATUS_TINY_BATCH_OVERFIT_FAILED},
        STATUS_TINY_BATCH_OVERFIT_PASSED: set(),
        STATUS_TINY_BATCH_OVERFIT_FAILED: set(),
    }

    @classmethod
    def transition(cls, current_state: str, next_state: str) -> str:
        allowed = cls._VALID_TRANSITIONS.get(current_state, set())
        if next_state not in allowed:
            raise ValidationError(
                f"Invalid training state transition from {current_state} to {next_state}"
            )
        return next_state
