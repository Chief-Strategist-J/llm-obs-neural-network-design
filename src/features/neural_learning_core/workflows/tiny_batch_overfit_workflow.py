from src.features.neural_learning_core.types.neural_learning_types import (
    OverfitVerificationConfigDTO,
    OverfitVerificationResult,
)
from src.features.neural_learning_core.rules.neural_stability_rules import NeuralStabilityRules
from src.features.neural_learning_core.machines.neural_training_machine import (
    NeuralTrainingMachine,
    STATUS_INITIALIZED,
)
from src.features.neural_learning_core.repository.pytorch_model_adapter import (
    PyTorchResidualTelemetryModel,
)
from src.shared.errors.app_errors import ValidationError


class TinyBatchOverfitWorkflow:
    @staticmethod
    def execute_overfit_verification(
        config: OverfitVerificationConfigDTO,
    ) -> OverfitVerificationResult:
        if not NeuralStabilityRules.validate_training_hyperparameters(config):
            raise ValidationError("Hyperparameters violate stability constraints.")

        current_state = STATUS_INITIALIZED

        model = PyTorchResidualTelemetryModel(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
        )

        res = model.fit_tiny_batch(config)
        next_state = NeuralTrainingMachine.transition(current_state, res.status)

        return OverfitVerificationResult(
            initial_loss=res.initial_loss,
            final_loss=res.final_loss,
            steps_taken=res.steps_taken,
            is_overfitted=res.is_overfitted,
            status=next_state,
        )
