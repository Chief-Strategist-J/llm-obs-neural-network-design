from src.features.neural_learning_core.types.neural_learning_types import (
    OverfitVerificationConfigDTO,
)


class NeuralStabilityRules:
    @staticmethod
    def validate_training_hyperparameters(config: OverfitVerificationConfigDTO) -> bool:
        if config.learning_rate <= 0 or config.learning_rate > 1.0:
            return False
        if config.max_steps < 5 or config.max_steps > 500:
            return False
        if config.hidden_dim < 4:
            return False
        return True
