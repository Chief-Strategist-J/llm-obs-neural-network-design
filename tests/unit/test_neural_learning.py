import pytest
from src.features.neural_learning_core.types.neural_learning_types import (
    TinyBatchSampleDTO,
    OverfitVerificationConfigDTO,
)
from src.features.neural_learning_core.service.neural_learning_service import (
    NeuralLearningService,
)
from src.features.neural_learning_core.machines.neural_training_machine import (
    STATUS_TINY_BATCH_OVERFIT_PASSED,
)
from src.shared.types.envelope_types import TraceContext


@pytest.mark.asyncio
async def test_tiny_batch_overfitting_with_pytorch():
    samples = (
        TinyBatchSampleDTO(features=(1.0, 0.0, -1.0, 0.5), label=0),
        TinyBatchSampleDTO(features=(0.0, 1.0, 1.0, -0.5), label=1),
        TinyBatchSampleDTO(features=(1.2, -0.1, -0.8, 0.4), label=0),
        TinyBatchSampleDTO(features=(-0.2, 0.9, 1.1, -0.4), label=1),
    )
    config = OverfitVerificationConfigDTO(
        model_name="test_residual_net",
        input_dim=4,
        hidden_dim=32,
        num_classes=2,
        samples=samples,
        learning_rate=0.05,
        max_steps=50,
        target_loss_threshold=0.08,
    )
    ctx = TraceContext(trace_id="test-overfit-trace")

    result = await NeuralLearningService.verify_tiny_batch_overfit(ctx, config)

    assert result.is_ok is True
    assert result.value is not None
    assert result.value.is_overfitted is True
    assert result.value.final_loss < result.value.initial_loss
    assert result.value.final_loss <= 0.08
    assert result.value.status == STATUS_TINY_BATCH_OVERFIT_PASSED
