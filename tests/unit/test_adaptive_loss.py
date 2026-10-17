import math
import pytest
import torch
from src.features.adaptive_loss.contracts.adaptive_loss_types import (
    LossType,
    AdaptiveLossConfigDTO,
)
from src.features.adaptive_loss.domain.rules.adaptive_loss_rules import AdaptiveLossRules
from src.features.adaptive_loss.infrastructure.adapters.torch_loss_modules import (
    FocalLoss,
    ClassBalancedLoss,
)
from src.features.adaptive_loss.infrastructure.adapters.adaptive_loss_repository_adapter import (
    InMemoryAdaptiveLossRepository,
)
from src.features.adaptive_loss.service.adaptive_loss_service import (
    AdaptiveLossService,
)
from src.shared.types.envelope_types import TraceContext
from src.shared.errors.app_errors import ValidationError


def test_effective_sample_weights_calculation():
    counts = [1000, 10]
    weights = AdaptiveLossRules.calculate_effective_sample_weights(counts, beta=0.999)

    assert len(weights) == 2
    assert weights[1] > weights[0]
    assert math.isclose(sum(weights), 2.0, rel_tol=1e-4)

    balanced_counts = [500, 500]
    balanced_weights = AdaptiveLossRules.calculate_effective_sample_weights(balanced_counts, beta=0.999)
    assert math.isclose(balanced_weights[0], balanced_weights[1], rel_tol=1e-4)


def test_effective_sample_weights_validation():
    with pytest.raises(ValidationError):
        AdaptiveLossRules.calculate_effective_sample_weights([])

    with pytest.raises(ValidationError):
        AdaptiveLossRules.calculate_effective_sample_weights([10, -5])

    with pytest.raises(ValidationError):
        AdaptiveLossRules.calculate_effective_sample_weights([10, 10], beta=1.5)


def test_focal_loss_numerical_stability():
    focal = FocalLoss(gamma=2.0)
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    targets = torch.tensor([0, 1])

    loss_easy = focal(logits, targets)
    assert not torch.isnan(loss_easy)
    assert not torch.isinf(loss_easy)
    assert float(loss_easy) < 1e-4

    logits_hard = torch.tensor([[-5.0, 5.0], [5.0, -5.0]])
    loss_hard = focal(logits_hard, targets)
    assert float(loss_hard) > float(loss_easy)


def test_class_balanced_loss_module():
    cb_loss = ClassBalancedLoss(class_weights=[0.1, 1.9], gamma=2.0)
    logits = torch.randn(8, 2, requires_grad=True)
    targets = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])

    loss = cb_loss(logits, targets)
    loss.backward()

    assert not torch.isnan(loss)
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


@pytest.mark.asyncio
async def test_adaptive_loss_service_workflow():
    repo = InMemoryAdaptiveLossRepository()
    service = AdaptiveLossService(repository=repo)
    ctx = TraceContext(trace_id="adaptive-loss-trace-001")

    res = await service.calculate_class_balance_profile(ctx, [5000, 50], beta=0.999)
    assert res.is_ok is True
    profile = res.value
    assert profile is not None
    assert profile.calculated_weights[1] > profile.calculated_weights[0]

    fetched_res = await service.get_profile(ctx, profile.profile_id)
    assert fetched_res.is_ok is True
    assert fetched_res.value is not None
    assert fetched_res.value.profile_id == profile.profile_id
