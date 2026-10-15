from typing import Sequence
from src.features.adaptive_loss.types.adaptive_loss_types import ClassBalanceProfileModel
from src.features.adaptive_loss.rules.adaptive_loss_rules import AdaptiveLossRules
from src.features.adaptive_loss.repository.adaptive_loss_repository import AdaptiveLossRepositoryPort
from src.shared.types.envelope_types import TraceContext
from src.shared.utils.date_utils import generate_uuidv7_str, get_utc_iso_timestamp


class AdaptiveLossWorkflow:
    def __init__(self, repo: AdaptiveLossRepositoryPort) -> None:
        self._repo = repo

    async def execute_calculate_profile_flow(
        self,
        ctx: TraceContext,
        class_counts: Sequence[int],
        beta: float = 0.999,
    ) -> ClassBalanceProfileModel:
        weights = AdaptiveLossRules.calculate_effective_sample_weights(class_counts, beta=beta)
        profile = ClassBalanceProfileModel(
            profile_id=generate_uuidv7_str(),
            class_counts=tuple(class_counts),
            calculated_weights=weights,
            beta_used=beta,
            created_at_iso=get_utc_iso_timestamp(),
        )
        return await self._repo.save_profile(ctx, profile)
