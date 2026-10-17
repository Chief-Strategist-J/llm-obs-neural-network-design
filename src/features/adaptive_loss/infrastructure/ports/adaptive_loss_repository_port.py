from typing import Protocol
from src.features.adaptive_loss.contracts.adaptive_loss_types import ClassBalanceProfileModel
from src.shared.types.envelope_types import TraceContext


class AdaptiveLossRepositoryPort(Protocol):
    async def save_profile(self, ctx: TraceContext, profile: ClassBalanceProfileModel) -> ClassBalanceProfileModel:
        ...

    async def find_profile_by_id(self, ctx: TraceContext, profile_id: str) -> ClassBalanceProfileModel | None:
        ...
