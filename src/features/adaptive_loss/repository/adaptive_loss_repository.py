from typing import Protocol
from src.features.adaptive_loss.types.adaptive_loss_types import ClassBalanceProfileModel
from src.shared.types.envelope_types import TraceContext


class AdaptiveLossRepositoryPort(Protocol):
    async def save_profile(self, ctx: TraceContext, profile: ClassBalanceProfileModel) -> ClassBalanceProfileModel:
        ...

    async def find_profile_by_id(self, ctx: TraceContext, profile_id: str) -> ClassBalanceProfileModel | None:
        ...


class InMemoryAdaptiveLossRepository(AdaptiveLossRepositoryPort):
    def __init__(self) -> None:
        self._profiles: dict[str, ClassBalanceProfileModel] = {}

    async def save_profile(self, ctx: TraceContext, profile: ClassBalanceProfileModel) -> ClassBalanceProfileModel:
        self._profiles[profile.profile_id] = profile
        return profile

    async def find_profile_by_id(self, ctx: TraceContext, profile_id: str) -> ClassBalanceProfileModel | None:
        return self._profiles.get(profile_id)
