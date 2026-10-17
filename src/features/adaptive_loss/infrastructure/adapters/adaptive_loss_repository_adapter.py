from src.features.adaptive_loss.contracts.adaptive_loss_types import ClassBalanceProfileModel
from src.features.adaptive_loss.infrastructure.ports.adaptive_loss_repository_port import AdaptiveLossRepositoryPort
from src.shared.types.envelope_types import TraceContext


class InMemoryAdaptiveLossRepository(AdaptiveLossRepositoryPort):
    def __init__(self) -> None:
        self._profiles: dict[str, ClassBalanceProfileModel] = {}

    async def save_profile(self, ctx: TraceContext, profile: ClassBalanceProfileModel) -> ClassBalanceProfileModel:
        self._profiles[profile.profile_id] = profile
        return profile

    async def find_profile_by_id(self, ctx: TraceContext, profile_id: str) -> ClassBalanceProfileModel | None:
        return self._profiles.get(profile_id)
