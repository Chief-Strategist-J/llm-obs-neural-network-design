from src.features.data_sanitization.contracts.data_sanitization_types import SanitizationProfileModel
from src.features.data_sanitization.infrastructure.ports.data_sanitization_repository_port import DataSanitizationRepositoryPort
from src.shared.types.envelope_types import TraceContext


class InMemoryDataSanitizationRepository(DataSanitizationRepositoryPort):
    def __init__(self) -> None:
        self._profiles: dict[str, SanitizationProfileModel] = {}

    async def save_profile(
        self, ctx: TraceContext, profile: SanitizationProfileModel
    ) -> SanitizationProfileModel:
        self._profiles[profile.profile_id] = profile
        return profile

    async def find_profile_by_id(
        self, ctx: TraceContext, profile_id: str
    ) -> SanitizationProfileModel | None:
        return self._profiles.get(profile_id)
