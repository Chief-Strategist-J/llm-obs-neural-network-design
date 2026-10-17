from abc import ABC, abstractmethod
from src.features.data_sanitization.contracts.data_sanitization_types import SanitizationProfileModel
from src.shared.types.envelope_types import TraceContext


class DataSanitizationRepositoryPort(ABC):
    @abstractmethod
    async def save_profile(
        self, ctx: TraceContext, profile: SanitizationProfileModel
    ) -> SanitizationProfileModel:
        pass

    @abstractmethod
    async def find_profile_by_id(
        self, ctx: TraceContext, profile_id: str
    ) -> SanitizationProfileModel | None:
        pass
