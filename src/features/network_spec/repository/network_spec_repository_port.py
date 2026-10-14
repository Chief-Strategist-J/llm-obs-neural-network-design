"""Outbound repository port contract for network_spec persistence."""

from abc import ABC, abstractmethod
from src.features.network_spec.types.network_spec_types import NetworkSpecModel
from src.shared.types.envelope_types import TraceContext


class NetworkSpecRepositoryPort(ABC):
    @abstractmethod
    async def save(self, ctx: TraceContext, model: NetworkSpecModel) -> NetworkSpecModel:
        pass

    @abstractmethod
    async def find_by_id(self, ctx: TraceContext, spec_id: str) -> NetworkSpecModel | None:
        pass
