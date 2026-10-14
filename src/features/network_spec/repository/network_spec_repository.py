"""In-memory resilient adapter implementing NetworkSpecRepositoryPort."""

from src.features.network_spec.repository.network_spec_repository_port import NetworkSpecRepositoryPort
from src.features.network_spec.types.network_spec_types import NetworkSpecModel
from src.shared.types.envelope_types import TraceContext


class InMemoryNetworkSpecRepository(NetworkSpecRepositoryPort):
    def __init__(self) -> None:
        self._store: dict[str, NetworkSpecModel] = {}

    async def save(self, ctx: TraceContext, model: NetworkSpecModel) -> NetworkSpecModel:
        self._store[model.id] = model
        return model

    async def find_by_id(self, ctx: TraceContext, spec_id: str) -> NetworkSpecModel | None:
        return self._store.get(spec_id)
