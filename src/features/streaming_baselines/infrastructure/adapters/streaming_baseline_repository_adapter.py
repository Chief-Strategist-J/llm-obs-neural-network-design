from src.features.streaming_baselines.contracts.streaming_baseline_types import (
    StreamingFilterStateModel,
)
from src.features.streaming_baselines.infrastructure.ports.streaming_baseline_repository_port import (
    StreamingBaselineRepositoryPort,
)
from src.shared.types.envelope_types import TraceContext


class InMemoryStreamingBaselineRepository(StreamingBaselineRepositoryPort):
    def __init__(self) -> None:
        self._states: dict[str, StreamingFilterStateModel] = {}

    async def get_state(self, ctx: TraceContext, metric_name: str) -> StreamingFilterStateModel | None:
        return self._states.get(metric_name)

    async def save_state(self, ctx: TraceContext, state: StreamingFilterStateModel) -> StreamingFilterStateModel:
        self._states[state.metric_name] = state
        return state
