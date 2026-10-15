from typing import Protocol
from src.features.streaming_baselines.types.streaming_baseline_types import (
    StreamingFilterStateModel,
)
from src.shared.types.envelope_types import TraceContext


class StreamingBaselineRepositoryPort(Protocol):
    async def get_state(self, ctx: TraceContext, metric_name: str) -> StreamingFilterStateModel | None:
        ...

    async def save_state(self, ctx: TraceContext, state: StreamingFilterStateModel) -> StreamingFilterStateModel:
        ...


class InMemoryStreamingBaselineRepository(StreamingBaselineRepositoryPort):
    def __init__(self) -> None:
        self._states: dict[str, StreamingFilterStateModel] = {}

    async def get_state(self, ctx: TraceContext, metric_name: str) -> StreamingFilterStateModel | None:
        return self._states.get(metric_name)

    async def save_state(self, ctx: TraceContext, state: StreamingFilterStateModel) -> StreamingFilterStateModel:
        self._states[state.metric_name] = state
        return state
