from typing import Protocol
from src.features.streaming_baselines.contracts.streaming_baseline_types import (
    StreamingFilterStateModel,
)
from src.shared.types.envelope_types import TraceContext


class StreamingBaselineRepositoryPort(Protocol):
    async def get_state(self, ctx: TraceContext, metric_name: str) -> StreamingFilterStateModel | None:
        ...

    async def save_state(self, ctx: TraceContext, state: StreamingFilterStateModel) -> StreamingFilterStateModel:
        ...
