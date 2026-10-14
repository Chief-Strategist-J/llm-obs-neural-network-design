from src.features.system_telemetry_pipeline.repository.telemetry_pipeline_repository_port import (
    TelemetryPipelineRepositoryPort,
)
from src.features.system_telemetry_pipeline.types.telemetry_pipeline_types import (
    TelemetryPipelineModel,
)
from src.shared.types.envelope_types import TraceContext


class InMemoryTelemetryPipelineRepository(TelemetryPipelineRepositoryPort):
    def __init__(self) -> None:
        self._store: dict[str, TelemetryPipelineModel] = {}

    async def save(
        self, ctx: TraceContext, model: TelemetryPipelineModel
    ) -> TelemetryPipelineModel:
        self._store[model.pipeline_id] = model
        return model

    async def find_by_id(
        self, ctx: TraceContext, pipeline_id: str
    ) -> TelemetryPipelineModel | None:
        return self._store.get(pipeline_id)
