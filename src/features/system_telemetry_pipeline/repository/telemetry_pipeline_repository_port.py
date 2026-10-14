from abc import ABC, abstractmethod
from src.features.system_telemetry_pipeline.types.telemetry_pipeline_types import TelemetryPipelineModel
from src.shared.types.envelope_types import TraceContext


class TelemetryPipelineRepositoryPort(ABC):
    @abstractmethod
    async def save(self, ctx: TraceContext, model: TelemetryPipelineModel) -> TelemetryPipelineModel:
        pass

    @abstractmethod
    async def find_by_id(self, ctx: TraceContext, pipeline_id: str) -> TelemetryPipelineModel | None:
        pass
