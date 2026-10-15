from typing import Any, Mapping
from fastapi import APIRouter, Request, Response, status
from src.features.system_telemetry_pipeline.service.telemetry_pipeline_service import TelemetryPipelineService
from src.features.system_telemetry_pipeline.repository.telemetry_pipeline_repository import InMemoryTelemetryPipelineRepository
from src.features.system_telemetry_pipeline.schema.telemetry_pipeline_schema import TelemetryPipelineSchema
from src.shared.types.envelope_types import TraceContext
from src.infra.config.config_loader import AppConfig

router = APIRouter(
    prefix=AppConfig.get_endpoint("telemetry_pipeline", "/v1/telemetry-pipeline"),
    tags=["telemetry-pipeline"],
)

_pipeline_repo = InMemoryTelemetryPipelineRepository()
_pipeline_service = TelemetryPipelineService(repository=_pipeline_repo)


from src.shared.errors.app_errors import ApplicationError

@router.post("", status_code=status.HTTP_201_CREATED)
async def initialize_pipeline(request: Request, response: Response):
    ctx = getattr(request.state, "trace_context", None) or TraceContext(
        trace_id=request.headers.get("traceparent", "trace-synthetic-001")
    )
    payload = await request.json()
    dto = TelemetryPipelineSchema.from_api(payload)
    res = await _pipeline_service.initialize_pipeline_and_baseline(ctx, dto)
    if not res.is_ok or not res.value:
        raise res.error or ApplicationError("Failed to initialize telemetry pipeline")
    data = TelemetryPipelineSchema.to_api(res.value)
    return {"success": True, "data": data, "meta": {"trace_id": ctx.trace_id}, "error": None}
