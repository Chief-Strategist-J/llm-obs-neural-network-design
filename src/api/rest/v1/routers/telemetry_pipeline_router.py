from typing import Any, Mapping
from fastapi import APIRouter, Request, Response, status
from src.features.system_telemetry_pipeline.service.telemetry_pipeline_service import TelemetryPipelineService
from src.features.system_telemetry_pipeline.repository.telemetry_pipeline_repository import InMemoryTelemetryPipelineRepository
from src.features.system_telemetry_pipeline.schema.telemetry_pipeline_schema import TelemetryPipelineSchema
from src.shared.types.envelope_types import TraceContext
from src.shared.errors.app_errors import ApplicationError

router = APIRouter(prefix="/v1/telemetry-pipeline", tags=["telemetry-pipeline"])

_pipeline_repo = InMemoryTelemetryPipelineRepository()
_pipeline_service = TelemetryPipelineService(repository=_pipeline_repo)


@router.post("", status_code=status.HTTP_201_CREATED)
async def initialize_pipeline(request: Request, response: Response):
    trace_id = request.headers.get("traceparent", "trace-synthetic-001")
    ctx = TraceContext(trace_id=trace_id)
    try:
        payload = await request.json()
        dto = TelemetryPipelineSchema.from_api(payload)
        res = await _pipeline_service.initialize_pipeline_and_baseline(ctx, dto)
        if res.is_ok and res.value:
            data = TelemetryPipelineSchema.to_api(res.value)
            response.status_code = status.HTTP_201_CREATED
            return {"success": True, "data": data, "meta": {"trace_id": trace_id}, "error": None}
        err = res.error or ApplicationError("Failed to initialize telemetry pipeline")
        response.status_code = err.status_code
        return {"success": False, "data": None, "meta": {"trace_id": trace_id}, "error": {"code": err.canonical_code, "message": err.message}}
    except ApplicationError as app_err:
        response.status_code = app_err.status_code
        return {"success": False, "data": None, "meta": {"trace_id": trace_id}, "error": {"code": app_err.canonical_code, "message": app_err.message}}
    except Exception as ex:
        response.status_code = 500
        return {"success": False, "data": None, "meta": {"trace_id": trace_id}, "error": {"code": "INTERNAL_SERVER_ERROR", "message": str(ex)}}
