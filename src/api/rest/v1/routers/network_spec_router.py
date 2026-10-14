from typing import Any, Mapping
from fastapi import APIRouter, Request, Response, status
from src.api.rest.v1.handlers.network_spec_handler import NetworkSpecHandler
from src.shared.types.envelope_types import TraceContext
from src.features.network_spec.service.network_spec_service import NetworkSpecService
from src.features.network_spec.repository.network_spec_repository import InMemoryNetworkSpecRepository
from src.features.network_spec.repository.neural_compute_adapter import PyTorchNeuralComputeAdapter

from src.infra.config.config_loader import AppConfig

router = APIRouter(prefix="/v1/network-spec", tags=["network-spec"])

_repo = InMemoryNetworkSpecRepository()
_compute = PyTorchNeuralComputeAdapter()
_service = NetworkSpecService(repository=_repo, compute=_compute)
_handler = NetworkSpecHandler(service=_service)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_network_spec(request: Request, response: Response):
    payload = await request.json()
    ctx = getattr(request.state, "trace_context", None) or TraceContext(
        trace_id=request.headers.get("traceparent", "trace-synthetic-001")
    )
    status_code, result = await _handler.handle_create_spec(ctx, payload)
    response.status_code = status_code
    return {
        "success": result.success,
        "data": result.data,
        "meta": result.meta,
        "error": result.error,
    }


@router.post("/verify-overfit", status_code=status.HTTP_200_OK)
async def verify_overfit(request: Request, response: Response):
    payload = await request.json()
    ctx = getattr(request.state, "trace_context", None) or TraceContext(
        trace_id=request.headers.get("traceparent", "trace-synthetic-001")
    )
    status_code, result = await _handler.handle_verify_overfit(ctx, payload)
    response.status_code = status_code
    return {
        "success": result.success,
        "data": result.data,
        "meta": result.meta,
        "error": result.error,
    }
