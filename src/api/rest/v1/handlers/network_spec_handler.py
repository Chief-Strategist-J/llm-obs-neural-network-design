"""REST ingress handler for network_spec creation."""

from typing import Any, Mapping
from src.features.network_spec.service.network_spec_service import NetworkSpecService
from src.features.network_spec.schema.network_spec_schema import NetworkSpecSchema
from src.shared.types.envelope_types import TraceContext, ApiResponse
from src.shared.errors.app_errors import ApplicationError


class NetworkSpecHandler:
    def __init__(self, service: NetworkSpecService) -> None:
        self._service = service

    async def handle_create_spec(
        self, ctx: TraceContext, payload: Mapping[str, Any]
    ) -> tuple[int, ApiResponse[Mapping[str, Any]]]:
        try:
            dto = NetworkSpecSchema.from_api(payload)
            result = await self._service.create_network_spec(ctx, dto)
            if result.is_ok and result.value:
                res_data = NetworkSpecSchema.to_api(result.value)
                return 201, ApiResponse(
                    success=True,
                    data=res_data,
                    meta={"trace_id": ctx.trace_id},
                )
            err = result.error or ApplicationError("Unknown domain error")
            return err.status_code, ApiResponse(
                success=False,
                data=None,
                meta={"trace_id": ctx.trace_id},
                error={"code": err.canonical_code, "message": err.message},
            )
        except ApplicationError as app_err:
            return app_err.status_code, ApiResponse(
                success=False,
                data=None,
                meta={"trace_id": ctx.trace_id},
                error={"code": app_err.canonical_code, "message": app_err.message},
            )
        except Exception as ex:
            return 500, ApiResponse(
                success=False,
                data=None,
                meta={"trace_id": ctx.trace_id},
                error={"code": "INTERNAL_SERVER_ERROR", "message": str(ex)},
            )
