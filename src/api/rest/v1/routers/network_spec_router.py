"""REST router configuring routes and extracting trace context."""

from typing import Any, Mapping
from src.api.rest.v1.handlers.network_spec_handler import NetworkSpecHandler
from src.shared.types.envelope_types import TraceContext
from src.infra.config.config_loader import AppConfig


class NetworkSpecRouter:
    def __init__(self, handler: NetworkSpecHandler) -> None:
        self._handler = handler
        self.endpoint = AppConfig.get_endpoint("network_spec", "/v1/network-spec")

    async def dispatch_post(self, headers: Mapping[str, str], payload: Mapping[str, Any]):
        trace_id = headers.get("traceparent", "trace-synthetic-001")
        ctx = TraceContext(trace_id=trace_id)
        return await self._handler.handle_create_spec(ctx, payload)
