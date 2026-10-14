"""W3C Distributed Trace Context and Request ID extraction middleware.

Algorithmic Workflow:
1. Extracts incoming W3C traceparent and correlation identifiers or generates deterministic synthetic root spans.
2. Injects trace and request headers into the Starlette/FastAPI request state and outbound response headers.
3. Coordinates end-to-end distributed tracing across all downstream service handlers and background jobs.
"""

from __future__ import annotations
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from python_shared.http.constants import HTTP_CONSTANTS
from src.shared.types.envelope_types import TraceContext
from src.shared.utils.date_utils import generate_uuidv7_str


class TraceContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        traceparent = (
            request.headers.get(HTTP_CONSTANTS.HEADER_TRACEPARENT)
            or f"00-{generate_uuidv7_str().replace('-', '')[:32]}-0000000000000001-01"
        )
        request_id = (
            request.headers.get(HTTP_CONSTANTS.HEADER_X_REQUEST_ID)
            or generate_uuidv7_str()
        )
        correlation_id = (
            request.headers.get(HTTP_CONSTANTS.HEADER_X_CORRELATION_ID)
            or request_id
        )

        trace_id = traceparent.split("-")[1] if len(traceparent.split("-")) > 1 else traceparent
        request.state.trace_context = TraceContext(trace_id=trace_id, span_id=request_id)
        request.state.correlation_id = correlation_id

        response = await call_next(request)
        response.headers[HTTP_CONSTANTS.HEADER_TRACEPARENT] = traceparent
        response.headers[HTTP_CONSTANTS.HEADER_X_REQUEST_ID] = request_id
        response.headers[HTTP_CONSTANTS.HEADER_X_CORRELATION_ID] = correlation_id
        return response
