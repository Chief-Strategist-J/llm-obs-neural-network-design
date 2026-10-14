"""Centralized exception-to-envelope middleware for FastAPI applications."""

from __future__ import annotations
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from src.shared.errors.app_errors import ApplicationError
from src.shared.constants.system_constants import (
    HTTP_HEADER_REQUEST_ID,
    HTTP_HEADER_TRACEPARENT,
    CODE_INTERNAL_ERROR,
)
from src.shared.utils.date_utils import generate_uuidv7_str, get_utc_iso_timestamp


class ErrorEnvelopeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(HTTP_HEADER_REQUEST_ID) or generate_uuidv7_str()
        traceparent = request.headers.get(HTTP_HEADER_TRACEPARENT) or f"00-{generate_uuidv7_str().replace('-', '')[:32]}-0000000000000001-01"

        try:
            response = await call_next(request)
            response.headers[HTTP_HEADER_REQUEST_ID] = request_id
            response.headers[HTTP_HEADER_TRACEPARENT] = traceparent
            return response
        except ApplicationError as err:
            return JSONResponse(
                status_code=err.status_code,
                content={
                    "success": False,
                    "data": None,
                    "error": {
                        "message": err.message,
                        "canonical_code": err.canonical_code,
                        "retryable": err.retryable,
                        "details": err.details,
                    },
                    "meta": {
                        "timestamp": get_utc_iso_timestamp(),
                        "request_id": request_id,
                        "trace_id": traceparent,
                    },
                },
                headers={
                    HTTP_HEADER_REQUEST_ID: request_id,
                    HTTP_HEADER_TRACEPARENT: traceparent,
                },
            )
        except Exception as ex:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "data": None,
                    "error": {
                        "message": str(ex),
                        "canonical_code": CODE_INTERNAL_ERROR,
                        "retryable": False,
                        "details": {},
                    },
                    "meta": {
                        "timestamp": get_utc_iso_timestamp(),
                        "request_id": request_id,
                        "trace_id": traceparent,
                    },
                },
                headers={
                    HTTP_HEADER_REQUEST_ID: request_id,
                    HTTP_HEADER_TRACEPARENT: traceparent,
                },
            )
