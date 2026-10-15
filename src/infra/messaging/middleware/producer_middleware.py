import logging
from typing import Callable, Any
from src.infra.messaging.middleware.pipeline import ProduceCtx, ProducerMiddleware
from python_shared.http.constants import HTTP_CONSTANTS

logger = logging.getLogger(__name__)


def with_tracing_producer(ctx: ProduceCtx, next_step: Callable[[ProduceCtx], None]) -> None:
    if HTTP_CONSTANTS.HEADER_TRACEPARENT not in ctx.headers and "traceparent" not in ctx.headers:
        trace_id = ctx.metadata.get("trace_id", "trace-synthetic-001")
        ctx.headers[HTTP_CONSTANTS.HEADER_TRACEPARENT] = f"00-{trace_id}-0000000000000001-01"
    if ctx.correlation_id:
        ctx.headers[HTTP_CONSTANTS.HEADER_X_CORRELATION_ID] = ctx.correlation_id
    next_step(ctx)


def with_retry_producer(max_retries: int = 3) -> ProducerMiddleware:
    def _middleware(ctx: ProduceCtx, next_step: Callable[[ProduceCtx], None]) -> None:
        attempts = 0
        while attempts <= max_retries:
            try:
                next_step(ctx)
                return
            except Exception as e:
                attempts += 1
                if attempts > max_retries:
                    raise e
    return _middleware
