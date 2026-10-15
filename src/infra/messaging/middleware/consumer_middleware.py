import logging
from typing import Callable, Any
from src.infra.messaging.middleware.pipeline import ConsumeCtx, ConsumerMiddleware
from python_shared.http.constants import HTTP_CONSTANTS

logger = logging.getLogger(__name__)


def with_tracing_consumer(ctx: ConsumeCtx, next_step: Callable[[ConsumeCtx], Any]) -> Any:
    traceparent = ctx.headers.get(HTTP_CONSTANTS.HEADER_TRACEPARENT) or ctx.headers.get("traceparent")
    if traceparent:
        parts = traceparent.split("-")
        trace_id = parts[1] if len(parts) > 1 else traceparent
        ctx.metadata["trace_id"] = trace_id
    correlation_id = ctx.headers.get(HTTP_CONSTANTS.HEADER_X_CORRELATION_ID) or ctx.headers.get("x-correlation-id")
    if correlation_id:
        ctx.correlation_id = correlation_id
    return next_step(ctx)


def with_dlq_on_failure(dlq_callback: Callable[[ConsumeCtx, Exception], None] | None = None) -> ConsumerMiddleware:
    def _middleware(ctx: ConsumeCtx, next_step: Callable[[ConsumeCtx], Any]) -> Any:
        try:
            return next_step(ctx)
        except Exception as e:
            logger.error(f"Consumer failure on topic {ctx.topic}, partition {ctx.partition}, offset {ctx.offset}: {e}")
            if dlq_callback:
                dlq_callback(ctx, e)
            return None
    return _middleware
