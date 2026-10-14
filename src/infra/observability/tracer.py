"""Centralized OpenTelemetry tracing engine and span context manager."""

from contextlib import contextmanager
from typing import Generator
from src.shared.types.envelope_types import TraceContext


class Tracer:
    @classmethod
    @contextmanager
    def start_span(cls, name: str, ctx: TraceContext | None = None) -> Generator[TraceContext, None, None]:
        trace_id = ctx.trace_id if ctx and ctx.trace_id else "trace-synthetic-001"
        active_ctx = TraceContext(trace_id=trace_id, span_id=f"span-{name}")
        try:
            yield active_ctx
        finally:
            pass
