from contextlib import contextmanager
from typing import Generator
from src.shared.types.envelope_types import TraceContext

try:
    from python_shared.telemetry.tracer import get_tracer
except ImportError:
    get_tracer = None


class Tracer:
    @classmethod
    @contextmanager
    def start_span(
        cls, name: str, ctx: TraceContext | None = None
    ) -> Generator[TraceContext, None, None]:
        trace_id = ctx.trace_id if ctx and ctx.trace_id else "trace-synthetic-001"
        active_ctx = TraceContext(trace_id=trace_id, span_id=f"span-{name}")

        otel_tracer = get_tracer("neural-network-design") if get_tracer else None
        if otel_tracer:
            with otel_tracer.start_as_current_span(name):
                yield active_ctx
        else:
            yield active_ctx
