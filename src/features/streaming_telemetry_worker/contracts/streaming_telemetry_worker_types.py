from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Sequence, Any


@dataclass(frozen=True)
class TelemetrySpanMessageDTO:
    span_id: str
    trace_id: str
    timestamp_epoch_ms: int
    features: Sequence[float]
    target_label: int = 0
    service_name: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MicroBatchDTO:
    batch_id: str
    created_at_iso: str
    records: Sequence[TelemetrySpanMessageDTO]
    trace_count: int
    total_spans: int


@dataclass(frozen=True)
class DeadLetterMessageDTO:
    dlq_id: str
    original_topic: str
    partition: int
    offset: str
    raw_payload_utf8: str
    error_reason: str
    failed_at_iso: str
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerMetricsDTO:
    total_consumed: int
    total_batched: int
    total_dlq_routed: int
    total_batches_dispatched: int
    current_buffer_size: int
    status: str
