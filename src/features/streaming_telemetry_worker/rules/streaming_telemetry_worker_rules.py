from __future__ import annotations
import json
from typing import Mapping, Sequence, Any
from src.features.streaming_telemetry_worker.types.streaming_telemetry_worker_types import (
    TelemetrySpanMessageDTO,
    MicroBatchDTO,
    DeadLetterMessageDTO,
)
from src.shared.types.envelope_types import Result
from src.shared.errors.app_errors import ValidationError


def parse_telemetry_span(payload: bytes | str) -> Result[TelemetrySpanMessageDTO, ValidationError]:
    try:
        raw_str = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        data = json.loads(raw_str)
        if not isinstance(data, dict):
            return Result.err(ValidationError("Telemetry payload must be a JSON object"))
        
        span_id = str(data.get("span_id", "")).strip()
        trace_id = str(data.get("trace_id", "")).strip()
        if not span_id:
            return Result.err(ValidationError("Missing required span_id field"))
        if not trace_id:
            return Result.err(ValidationError("Missing required trace_id field"))

        raw_features = data.get("features", [])
        if not isinstance(raw_features, (list, tuple)):
            return Result.err(ValidationError("Features field must be a list of numeric values"))
        
        features_tuple = tuple(float(x) for x in raw_features)
        timestamp_epoch_ms = int(data.get("timestamp_epoch_ms", 0))
        target_label = int(data.get("target_label", 0))
        service_name = str(data.get("service_name", ""))
        attributes = dict(data.get("attributes", {}))

        dto = TelemetrySpanMessageDTO(
            span_id=span_id,
            trace_id=trace_id,
            timestamp_epoch_ms=timestamp_epoch_ms,
            features=features_tuple,
            target_label=target_label,
            service_name=service_name,
            attributes=attributes,
        )
        return Result.ok(dto)
    except Exception as exc:
        return Result.err(ValidationError(f"Invalid telemetry span payload: {str(exc)}"))


def should_flush_buffer(
    current_size: int,
    max_batch_size: int,
    elapsed_ms: float,
    max_timeout_ms: float,
) -> bool:
    if current_size <= 0:
        return False
    if current_size >= max_batch_size:
        return True
    if elapsed_ms >= max_timeout_ms:
        return True
    return False


def build_micro_batch(
    batch_id: str,
    created_at_iso: str,
    spans: Sequence[TelemetrySpanMessageDTO],
) -> MicroBatchDTO:
    unique_traces = len({s.trace_id for s in spans})
    return MicroBatchDTO(
        batch_id=batch_id,
        created_at_iso=created_at_iso,
        records=tuple(spans),
        trace_count=unique_traces,
        total_spans=len(spans),
    )


def build_dead_letter_message(
    dlq_id: str,
    original_topic: str,
    partition: int,
    offset: str,
    raw_payload: bytes | str,
    error_reason: str,
    failed_at_iso: str,
    headers: Mapping[str, str] | None = None,
) -> DeadLetterMessageDTO:
    raw_str = raw_payload.decode("utf-8", errors="replace") if isinstance(raw_payload, bytes) else str(raw_payload)
    return DeadLetterMessageDTO(
        dlq_id=dlq_id,
        original_topic=original_topic,
        partition=partition,
        offset=offset,
        raw_payload_utf8=raw_str,
        error_reason=error_reason,
        failed_at_iso=failed_at_iso,
        headers=dict(headers or {}),
    )
