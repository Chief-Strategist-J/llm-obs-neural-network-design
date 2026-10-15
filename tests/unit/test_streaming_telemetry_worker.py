import pytest
import json
from src.features.streaming_telemetry_worker.rules.streaming_telemetry_worker_rules import (
    parse_telemetry_span,
    should_flush_buffer,
    build_micro_batch,
    build_dead_letter_message,
)
from src.features.streaming_telemetry_worker.repository.streaming_telemetry_worker_repository import (
    InMemoryStreamingTelemetryWorkerRepository,
)
from src.features.streaming_telemetry_worker.service.streaming_telemetry_worker_service import (
    StreamingTelemetryWorkerService,
)
from src.infra.messaging.kafka_adapter import KafkaProducerAdapter
from src.shared.types.envelope_types import TraceContext
from src.shared.constants.system_constants import TOPIC_TELEMETRY_SPANS_RAW, TOPIC_TELEMETRY_SPANS_DLQ


def test_streaming_worker_parse_telemetry_span_success():
    valid_payload = json.dumps({
        "span_id": "span-123",
        "trace_id": "trace-456",
        "timestamp_epoch_ms": 1729000000000,
        "features": [0.1, 0.2, 0.3, 0.4],
        "target_label": 1,
        "service_name": "gateway-service",
    }).encode("utf-8")

    res = parse_telemetry_span(valid_payload)
    assert res.is_ok is True
    assert res.value is not None
    span = res.value
    assert span.span_id == "span-123"
    assert span.trace_id == "trace-456"
    assert span.features == (0.1, 0.2, 0.3, 0.4)
    assert span.target_label == 1
    assert span.service_name == "gateway-service"


def test_streaming_worker_parse_telemetry_span_invalid_payload():
    corrupted_payload = b"not-a-json-payload-invalid-binary"
    res = parse_telemetry_span(corrupted_payload)
    assert res.is_ok is False
    assert res.error is not None
    assert "Invalid telemetry span payload" in res.error.message

    missing_fields_payload = json.dumps({"features": [1.0]}).encode("utf-8")
    res_missing = parse_telemetry_span(missing_fields_payload)
    assert res_missing.is_ok is False
    assert "Missing required span_id field" in res_missing.error.message


def test_streaming_worker_buffer_flushing_rules():
    assert should_flush_buffer(current_size=0, max_batch_size=10, elapsed_ms=5000.0, max_timeout_ms=1000.0) is False
    assert should_flush_buffer(current_size=10, max_batch_size=10, elapsed_ms=100.0, max_timeout_ms=1000.0) is True
    assert should_flush_buffer(current_size=5, max_batch_size=10, elapsed_ms=1200.0, max_timeout_ms=1000.0) is True
    assert should_flush_buffer(current_size=5, max_batch_size=10, elapsed_ms=800.0, max_timeout_ms=1000.0) is False


@pytest.mark.asyncio
async def test_streaming_worker_service_valid_flow_and_micro_batching():
    repo = InMemoryStreamingTelemetryWorkerRepository()
    service = StreamingTelemetryWorkerService(
        repository=repo,
        batch_size=2,
        batch_timeout_ms=5000.0,
    )
    ctx = TraceContext(trace_id="test-trace-001")

    p1 = json.dumps({"span_id": "s1", "trace_id": "t1", "features": [1.0, 2.0]}).encode("utf-8")
    p2 = json.dumps({"span_id": "s2", "trace_id": "t1", "features": [2.0, 3.0]}).encode("utf-8")

    res1 = await service.process_raw_message(ctx, TOPIC_TELEMETRY_SPANS_RAW, 0, "0", p1)
    assert res1.is_ok is True

    res2 = await service.process_raw_message(ctx, TOPIC_TELEMETRY_SPANS_RAW, 0, "1", p2)
    assert res2.is_ok is True

    flush_res = await service.flush_buffer(ctx, force=False)
    assert flush_res.is_ok is True
    batch = flush_res.value
    assert batch is not None
    assert batch.total_spans == 2
    assert batch.trace_count == 1

    batches_res = await repo.list_micro_batches()
    assert batches_res.is_ok is True
    assert len(batches_res.value) == 1


@pytest.mark.asyncio
async def test_streaming_worker_service_dlq_routing_on_corruption():
    repo = InMemoryStreamingTelemetryWorkerRepository()
    producer = KafkaProducerAdapter(producer_instance=None)
    producer._producer = None
    service = StreamingTelemetryWorkerService(
        repository=repo,
        producer=producer,
        batch_size=10,
        batch_timeout_ms=5000.0,
    )
    ctx = TraceContext(trace_id="test-trace-dlq")

    corrupted = b"corrupted-span-payload-without-json"
    res = await service.process_raw_message(ctx, TOPIC_TELEMETRY_SPANS_RAW, 1, "42", corrupted)
    assert res.is_ok is True
    dlq_record = res.value
    assert dlq_record.error_reason != ""
    assert dlq_record.partition == 1
    assert dlq_record.offset == "42"

    dlq_list_res = await repo.list_dead_letters()
    assert dlq_list_res.is_ok is True
    assert len(dlq_list_res.value) == 1
    assert dlq_list_res.value[0].raw_payload_utf8 == "corrupted-span-payload-without-json"

    assert len(producer._in_memory_log) == 1
    assert producer._in_memory_log[0]["topic"] == TOPIC_TELEMETRY_SPANS_DLQ
