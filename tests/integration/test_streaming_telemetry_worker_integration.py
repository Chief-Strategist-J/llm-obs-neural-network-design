import pytest
import json
from src.features.streaming_telemetry_worker.repository.streaming_telemetry_worker_repository import (
    InMemoryStreamingTelemetryWorkerRepository,
)
from src.features.streaming_telemetry_worker.service.streaming_telemetry_worker_service import (
    StreamingTelemetryWorkerService,
)
from src.features.streaming_baselines.repository.streaming_baseline_repository import (
    InMemoryStreamingBaselineRepository,
)
from src.features.streaming_baselines.service.streaming_baseline_service import (
    StreamingBaselineService,
)
from src.infra.messaging.kafka_adapter import KafkaProducerAdapter, KafkaConsumerAdapter
from src.shared.types.envelope_types import TraceContext
from src.shared.constants.system_constants import TOPIC_TELEMETRY_SPANS_RAW


class MockKafkaMessage:
    def __init__(self, topic: str, partition: int, offset: int, value: bytes, headers: list = None):
        self._topic = topic
        self._partition = partition
        self._offset = offset
        self._value = value
        self._headers = headers or []

    def topic(self):
        return self._topic

    def partition(self):
        return self._partition

    def offset(self):
        return self._offset

    def value(self):
        return self._value

    def headers(self):
        return self._headers


class MockKafkaConsumer:
    def __init__(self, messages):
        self._messages = list(messages)
        self.committed = False

    def poll(self, timeout=1.0):
        if self._messages:
            return self._messages.pop(0)
        return None

    def commit(self, asynchronous=True):
        self.committed = True

    def subscribe(self, topics):
        pass

    def close(self):
        pass


@pytest.mark.asyncio
async def test_streaming_worker_and_fast_path_gating_integration():
    mock_spans = [
        MockKafkaMessage(
            topic=TOPIC_TELEMETRY_SPANS_RAW,
            partition=0,
            offset=101,
            value=json.dumps({"span_id": "s101", "trace_id": "t1", "features": [0.05, 0.08]}).encode("utf-8"),
            headers=[("traceparent", b"00-trace101-span01-01")],
        ),
        MockKafkaMessage(
            topic=TOPIC_TELEMETRY_SPANS_RAW,
            partition=0,
            offset=102,
            value=json.dumps({"span_id": "s102", "trace_id": "t1", "features": [0.06, 0.09]}).encode("utf-8"),
            headers=[("traceparent", b"00-trace101-span02-01")],
        ),
    ]

    mock_cons = MockKafkaConsumer(mock_spans)
    consumer_adapter = KafkaConsumerAdapter(consumer_instance=mock_cons)
    worker_repo = InMemoryStreamingTelemetryWorkerRepository()
    worker_service = StreamingTelemetryWorkerService(
        repository=worker_repo,
        consumer=consumer_adapter,
        batch_size=2,
        batch_timeout_ms=100.0,
    )

    ctx = TraceContext(trace_id="test-integration-worker-ctx")
    poll_res = await worker_service.poll_and_dispatch_cycle(ctx, max_records=5)
    assert poll_res.is_ok is True
    assert poll_res.value == 2
    assert mock_cons.committed is True

    batches_res = await worker_repo.list_micro_batches()
    assert batches_res.is_ok is True
    assert len(batches_res.value) == 1
    batch = batches_res.value[0]
    assert batch.total_spans == 2

    baseline_repo = InMemoryStreamingBaselineRepository()
    baseline_service = StreamingBaselineService(repository=baseline_repo)
    for span in batch.records:
        triage_res = await baseline_service.triage_span_metric(ctx, "latency_ms", span.features[0])
        assert triage_res.is_ok is True
        assert triage_res.value.decision in ("benign", "anomalous", "ambiguous")
