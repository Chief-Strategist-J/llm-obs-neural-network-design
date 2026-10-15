import pytest
from unittest.mock import MagicMock
from src.infra.messaging.kafka_adapter import (
    KafkaProducerAdapter,
    KafkaConsumerAdapter,
)


def test_kafka_producer_consumer_roundtrip_contract():
    mock_producer_client = MagicMock()
    mock_consumer_client = MagicMock()

    captured_messages = []

    def mock_produce(topic, key, value, headers):
        captured_messages.append({"topic": topic, "key": key, "value": value, "headers": headers})

    mock_producer_client.produce.side_effect = mock_produce
    mock_producer_client.poll.return_value = 0

    producer = KafkaProducerAdapter(producer_instance=mock_producer_client)
    producer.produce(
        topic="telemetry.spans.raw",
        key="trace_abc_123",
        value=b'{"trace_id": "trace_abc_123", "latency_ms": 250.0}',
        headers={"x-correlation-id": "corr-999", "traceparent": "test-kafka-integration-trace-777"},
    )
    producer.flush(1.0)

    assert len(captured_messages) == 1
    produced = captured_messages[0]
    assert produced["topic"] == "telemetry.spans.raw"
    assert produced["key"] == b"trace_abc_123"
    header_keys = [h[0] for h in produced["headers"]]
    assert "traceparent" in header_keys

    mock_msg = MagicMock()
    mock_msg.error.return_value = None
    mock_msg.topic.return_value = "telemetry.spans.raw"
    mock_msg.key.return_value = b"trace_abc_123"
    mock_msg.value.return_value = b'{"trace_id": "trace_abc_123", "latency_ms": 250.0}'
    mock_msg.headers.return_value = [("traceparent", b"test-kafka-integration-trace-777")]

    mock_consumer_client.poll.return_value = mock_msg

    consumer = KafkaConsumerAdapter(consumer_instance=mock_consumer_client)
    consumer.subscribe(["telemetry.spans.raw"])
    received_record = consumer.poll(timeout=1.0)

    assert received_record is not None
    assert received_record.topic() == "telemetry.spans.raw"
    assert received_record.key() == b"trace_abc_123"
