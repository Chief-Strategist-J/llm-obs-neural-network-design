import pytest
from unittest.mock import MagicMock
from src.infra.messaging.kafka_adapter import KafkaProducerAdapter, KafkaConsumerAdapter
from src.shared.constants.system_constants import (
    TOPIC_TELEMETRY_SPANS_RAW,
    TOPIC_MODEL_RETRAIN_COMMANDS,
)


def test_kafka_producer_confluent_calls():
    mock_confluent_prod = MagicMock()
    producer = KafkaProducerAdapter(
        bootstrap_servers="localhost:9092",
        producer_instance=mock_confluent_prod,
    )

    test_key = "trace-4bf92f3577b34da6a3ce929d0e0e4736"
    test_val = b'{"record_id": "r1", "features": [1.0, 0.0]}'
    headers = {"x-request-id": "req-123"}

    producer.produce(
        topic=TOPIC_TELEMETRY_SPANS_RAW,
        key=test_key,
        value=test_val,
        headers=headers,
    )
    producer.flush(timeout=5.0)

    mock_confluent_prod.produce.assert_called_once()
    mock_confluent_prod.flush.assert_called_once_with(5.0)


def test_kafka_consumer_confluent_calls():
    mock_confluent_cons = MagicMock()
    mock_msg = MagicMock()
    mock_msg.value.return_value = b'{"record_id": "r1"}'
    mock_confluent_cons.poll.return_value = mock_msg

    consumer = KafkaConsumerAdapter(
        group_id="test-group",
        bootstrap_servers="localhost:9092",
        consumer_instance=mock_confluent_cons,
    )

    consumer.subscribe([TOPIC_TELEMETRY_SPANS_RAW])
    mock_confluent_cons.subscribe.assert_called_once_with([TOPIC_TELEMETRY_SPANS_RAW])

    msg = consumer.poll(timeout=1.5)
    assert msg == mock_msg
    mock_confluent_cons.poll.assert_called_once_with(1.5)

    consumer.commit(asynchronous=True)
    mock_confluent_cons.commit.assert_called_once_with(asynchronous=True)

    consumer.close()
    mock_confluent_cons.close.assert_called_once()


def test_kafka_producer_in_memory_fallback():
    producer = KafkaProducerAdapter(
        bootstrap_servers="localhost:9092",
        producer_instance=None,
    )
    producer._producer = None

    producer.produce(
        topic=TOPIC_MODEL_RETRAIN_COMMANDS,
        key="trace-fallback-001",
        value=b'{"retrain": true}',
    )
    assert len(producer._in_memory_log) == 1
    assert producer._in_memory_log[0]["topic"] == TOPIC_MODEL_RETRAIN_COMMANDS
    assert producer._in_memory_log[0]["key"] == "trace-fallback-001"
