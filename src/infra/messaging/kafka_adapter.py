"""Infrastructure Kafka Consumer and Producer Adapter for Streaming Telemetry.

Execution Workflow:
1. Implements KafkaProducerPort and KafkaConsumerPort using confluent_kafka or in-memory fallback.
2. Ingests high-throughput telemetry streams partitioned by trace_id to preserve causal ordering.
3. Buffers and commits offsets asynchronously without blocking PyTorch model inference or training.
"""

from __future__ import annotations
import logging
from typing import Any, Mapping
from src.shared.ports.kafka_port import KafkaProducerPort, KafkaConsumerPort

logger = logging.getLogger(__name__)

try:
    from confluent_kafka import Producer as ConfluentProducer, Consumer as ConfluentConsumer
except ImportError:
    ConfluentProducer = None
    ConfluentConsumer = None


class KafkaProducerAdapter(KafkaProducerPort):
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        producer_instance: Any = None,
    ) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._producer = (
            producer_instance
            if producer_instance is not None
            else (ConfluentProducer({"bootstrap.servers": bootstrap_servers}) if ConfluentProducer else None)
        )
        self._in_memory_log: list[dict[str, Any]] = []

    def produce(
        self,
        topic: str,
        key: str,
        value: bytes,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if self._producer:
            self._producer.produce(
                topic=topic,
                key=key.encode("utf-8") if isinstance(key, str) else key,
                value=value,
                headers=[(k, v.encode("utf-8")) for k, v in (headers or {}).items()],
            )
            self._producer.poll(0)
        else:
            self._in_memory_log.append({"topic": topic, "key": key, "value": value})

    def flush(self, timeout: float = 10.0) -> None:
        if self._producer:
            self._producer.flush(timeout)


class KafkaConsumerAdapter(KafkaConsumerPort):
    def __init__(
        self,
        group_id: str = "neural-network-telemetry-group",
        bootstrap_servers: str = "localhost:9092",
        consumer_instance: Any = None,
    ) -> None:
        self._group_id = group_id
        self._bootstrap_servers = bootstrap_servers
        self._consumer = (
            consumer_instance
            if consumer_instance is not None
            else (
                ConfluentConsumer(
                    {
                        "bootstrap.servers": bootstrap_servers,
                        "group.id": group_id,
                        "auto.offset.reset": "earliest",
                        "enable.auto.commit": False,
                    }
                )
                if ConfluentConsumer
                else None
            )
        )
        self._subscribed_topics: list[str] = []

    def subscribe(self, topics: list[str]) -> None:
        self._subscribed_topics = list(topics)
        if self._consumer:
            self._consumer.subscribe(topics)

    def poll(self, timeout: float = 1.0) -> object | None:
        if self._consumer:
            return self._consumer.poll(timeout)
        return None

    def commit(self, asynchronous: bool = True) -> None:
        if self._consumer:
            self._consumer.commit(asynchronous=asynchronous)

    def close(self) -> None:
        if self._consumer:
            self._consumer.close()
