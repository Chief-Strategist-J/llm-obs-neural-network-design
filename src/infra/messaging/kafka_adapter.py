"""Infrastructure Kafka Consumer and Producer Adapter for Streaming Telemetry.

Execution Workflow:
1. Implements KafkaProducerPort and KafkaConsumerPort using confluent_kafka or in-memory fallback.
2. Ingests high-throughput telemetry streams partitioned by trace_id to preserve causal ordering.
3. Buffers and commits offsets asynchronously without blocking PyTorch model inference or training.
"""

from __future__ import annotations
import logging
from typing import Any, Mapping
from src.infra.messaging.ports.kafka_port import KafkaProducerPort, KafkaConsumerPort

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
        from src.infra.messaging.middleware import (
            ProducerMiddlewarePipeline,
            ProduceCtx,
            with_tracing_producer,
            with_retry_producer,
        )
        self._pipeline = ProducerMiddlewarePipeline(
            middlewares=[with_tracing_producer, with_retry_producer(max_retries=2)]
        )

    def produce(
        self,
        topic: str,
        key: str,
        value: bytes,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        from src.infra.messaging.middleware import ProduceCtx

        ctx = ProduceCtx(
            topic=topic,
            key=key,
            payload=value,
            headers=dict(headers or {}),
        )

        def _raw_produce(p_ctx: ProduceCtx) -> None:
            if self._producer:
                self._producer.produce(
                    topic=p_ctx.topic,
                    key=p_ctx.key.encode("utf-8") if isinstance(p_ctx.key, str) else p_ctx.key,
                    value=p_ctx.payload,
                    headers=[(k, v.encode("utf-8") if isinstance(v, str) else v) for k, v in p_ctx.headers.items()],
                )
                self._producer.poll(0)
            else:
                self._in_memory_log.append(
                    {"topic": p_ctx.topic, "key": p_ctx.key, "value": p_ctx.payload, "headers": p_ctx.headers}
                )

        self._pipeline.execute(ctx, _raw_produce)

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

    def poll_with_pipeline(
        self, timeout: float = 1.0, handler: Any = None
    ) -> Any:
        from src.infra.messaging.middleware import (
            ConsumerMiddlewarePipeline,
            ConsumeCtx,
            with_tracing_consumer,
            with_dlq_on_failure,
        )

        pipeline = ConsumerMiddlewarePipeline(
            middlewares=[with_tracing_consumer, with_dlq_on_failure()]
        )

        msg = self.poll(timeout)
        if not msg:
            return None

        headers_dict = {}
        if hasattr(msg, "headers") and msg.headers():
            for k, v in msg.headers():
                key_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
                val_str = v.decode("utf-8") if isinstance(v, bytes) else str(v)
                headers_dict[key_str] = val_str

        topic_str = msg.topic() if hasattr(msg, "topic") else ""
        partition_num = msg.partition() if hasattr(msg, "partition") else 0
        offset_val = str(msg.offset()) if hasattr(msg, "offset") else "0"
        value_bytes = msg.value() if hasattr(msg, "value") else None

        ctx = ConsumeCtx(
            topic=topic_str,
            partition=partition_num,
            offset=offset_val,
            raw_message=msg,
            payload=value_bytes,
            headers=headers_dict,
        )

        def _execute_handler(c_ctx: ConsumeCtx) -> Any:
            if handler:
                return handler(c_ctx)
            return c_ctx.payload

        return pipeline.execute(ctx, _execute_handler)

    def commit(self, asynchronous: bool = True) -> None:
        if self._consumer:
            self._consumer.commit(asynchronous=asynchronous)

    def close(self) -> None:
        if self._consumer:
            self._consumer.close()

