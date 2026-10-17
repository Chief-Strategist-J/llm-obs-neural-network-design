from __future__ import annotations
import json
import time
from typing import Sequence, Mapping, Any
from src.features.streaming_telemetry_worker.infrastructure.ports.streaming_telemetry_worker_port import (
    StreamingTelemetryWorkerRepositoryPort,
)
from src.features.streaming_telemetry_worker.contracts.streaming_telemetry_worker_types import (
    TelemetrySpanMessageDTO,
    MicroBatchDTO,
    DeadLetterMessageDTO,
    WorkerMetricsDTO,
)
from src.features.streaming_telemetry_worker.domain.rules.streaming_telemetry_worker_rules import (
    parse_telemetry_span,
    should_flush_buffer,
    build_micro_batch,
    build_dead_letter_message,
)
from src.infra.messaging.ports.kafka_port import KafkaProducerPort, KafkaConsumerPort
from src.infra.observability.tracer import Tracer
from src.shared.types.envelope_types import TraceContext, Result
from src.shared.errors.app_errors import ApplicationError
from src.shared.constants.system_constants import (
    TOPIC_TELEMETRY_SPANS_RAW,
    TOPIC_TELEMETRY_SPANS_DLQ,
    STATUS_ACTIVE,
    STATUS_STOPPED,
)
from src.shared.utils.date_utils import generate_uuidv7_str, get_utc_iso_timestamp


class StreamingTelemetryWorkerService:
    def __init__(
        self,
        repository: StreamingTelemetryWorkerRepositoryPort,
        producer: KafkaProducerPort | None = None,
        consumer: KafkaConsumerPort | None = None,
        batch_size: int = 100,
        batch_timeout_ms: float = 1000.0,
    ) -> None:
        self._repository = repository
        self._producer = producer
        self._consumer = consumer
        self._batch_size = batch_size
        self._batch_timeout_ms = batch_timeout_ms
        self._buffer: list[TelemetrySpanMessageDTO] = []
        self._last_flush_time_ms: float = time.monotonic() * 1000.0
        self._total_consumed: int = 0
        self._total_batched: int = 0
        self._total_dlq_routed: int = 0
        self._total_batches_dispatched: int = 0
        self._is_running: bool = False

    async def process_raw_message(
        self,
        ctx: TraceContext,
        topic: str,
        partition: int,
        offset: str,
        payload: bytes | str,
        headers: Mapping[str, str] | None = None,
    ) -> Result[TelemetrySpanMessageDTO | DeadLetterMessageDTO, ApplicationError]:
        with Tracer.start_span("streaming_telemetry_worker.process_message", ctx=ctx) as child_ctx:
            self._total_consumed += 1
        parse_res = parse_telemetry_span(payload)
        if parse_res.is_ok and parse_res.value:
            span_dto = parse_res.value
            self._buffer.append(span_dto)
            self._total_batched += 1
            await self._sync_metrics()
            return Result.ok(span_dto)

        err_reason = parse_res.error.message if parse_res.error else "Payload validation failure"
        dlq = build_dead_letter_message(
            dlq_id=generate_uuidv7_str(),
            original_topic=topic,
            partition=partition,
            offset=offset,
            raw_payload=payload,
            error_reason=err_reason,
            failed_at_iso=get_utc_iso_timestamp(),
            headers=headers,
        )
        self._total_dlq_routed += 1
        await self._repository.save_dead_letter(dlq)
        if self._producer:
            dlq_payload = json.dumps({
                "dlq_id": dlq.dlq_id,
                "original_topic": dlq.original_topic,
                "partition": dlq.partition,
                "offset": dlq.offset,
                "raw_payload_utf8": dlq.raw_payload_utf8,
                "error_reason": dlq.error_reason,
                "failed_at_iso": dlq.failed_at_iso,
            }).encode("utf-8")
            self._producer.produce(
                topic=TOPIC_TELEMETRY_SPANS_DLQ,
                key=dlq.dlq_id,
                value=dlq_payload,
                headers=headers,
            )
        await self._sync_metrics()
        return Result.ok(dlq)

    async def flush_buffer(
        self, ctx: TraceContext, force: bool = False
    ) -> Result[MicroBatchDTO | None, ApplicationError]:
        with Tracer.start_span("streaming_telemetry_worker.flush_buffer", ctx=ctx) as child_ctx:
            current_time_ms = time.monotonic() * 1000.0
            elapsed_ms = current_time_ms - self._last_flush_time_ms

            if not force and not should_flush_buffer(
                current_size=len(self._buffer),
                max_batch_size=self._batch_size,
                elapsed_ms=elapsed_ms,
                max_timeout_ms=self._batch_timeout_ms,
            ):
                return Result.ok(None)

            if not self._buffer:
                return Result.ok(None)

            spans_to_batch = list(self._buffer)
            self._buffer.clear()
            self._last_flush_time_ms = current_time_ms

            batch = build_micro_batch(
                batch_id=generate_uuidv7_str(),
                created_at_iso=get_utc_iso_timestamp(),
                spans=spans_to_batch,
            )
            await self._repository.save_micro_batch(batch)
            self._total_batches_dispatched += 1
            await self._sync_metrics()
            return Result.ok(batch)

    async def poll_and_dispatch_cycle(
        self, ctx: TraceContext, max_records: int = 10
    ) -> Result[int, ApplicationError]:
        with Tracer.start_span("streaming_telemetry_worker.poll_and_dispatch", ctx=ctx) as child_ctx:
            if not self._consumer:
                return Result.ok(0)

        consumed_count = 0
        for _ in range(max_records):
            msg = self._consumer.poll(0.05)
            if not msg:
                break

            topic = getattr(msg, "topic", lambda: TOPIC_TELEMETRY_SPANS_RAW)()
            partition = getattr(msg, "partition", lambda: 0)()
            offset = str(getattr(msg, "offset", lambda: 0)())
            val = getattr(msg, "value", lambda: None)()
            raw_headers = getattr(msg, "headers", lambda: None)()
            headers_dict: dict[str, str] = {}
            if raw_headers:
                for k, v in raw_headers:
                    key_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
                    val_str = v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    headers_dict[key_str] = val_str

            if val is not None:
                await self.process_raw_message(
                    ctx=ctx,
                    topic=topic,
                    partition=partition,
                    offset=offset,
                    payload=val,
                    headers=headers_dict,
                )
                consumed_count += 1

        await self.flush_buffer(ctx, force=False)
        if consumed_count > 0:
            self._consumer.commit(asynchronous=True)
        return Result.ok(consumed_count)

    async def get_worker_metrics(
        self, ctx: TraceContext
    ) -> Result[WorkerMetricsDTO, ApplicationError]:
        return await self._repository.get_metrics()

    async def _sync_metrics(self) -> None:
        metrics = WorkerMetricsDTO(
            total_consumed=self._total_consumed,
            total_batched=self._total_batched,
            total_dlq_routed=self._total_dlq_routed,
            total_batches_dispatched=self._total_batches_dispatched,
            current_buffer_size=len(self._buffer),
            status=STATUS_ACTIVE if self._is_running else STATUS_STOPPED,
        )
        await self._repository.update_metrics(metrics)
