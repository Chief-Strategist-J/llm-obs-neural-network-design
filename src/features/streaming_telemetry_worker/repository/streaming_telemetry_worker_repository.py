from __future__ import annotations
from typing import Sequence
from src.features.streaming_telemetry_worker.repository.streaming_telemetry_worker_port import (
    StreamingTelemetryWorkerRepositoryPort,
)
from src.features.streaming_telemetry_worker.types.streaming_telemetry_worker_types import (
    MicroBatchDTO,
    DeadLetterMessageDTO,
    WorkerMetricsDTO,
)
from src.shared.types.envelope_types import Result
from src.shared.errors.app_errors import ApplicationError
from src.shared.constants.system_constants import STATUS_ACTIVE


class InMemoryStreamingTelemetryWorkerRepository(StreamingTelemetryWorkerRepositoryPort):
    def __init__(self) -> None:
        self._batches: list[MicroBatchDTO] = []
        self._dead_letters: list[DeadLetterMessageDTO] = []
        self._metrics: WorkerMetricsDTO = WorkerMetricsDTO(
            total_consumed=0,
            total_batched=0,
            total_dlq_routed=0,
            total_batches_dispatched=0,
            current_buffer_size=0,
            status=STATUS_ACTIVE,
        )

    async def save_micro_batch(self, batch: MicroBatchDTO) -> Result[MicroBatchDTO, ApplicationError]:
        self._batches.append(batch)
        return Result.ok(batch)

    async def save_dead_letter(self, dlq: DeadLetterMessageDTO) -> Result[DeadLetterMessageDTO, ApplicationError]:
        self._dead_letters.append(dlq)
        return Result.ok(dlq)

    async def list_micro_batches(self, limit: int = 50) -> Result[Sequence[MicroBatchDTO], ApplicationError]:
        return Result.ok(tuple(self._batches[-limit:]))

    async def list_dead_letters(self, limit: int = 50) -> Result[Sequence[DeadLetterMessageDTO], ApplicationError]:
        return Result.ok(tuple(self._dead_letters[-limit:]))

    async def update_metrics(self, metrics: WorkerMetricsDTO) -> Result[WorkerMetricsDTO, ApplicationError]:
        self._metrics = metrics
        return Result.ok(metrics)

    async def get_metrics(self) -> Result[WorkerMetricsDTO, ApplicationError]:
        return Result.ok(self._metrics)
