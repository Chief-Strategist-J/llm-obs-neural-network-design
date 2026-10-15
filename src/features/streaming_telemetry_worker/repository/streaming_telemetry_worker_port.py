from __future__ import annotations
from typing import Protocol, Sequence
from src.features.streaming_telemetry_worker.types.streaming_telemetry_worker_types import (
    MicroBatchDTO,
    DeadLetterMessageDTO,
    WorkerMetricsDTO,
)
from src.shared.types.envelope_types import Result
from src.shared.errors.app_errors import ApplicationError


class StreamingTelemetryWorkerRepositoryPort(Protocol):
    async def save_micro_batch(self, batch: MicroBatchDTO) -> Result[MicroBatchDTO, ApplicationError]:
        ...

    async def save_dead_letter(self, dlq: DeadLetterMessageDTO) -> Result[DeadLetterMessageDTO, ApplicationError]:
        ...

    async def list_micro_batches(self, limit: int = 50) -> Result[Sequence[MicroBatchDTO], ApplicationError]:
        ...

    async def list_dead_letters(self, limit: int = 50) -> Result[Sequence[DeadLetterMessageDTO], ApplicationError]:
        ...

    async def update_metrics(self, metrics: WorkerMetricsDTO) -> Result[WorkerMetricsDTO, ApplicationError]:
        ...

    async def get_metrics(self) -> Result[WorkerMetricsDTO, ApplicationError]:
        ...
