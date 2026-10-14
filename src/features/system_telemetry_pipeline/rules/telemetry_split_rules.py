from typing import Sequence
from src.features.system_telemetry_pipeline.types.telemetry_pipeline_types import (
    TelemetryRecordDTO,
    DatasetSplitResult,
)
from src.shared.errors.app_errors import ValidationError


class TelemetrySplitRules:
    @staticmethod
    def verify_split_leakage(split: DatasetSplitResult) -> None:
        train_traces_set = set(split.train_traces)
        val_traces_set = set(split.val_traces)

        overlap = train_traces_set.intersection(val_traces_set)
        if len(overlap) > 0:
            raise ValidationError(f"Data leakage detected! Traces overlap across splits: {overlap}")

        if len(split.train_records) > 0 and len(split.val_records) > 0:
            max_train_ts = max(r.timestamp_epoch_ms for r in split.train_records)
            min_val_ts = min(r.timestamp_epoch_ms for r in split.val_records)
            if max_train_ts > min_val_ts:
                raise ValidationError(
                    f"Temporal lookahead leakage detected! Max train ts ({max_train_ts}) > Min val ts ({min_val_ts})"
                )

    @staticmethod
    def execute_leak_free_split(
        records: Sequence[TelemetryRecordDTO], train_ratio: float
    ) -> DatasetSplitResult:
        traces_ordered: list[tuple[str, int]] = []
        seen_traces: set[str] = set()

        for rec in sorted(records, key=lambda r: r.timestamp_epoch_ms):
            if rec.trace_id not in seen_traces:
                seen_traces.add(rec.trace_id)
                traces_ordered.append((rec.trace_id, rec.timestamp_epoch_ms))

        split_idx = int(len(traces_ordered) * train_ratio)
        if split_idx == 0:
            split_idx = 1
        elif split_idx >= len(traces_ordered):
            split_idx = len(traces_ordered) - 1

        train_trace_ids = set(t[0] for t in traces_ordered[:split_idx])
        val_trace_ids = set(t[0] for t in traces_ordered[split_idx:])

        train_recs = [r for r in records if r.trace_id in train_trace_ids]
        val_recs = [r for r in records if r.trace_id in val_trace_ids]

        result = DatasetSplitResult(
            train_records=tuple(train_recs),
            val_records=tuple(val_recs),
            train_traces=tuple(train_trace_ids),
            val_traces=tuple(val_trace_ids),
        )
        TelemetrySplitRules.verify_split_leakage(result)
        return result
