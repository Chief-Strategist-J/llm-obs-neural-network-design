from typing import Any, Mapping, Sequence
from src.shared.errors.app_errors import ValidationError
from src.features.system_telemetry_pipeline.types.telemetry_pipeline_types import (
    TelemetryRecordDTO,
    PipelineTaskSpecDTO,
    TelemetryPipelineModel,
)


class TelemetryPipelineSchema:
    @staticmethod
    def from_api(payload: Mapping[str, Any]) -> PipelineTaskSpecDTO:
        task_name = payload.get("task_name")
        modality = payload.get("modality", "distributed_trace_sequence")
        metric_target = payload.get("metric_target", "f1_macro")
        strategy = payload.get("split_strategy", "temporal_group")
        ratio = payload.get("train_ratio", 0.8)
        raw_records = payload.get("records")

        if not task_name or not isinstance(task_name, str):
            raise ValidationError("Field 'task_name' must be a non-empty string.")
        if not isinstance(raw_records, (list, tuple)) or len(raw_records) == 0:
            raise ValidationError("Field 'records' must be a non-empty sequence.")
        if not isinstance(ratio, (int, float)) or not (0.1 <= ratio <= 0.9):
            raise ValidationError("Field 'train_ratio' must be between 0.1 and 0.9.")

        parsed_records: list[TelemetryRecordDTO] = []
        for item in raw_records:
            if not isinstance(item, dict):
                raise ValidationError("Each record item must be an object.")
            rec_id = item.get("record_id", "")
            trace_id = item.get("trace_id", "")
            ts = item.get("timestamp_epoch_ms", 0)
            feats = item.get("features", [])
            label = item.get("target_label", 0)
            if not rec_id or not trace_id:
                raise ValidationError("Record must have 'record_id' and 'trace_id'.")
            parsed_records.append(
                TelemetryRecordDTO(
                    record_id=str(rec_id),
                    trace_id=str(trace_id),
                    timestamp_epoch_ms=int(ts),
                    features=tuple(float(x) for x in feats),
                    target_label=int(label),
                )
            )

        return PipelineTaskSpecDTO(
            task_name=task_name,
            modality=str(modality),
            metric_target=str(metric_target),
            records=tuple(parsed_records),
            split_strategy=str(strategy),
            train_ratio=float(ratio),
        )

    @staticmethod
    def to_api(model: TelemetryPipelineModel) -> Mapping[str, Any]:
        return {
            "pipeline_id": model.pipeline_id,
            "task_name": model.task_name,
            "status": model.status,
            "baseline_result": {
                "majority_class": model.baseline_result.majority_class,
                "accuracy": model.baseline_result.accuracy,
                "f1_score": model.baseline_result.f1_score,
                "sample_count": model.baseline_result.sample_count,
            },
            "created_at_iso": model.created_at_iso,
        }
