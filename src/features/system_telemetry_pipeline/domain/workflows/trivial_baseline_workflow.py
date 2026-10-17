from collections import Counter
from src.features.system_telemetry_pipeline.contracts.telemetry_pipeline_types import (
    PipelineTaskSpecDTO,
    TelemetryPipelineModel,
    BaselineEvaluationResult,
)
from src.features.system_telemetry_pipeline.domain.rules.telemetry_split_rules import TelemetrySplitRules
from src.features.system_telemetry_pipeline.domain.machines.telemetry_pipeline_machine import (
    TelemetryPipelineMachine,
    STATUS_SPECIFIED,
    STATUS_SPLIT_VALIDATED,
    STATUS_BASELINE_ESTABLISHED,
)
from src.features.system_telemetry_pipeline.infrastructure.ports.telemetry_pipeline_repository_port import (
    TelemetryPipelineRepositoryPort,
)
from src.shared.types.envelope_types import TraceContext
from src.shared.utils.date_utils import generate_uuidv7_str, get_utc_iso_timestamp


class TrivialBaselineWorkflow:
    def __init__(self, repo: TelemetryPipelineRepositoryPort) -> None:
        self._repo = repo

    async def execute_foundation_flow(
        self, ctx: TraceContext, dto: PipelineTaskSpecDTO
    ) -> TelemetryPipelineModel:
        current_state = STATUS_SPECIFIED

        split_result = TelemetrySplitRules.execute_leak_free_split(
            records=dto.records,
            train_ratio=dto.train_ratio,
        )
        current_state = TelemetryPipelineMachine.transition(
            current_state, STATUS_SPLIT_VALIDATED
        )

        train_labels = [r.target_label for r in split_result.train_records]
        counter = Counter(train_labels)
        majority_class = counter.most_common(1)[0][0] if counter else 0

        val_labels = [r.target_label for r in split_result.val_records]
        if len(val_labels) > 0:
            correct_count = sum(1 for y in val_labels if y == majority_class)
            accuracy = correct_count / len(val_labels)
            tp = sum(1 for y in val_labels if y == majority_class and y == 1)
            predicted_pos = len(val_labels) if majority_class == 1 else 0
            actual_pos = sum(1 for y in val_labels if y == 1)
            precision = tp / predicted_pos if predicted_pos > 0 else 0.0
            recall = tp / actual_pos if actual_pos > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
        else:
            accuracy = 0.0
            f1 = 0.0

        baseline = BaselineEvaluationResult(
            majority_class=majority_class,
            accuracy=accuracy,
            f1_score=f1,
            sample_count=len(val_labels),
        )

        current_state = TelemetryPipelineMachine.transition(
            current_state, STATUS_BASELINE_ESTABLISHED
        )

        model = TelemetryPipelineModel(
            pipeline_id=generate_uuidv7_str(),
            task_name=dto.task_name,
            status=current_state,
            baseline_result=baseline,
            created_at_iso=get_utc_iso_timestamp(),
        )
        return await self._repo.save(ctx, model)
