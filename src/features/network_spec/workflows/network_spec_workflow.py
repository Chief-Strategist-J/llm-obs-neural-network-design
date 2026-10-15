from src.features.network_spec.types.network_spec_types import (
    NetworkSpecCreateDTO,
    NetworkSpecModel,
    TinyBatchOverfitCommand,
    DatasetTrainingCommand,
    DatasetTrainingResult,
)
from src.features.network_spec.rules.network_spec_rules import NetworkSpecRules
from src.features.network_spec.repository.neural_compute_adapter import (
    NetworkSpecRepositoryPort,
    NeuralComputePort,
)
from src.features.network_spec.machines.network_spec_machine import (
    NetworkSpecMachine,
    STATUS_PENDING,
    STATUS_TINY_BATCH_OVERFIT_VERIFIED,
    STATUS_TRAINED,
)
from src.shared.types.envelope_types import TraceContext
from src.shared.errors.app_errors import ValidationError, NotFoundError
from src.shared.utils.date_utils import generate_uuidv7_str


class NetworkSpecWorkflow:
    def __init__(
        self, repo: NetworkSpecRepositoryPort, compute: NeuralComputePort
    ) -> None:
        self._repo = repo
        self._compute = compute

    async def execute_creation_flow(
        self, ctx: TraceContext, dto: NetworkSpecCreateDTO
    ) -> NetworkSpecModel:
        if not NetworkSpecRules.can_create_spec(dto):
            raise ValidationError("Network specification violates parameter density rules.")

        model = NetworkSpecModel(
            id=generate_uuidv7_str(),
            name=dto.name,
            architecture_type=dto.architecture_type,
            input_dim=dto.input_dim,
            hidden_dim=dto.hidden_dim,
            num_classes=dto.num_classes,
            layer_count=dto.layer_count,
            parameter_budget=dto.parameter_budget,
            status=STATUS_PENDING,
        )
        return await self._repo.save(ctx, model)

    async def execute_overfit_verification_flow(
        self, ctx: TraceContext, command: TinyBatchOverfitCommand
    ) -> NetworkSpecModel:
        existing = await self._repo.find_by_id(ctx, command.spec_id)
        if not existing:
            raise NotFoundError(f"Network specification {command.spec_id} not found.")

        initial_loss, final_loss, steps, is_overfit = self._compute.fit_and_verify_overfit(
            existing, command
        )

        next_status = (
            NetworkSpecMachine.transition(existing.status, STATUS_TINY_BATCH_OVERFIT_VERIFIED)
            if is_overfit
            else existing.status
        )

        updated_model = NetworkSpecModel(
            id=existing.id,
            name=existing.name,
            architecture_type=existing.architecture_type,
            input_dim=existing.input_dim,
            hidden_dim=existing.hidden_dim,
            num_classes=existing.num_classes,
            layer_count=existing.layer_count,
            parameter_budget=existing.parameter_budget,
            status=next_status,
            initial_loss=initial_loss,
            final_loss=final_loss,
            steps_taken=steps,
            is_overfit_verified=is_overfit,
        )
        return await self._repo.save(ctx, updated_model)

    async def execute_dataset_training_flow(
        self, ctx: TraceContext, command: DatasetTrainingCommand
    ) -> tuple[NetworkSpecModel, DatasetTrainingResult]:
        existing = await self._repo.find_by_id(ctx, command.spec_id)
        if not existing:
            raise NotFoundError(f"Network specification {command.spec_id} not found.")

        train_tuples = [(list(s.features), s.label) for s in command.train_samples]
        val_tuples = [(list(s.features), s.label) for s in command.val_samples]

        epochs, tr_loss, val_loss, val_acc, ckpt_path, ckpt_sha = self._compute.fit_dataset(
            model_spec=existing,
            train_samples=train_tuples,
            val_samples=val_tuples,
            epochs=command.epochs,
            batch_size=command.batch_size,
            learning_rate=command.learning_rate,
            weight_decay=command.weight_decay,
            checkpoint_dir=command.target_checkpoint_dir,
        )

        next_status = NetworkSpecMachine.transition(existing.status, STATUS_TRAINED)

        updated_model = NetworkSpecModel(
            id=existing.id,
            name=existing.name,
            architecture_type=existing.architecture_type,
            input_dim=existing.input_dim,
            hidden_dim=existing.hidden_dim,
            num_classes=existing.num_classes,
            layer_count=existing.layer_count,
            parameter_budget=existing.parameter_budget,
            status=next_status,
            initial_loss=existing.initial_loss,
            final_loss=tr_loss,
            steps_taken=epochs,
            is_overfit_verified=existing.is_overfit_verified,
        )
        saved_model = await self._repo.save(ctx, updated_model)

        training_result = DatasetTrainingResult(
            spec_id=existing.id,
            trained_epochs=epochs,
            train_loss=tr_loss,
            val_loss=val_loss,
            val_accuracy=val_acc,
            checkpoint_path=ckpt_path,
            checkpoint_sha256=ckpt_sha,
        )
        return saved_model, training_result
