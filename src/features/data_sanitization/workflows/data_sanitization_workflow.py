from typing import Sequence
from src.features.data_sanitization.types.data_sanitization_types import (
    SanitizationProfileModel,
    SanitizationConfigDTO,
    CleanedSampleDTO,
)
from src.features.data_sanitization.rules.data_sanitization_rules import DataSanitizationRules
from src.features.data_sanitization.machines.data_sanitization_machine import (
    DataSanitizationMachine,
    STATUS_FITTED,
)
from src.features.data_sanitization.repository.data_sanitization_repository import (
    DataSanitizationRepositoryPort,
)
from src.shared.types.envelope_types import TraceContext
from src.shared.errors.app_errors import NotFoundError
from src.shared.utils.date_utils import generate_uuidv7_str, get_utc_iso_timestamp


class DataSanitizationWorkflow:
    def __init__(self, repo: DataSanitizationRepositoryPort) -> None:
        self._repo = repo

    async def execute_fit_flow(
        self,
        ctx: TraceContext,
        train_feature_matrix: Sequence[Sequence[float]],
        config: SanitizationConfigDTO | None = None,
    ) -> SanitizationProfileModel:
        cfg = config or SanitizationConfigDTO()
        stats = DataSanitizationRules.compute_feature_statistics(train_feature_matrix)

        profile = SanitizationProfileModel(
            profile_id=generate_uuidv7_str(),
            feature_count=len(stats),
            statistics=stats,
            config=cfg,
            status=STATUS_FITTED,
            created_at_iso=get_utc_iso_timestamp(),
        )
        return await self._repo.save_profile(ctx, profile)

    async def execute_transform_flow(
        self,
        ctx: TraceContext,
        profile_id: str,
        samples: Sequence[tuple[str, Sequence[float], int]],
    ) -> tuple[CleanedSampleDTO, ...]:
        profile = await self._repo.find_profile_by_id(ctx, profile_id)
        if not profile:
            raise NotFoundError(f"Sanitization profile {profile_id} not found.")

        cleaned: list[CleanedSampleDTO] = []
        for s_id, feats, label in samples:
            cleaned.append(
                DataSanitizationRules.sanitize_sample(
                    sample_id=s_id,
                    features=feats,
                    label=label,
                    statistics=profile.statistics,
                    config=profile.config,
                )
            )
        return tuple(cleaned)
