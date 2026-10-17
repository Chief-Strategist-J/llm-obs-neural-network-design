from typing import Any, Mapping
from fastapi import APIRouter, Request, Response, status
from src.features.data_sanitization.service.data_sanitization_service import DataSanitizationService
from src.features.data_sanitization.infrastructure.adapters.data_sanitization_repository_adapter import InMemoryDataSanitizationRepository
from src.features.data_sanitization.contracts.data_sanitization_types import SanitizationConfigDTO
from src.shared.types.envelope_types import TraceContext
from src.infra.config.config_loader import AppConfig

router = APIRouter(
    prefix=AppConfig.get_endpoint("data_sanitization", "/v1/data-sanitization"),
    tags=["data-sanitization"],
)

_sanitization_repo = InMemoryDataSanitizationRepository()
_sanitization_service = DataSanitizationService(repository=_sanitization_repo)


from src.shared.errors.app_errors import ApplicationError, ValidationError


@router.post("/profile", status_code=status.HTTP_201_CREATED)
async def fit_sanitization_profile(request: Request, response: Response):
    ctx = getattr(request.state, "trace_context", None) or TraceContext(
        trace_id=request.headers.get("traceparent", "trace-synthetic-001")
    )
    payload = await request.json()
    train_features = payload.get("train_features")
    if not train_features or not isinstance(train_features, list):
        raise ValidationError("Field 'train_features' must be a non-empty list of float lists.")

    cfg_dict = payload.get("config", {})
    config = SanitizationConfigDTO(
        impute_nan_value=float(cfg_dict.get("impute_nan_value", 0.0)),
        impute_inf_value=float(cfg_dict.get("impute_inf_value", 1e6)),
        clip_outliers=bool(cfg_dict.get("clip_outliers", True)),
        outlier_std_threshold=float(cfg_dict.get("outlier_std_threshold", 5.0)),
        normalize_zscore=bool(cfg_dict.get("normalize_zscore", True)),
    )
    res = await _sanitization_service.fit_profile(ctx, train_features, config)
    if not res.is_ok or not res.value:
        raise res.error or ApplicationError("Failed to fit sanitization profile")
    p = res.value
    data = {
        "profile_id": p.profile_id,
        "feature_count": p.feature_count,
        "status": p.status,
        "created_at_iso": p.created_at_iso,
        "statistics": [
            {
                "feature_index": s.feature_index,
                "mean": s.mean,
                "std": s.std,
                "min_val": s.min_val,
                "max_val": s.max_val,
                "is_constant": s.is_constant,
            }
            for s in p.statistics
        ],
    }
    return {"success": True, "data": data, "meta": {"trace_id": ctx.trace_id}, "error": None}


@router.post("/transform", status_code=status.HTTP_200_OK)
async def transform_samples(request: Request, response: Response):
    ctx = getattr(request.state, "trace_context", None) or TraceContext(
        trace_id=request.headers.get("traceparent", "trace-synthetic-001")
    )
    payload = await request.json()
    profile_id = payload.get("profile_id")
    raw_samples = payload.get("samples")
    if not profile_id or not isinstance(profile_id, str):
        raise ValidationError("Field 'profile_id' is required.")
    if not raw_samples or not isinstance(raw_samples, list):
        raise ValidationError("Field 'samples' must be a non-empty list of samples.")

    formatted_samples = [
        (str(s["sample_id"]), [float(x) for x in s["features"]], int(s.get("label", 0)))
        for s in raw_samples
    ]
    res = await _sanitization_service.transform_samples(ctx, profile_id, formatted_samples)
    if not res.is_ok or not res.value:
        raise res.error or ApplicationError("Failed to transform samples")
    cleaned = [
        {
            "sample_id": s.sample_id,
            "cleaned_features": list(s.cleaned_features),
            "target_label": s.target_label,
            "imputed_count": s.imputed_count,
        }
        for s in res.value
    ]
    return {"success": True, "data": cleaned, "meta": {"trace_id": ctx.trace_id}, "error": None}
