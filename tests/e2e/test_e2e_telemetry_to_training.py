import pytest
import httpx
from src.api.rest.v1.app import create_app


@pytest.mark.asyncio
async def test_e2e_telemetry_to_training_api_lifecycle():
    app = create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json() == {"status": "ok"}

        raw_telemetry_payload = {
            "task_name": "e2e_trace_classification",
            "modality": "distributed_trace_sequence",
            "metric_target": "f1_macro",
            "split_strategy": "temporal_group",
            "train_ratio": 0.67,
            "records": [
                {
                    "record_id": "r1",
                    "trace_id": "t1",
                    "timestamp_epoch_ms": 1000,
                    "features": [50.0, 1.0, 10.0],
                    "target_label": 0,
                },
                {
                    "record_id": "r2",
                    "trace_id": "t2",
                    "timestamp_epoch_ms": 2000,
                    "features": [250.0, 2.0, 20.0],
                    "target_label": 1,
                },
                {
                    "record_id": "r3",
                    "trace_id": "t3",
                    "timestamp_epoch_ms": 3000,
                    "features": [60.0, 1.5, 15.0],
                    "target_label": 0,
                },
            ],
        }

        pipeline_resp = await client.post(
            "/v1/telemetry-pipeline",
            json=raw_telemetry_payload,
            headers={"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
        )
        assert pipeline_resp.status_code == 201
        p_data = pipeline_resp.json()
        assert p_data["success"] is True
        assert p_data["data"]["status"] == "BASELINE_ESTABLISHED"

        sanitization_profile_payload = {
            "train_features": [
                [50.0, 1.0, 10.0],
                [250.0, 2.0, 20.0],
            ],
            "config": {
                "impute_nan_value": 0.0,
                "impute_inf_value": 1e6,
                "clip_outliers": True,
                "outlier_std_threshold": 3.0,
                "normalize_zscore": True,
            },
        }

        profile_resp = await client.post(
            "/v1/data-sanitization/profile",
            json=sanitization_profile_payload,
            headers={"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
        )
        assert profile_resp.status_code == 201
        prof_data = profile_resp.json()
        assert prof_data["success"] is True
        profile_id = prof_data["data"]["profile_id"]

        transform_payload = {
            "profile_id": profile_id,
            "samples": [
                {"sample_id": "s1", "features": [50.0, 1.0, 10.0], "label": 0},
                {"sample_id": "s2", "features": [250.0, 2.0, 20.0], "label": 1},
                {"sample_id": "s3", "features": [60.0, 1.5, 15.0], "label": 0},
            ],
        }

        transform_resp = await client.post(
            "/v1/data-sanitization/transform",
            json=transform_payload,
        )
        assert transform_resp.status_code == 200
        trans_data = transform_resp.json()
        assert trans_data["success"] is True
        assert len(trans_data["data"]) == 3

        spec_payload = {
            "name": "e2e_mlp_classifier",
            "architecture_type": "mlp",
            "input_dim": 3,
            "hidden_dim": 16,
            "num_classes": 2,
            "layer_count": 2,
            "parameter_budget": 50000,
        }

        spec_resp = await client.post("/v1/network-spec", json=spec_payload)
        assert spec_resp.status_code == 201
        s_data = spec_resp.json()
        assert s_data["success"] is True
        spec_id = s_data["data"]["id"]

        overfit_samples = [
            {"features": s["cleaned_features"], "label": s["target_label"]}
            for s in trans_data["data"][:2]
        ]
        overfit_resp = await client.post(
            "/v1/network-spec/verify-overfit",
            json={"spec_id": spec_id, "samples": overfit_samples, "max_steps": 50, "target_loss_threshold": 0.08},
        )
        assert overfit_resp.status_code == 200
        ov_data = overfit_resp.json()
        assert ov_data["success"] is True
        assert ov_data["data"]["is_overfit_verified"] is True

        train_samples = [
            {"features": s["cleaned_features"], "label": s["target_label"]}
            for s in trans_data["data"][:2]
        ]
        val_samples = [
            {"features": s["cleaned_features"], "label": s["target_label"]}
            for s in trans_data["data"][2:]
        ]

        train_resp = await client.post(
            "/v1/network-spec/train",
            json={
                "spec_id": spec_id,
                "train_samples": train_samples,
                "val_samples": val_samples,
                "epochs": 10,
                "batch_size": 2,
                "learning_rate": 0.01,
            },
        )
        assert train_resp.status_code == 200
        tr_data = train_resp.json()
        assert tr_data["success"] is True
        assert tr_data["data"]["trained_epochs"] == 10
        assert tr_data["data"]["checkpoint_path"] != ""
