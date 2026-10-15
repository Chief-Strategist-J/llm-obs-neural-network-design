import pytest
from httpx import AsyncClient, ASGITransport
from src.api.rest.v1.app import app


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_network_spec_creation_and_overfit_api(tmp_path):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        create_payload = {
            "name": "RestApiResidualNet",
            "architecture_type": "transformer",
            "input_dim": 4,
            "hidden_dim": 32,
            "num_classes": 2,
            "layer_count": 2,
            "parameter_budget": 50000,
        }
        res = await ac.post("/v1/network-spec", json=create_payload)
        assert res.status_code == 201
        data = res.json()["data"]
        spec_id = data["id"]
        assert data["status"] == "PENDING"

        overfit_payload = {
            "spec_id": spec_id,
            "max_steps": 50,
            "target_loss_threshold": 0.08,
            "samples": [
                {"features": [1.0, 0.0, -1.0, 0.5], "label": 0},
                {"features": [0.0, 1.0, 1.0, -0.5], "label": 1},
                {"features": [1.2, -0.1, -0.8, 0.4], "label": 0},
                {"features": [-0.2, 0.9, 1.1, -0.4], "label": 1},
            ],
        }
        overfit_res = await ac.post("/v1/network-spec/verify-overfit", json=overfit_payload)
        assert overfit_res.status_code == 200
        overfit_data = overfit_res.json()["data"]
        assert overfit_data["is_overfit_verified"] is True
        assert overfit_data["status"] == "TINY_BATCH_OVERFIT_VERIFIED"
        assert overfit_data["final_loss"] <= 0.08

        train_payload = {
            "spec_id": spec_id,
            "epochs": 3,
            "batch_size": 4,
            "target_checkpoint_dir": str(tmp_path / "checkpoints"),
            "train_samples": [
                {"features": [1.0, 0.0, -1.0, 0.5], "label": 0},
                {"features": [0.0, 1.0, 1.0, -0.5], "label": 1},
                {"features": [1.2, -0.1, -0.8, 0.4], "label": 0},
                {"features": [-0.2, 0.9, 1.1, -0.4], "label": 1},
            ],
            "val_samples": [
                {"features": [1.0, 0.0, -1.0, 0.5], "label": 0},
                {"features": [0.0, 1.0, 1.0, -0.5], "label": 1},
            ],
        }
        train_res = await ac.post("/v1/network-spec/train", json=train_payload)
        assert train_res.status_code == 200
        train_data = train_res.json()["data"]
        assert train_data["spec_id"] == spec_id
        assert train_data["trained_epochs"] == 3
        assert "checkpoint_path" in train_data
        assert "checkpoint_sha256" in train_data



@pytest.mark.asyncio
async def test_telemetry_pipeline_api():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "task_name": "api_test_task",
            "records": [
                {"record_id": "r1", "trace_id": "t1", "timestamp_epoch_ms": 1000, "features": [0.1], "target_label": 0},
                {"record_id": "r2", "trace_id": "t1", "timestamp_epoch_ms": 1100, "features": [0.2], "target_label": 0},
                {"record_id": "r3", "trace_id": "t2", "timestamp_epoch_ms": 2000, "features": [0.3], "target_label": 1},
            ],
            "train_ratio": 0.5,
        }
        res = await ac.post("/v1/telemetry-pipeline", json=payload)
        assert res.status_code == 201
        data = res.json()["data"]
        assert data["status"] == "BASELINE_ESTABLISHED"
        assert "baseline_result" in data
