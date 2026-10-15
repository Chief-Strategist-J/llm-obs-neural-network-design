import pytest
import httpx
from src.api.rest.v1.app import create_app


@pytest.mark.asyncio
async def test_e2e_error_envelopes_and_trace_headers():
    app = create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        custom_trace = "00-testtrace1234567890abcdef-0011223344556677-01"
        resp = await client.post(
            "/v1/data-sanitization/profile",
            json={"train_features": []},
            headers={"traceparent": custom_trace},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["success"] is False
        assert body["data"] is None
        assert "trace_id" in body["meta"]
        assert body["error"]["code"] == "VALIDATION_FAILED"

        resp_404 = await client.post(
            "/v1/data-sanitization/transform",
            json={
                "profile_id": "non_existent_profile_id_999",
                "samples": [{"sample_id": "s1", "features": [1.0, 2.0], "label": 0}],
            },
            headers={"traceparent": custom_trace},
        )
        assert resp_404.status_code == 404
        body_404 = resp_404.json()
        assert body_404["success"] is False
        assert body_404["data"] is None
        assert body_404["error"]["code"] == "RESOURCE_NOT_FOUND"

        invalid_spec = {
            "name": "bad_spec",
            "architecture_type": "invalid_arch_type_xyz",
            "input_dim": 10,
            "hidden_dim": 64,
            "num_classes": 2,
            "layer_count": 4,
            "parameter_budget": 1000,
        }
        resp_bad_arch = await client.post("/v1/network-spec", json=invalid_spec)
        assert resp_bad_arch.status_code == 400
        body_bad_arch = resp_bad_arch.json()
        assert body_bad_arch["success"] is False
        assert body_bad_arch["error"]["code"] == "VALIDATION_FAILED"
