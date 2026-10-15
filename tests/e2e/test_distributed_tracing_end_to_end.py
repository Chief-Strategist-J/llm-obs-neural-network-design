import pytest
import json
import httpx
from unittest.mock import MagicMock
from src.api.rest.v1.app import create_app
from src.infra.messaging.kafka_adapter import KafkaProducerAdapter, KafkaConsumerAdapter
from src.features.data_sanitization.index import (
    DataSanitizationService,
    InMemoryDataSanitizationRepository,
    SanitizationConfigDTO,
)
from src.features.network_spec.index import (
    NetworkSpecService,
    InMemoryNetworkSpecRepository,
    PyTorchNeuralComputeAdapter,
    NetworkSpecCreateDTO,
    TelemetrySampleDTO,
    DatasetTrainingCommand,
)
from src.shared.types.envelope_types import TraceContext
from python_shared.http.constants import HTTP_CONSTANTS


@pytest.mark.asyncio
async def test_end_to_end_distributed_tracing_and_messaging_pipeline():
    inbound_trace = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    inbound_corr_id = "corr-test-pipeline-999"

    app = create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        telemetry_payload = {
            "task_name": "tracing_e2e_task",
            "modality": "distributed_trace_sequence",
            "metric_target": "f1_macro",
            "split_strategy": "temporal_group",
            "train_ratio": 0.67,
            "records": [
                {
                    "record_id": "rec_t1",
                    "trace_id": "trace_001",
                    "timestamp_epoch_ms": 1000,
                    "features": [100.0, 1.0, 0.0],
                    "target_label": 0,
                },
                {
                    "record_id": "rec_t2",
                    "trace_id": "trace_002",
                    "timestamp_epoch_ms": 2000,
                    "features": [500.0, 2.0, 3.0],
                    "target_label": 1,
                },
                {
                    "record_id": "rec_t3",
                    "trace_id": "trace_003",
                    "timestamp_epoch_ms": 3000,
                    "features": [110.0, 1.0, 0.0],
                    "target_label": 0,
                },
            ],
        }

        resp = await client.post(
            "/v1/telemetry-pipeline",
            json=telemetry_payload,
            headers={
                HTTP_CONSTANTS.HEADER_TRACEPARENT: inbound_trace,
                HTTP_CONSTANTS.HEADER_X_CORRELATION_ID: inbound_corr_id,
            },
        )
        assert resp.status_code == 201
        resp_data = resp.json()
        assert resp_data["success"] is True
        assert resp.headers.get(HTTP_CONSTANTS.HEADER_X_CORRELATION_ID) == inbound_corr_id
        assert HTTP_CONSTANTS.HEADER_TRACEPARENT in resp.headers

        extracted_trace_id = resp_data["meta"]["trace_id"]
        assert extracted_trace_id in inbound_trace

    mock_producer = MagicMock()
    mock_producer.poll.return_value = 0
    captured_events = []
    mock_producer.produce.side_effect = lambda topic, key, value, headers: captured_events.append(
        {"topic": topic, "key": key, "value": value, "headers": headers}
    )

    kafka_producer = KafkaProducerAdapter(producer_instance=mock_producer)
    raw_span_event = {
        "trace_id": extracted_trace_id,
        "features": [100.0, 1.0, 0.0],
        "target_label": 0,
    }
    kafka_producer.produce(
        topic="telemetry.spans.raw",
        key=extracted_trace_id,
        value=json.dumps(raw_span_event).encode("utf-8"),
        headers={
            HTTP_CONSTANTS.HEADER_TRACEPARENT: inbound_trace,
            HTTP_CONSTANTS.HEADER_X_CORRELATION_ID: inbound_corr_id,
        },
    )
    kafka_producer.flush(1.0)

    assert len(captured_events) == 1
    produced = captured_events[0]
    headers_dict = {
        (h[0].decode("utf-8") if isinstance(h[0], bytes) else str(h[0])): (
            h[1].decode("utf-8") if isinstance(h[1], bytes) else str(h[1])
        )
        for h in produced["headers"]
    }
    assert headers_dict.get(HTTP_CONSTANTS.HEADER_TRACEPARENT) == inbound_trace
    assert headers_dict.get(HTTP_CONSTANTS.HEADER_X_CORRELATION_ID) == inbound_corr_id

    mock_consumer = MagicMock()
    mock_msg = MagicMock()
    mock_msg.topic.return_value = "telemetry.spans.raw"
    mock_msg.partition.return_value = 0
    mock_msg.offset.return_value = 101
    mock_msg.value.return_value = produced["value"]
    mock_msg.headers.return_value = [
        (HTTP_CONSTANTS.HEADER_TRACEPARENT.encode("utf-8"), inbound_trace.encode("utf-8")),
        (HTTP_CONSTANTS.HEADER_X_CORRELATION_ID.encode("utf-8"), inbound_corr_id.encode("utf-8")),
    ]
    mock_consumer.poll.return_value = mock_msg

    kafka_consumer = KafkaConsumerAdapter(consumer_instance=mock_consumer)

    consumed_trace_id = None
    consumed_payload = None

    def _sample_handler(ctx):
        nonlocal consumed_trace_id, consumed_payload
        consumed_trace_id = ctx.metadata.get("trace_id")
        consumed_payload = json.loads(ctx.payload.decode("utf-8"))
        return consumed_payload

    result = kafka_consumer.poll_with_pipeline(timeout=1.0, handler=_sample_handler)
    assert result is not None
    assert consumed_trace_id == extracted_trace_id
    assert consumed_payload["target_label"] == 0

    trace_ctx = TraceContext(trace_id=consumed_trace_id)
    sanitization_repo = InMemoryDataSanitizationRepository()
    sanitization_service = DataSanitizationService(repository=sanitization_repo)

    fit_res = await sanitization_service.fit_profile(
        trace_ctx,
        [[100.0, 1.0, 0.0], [500.0, 2.0, 3.0]],
        SanitizationConfigDTO(),
    )
    assert fit_res.is_ok is True
    profile = fit_res.value

    clean_res = await sanitization_service.transform_samples(
        trace_ctx,
        profile.profile_id,
        [("s1", consumed_payload["features"], consumed_payload["target_label"])],
    )
    assert clean_res.is_ok is True
    cleaned_sample = clean_res.value[0]

    net_repo = InMemoryNetworkSpecRepository()
    compute = PyTorchNeuralComputeAdapter()
    net_service = NetworkSpecService(repository=net_repo, compute=compute)

    create_res = await net_service.create_network_spec(
        trace_ctx,
        NetworkSpecCreateDTO(
            name="traced_model",
            architecture_type="mlp",
            input_dim=3,
            hidden_dim=16,
            num_classes=2,
            layer_count=2,
            parameter_budget=50000,
        ),
    )
    assert create_res.is_ok is True

    train_cmd = DatasetTrainingCommand(
        spec_id=create_res.value.id,
        train_samples=(TelemetrySampleDTO(features=cleaned_sample.cleaned_features, label=cleaned_sample.target_label),),
        val_samples=(TelemetrySampleDTO(features=cleaned_sample.cleaned_features, label=cleaned_sample.target_label),),
        epochs=5,
        batch_size=1,
        learning_rate=0.01,
        weight_decay=1e-4,
        target_checkpoint_dir="checkpoints/traced_e2e",
        loss_type="focal_loss",
    )
    train_res = await net_service.train_dataset(trace_ctx, train_cmd)
    assert train_res.is_ok is True
    assert train_res.value.trained_epochs == 5
