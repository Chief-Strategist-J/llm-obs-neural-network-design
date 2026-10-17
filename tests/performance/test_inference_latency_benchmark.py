import time
import torch
from src.features.network_spec.infrastructure.adapters.neural_compute_adapter import (
    PyTorchNeuralArchitecture,
)


def test_pytorch_forward_inference_latency_benchmark():
    input_dim = 64
    hidden_dim = 128
    num_classes = 4
    layer_count = 3

    model = PyTorchNeuralArchitecture(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        layer_count=layer_count,
    )
    model.eval()

    batch_input = torch.randn(64, input_dim)

    with torch.no_grad():
        for _ in range(10):
            _ = model(batch_input)

    iterations = 100
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            out = model(batch_input)
    total_time = time.perf_counter() - t0
    avg_latency_ms = (total_time / iterations) * 1000.0

    assert out.shape == (64, num_classes)
    assert avg_latency_ms < 15.0

