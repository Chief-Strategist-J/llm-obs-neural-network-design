import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from src.features.neural_learning_core.types.neural_learning_types import (
    OverfitVerificationConfigDTO,
    OverfitVerificationResult,
)


class NeuralModelPort(ABC):
    @abstractmethod
    def fit_tiny_batch(self, config: OverfitVerificationConfigDTO) -> OverfitVerificationResult:
        pass


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        return self.act(out + residual)


class PyTorchResidualTelemetryModel(nn.Module, NeuralModelPort):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res_block = ResidualBlock(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.res_block(h)
        return self.head(h)

    def fit_tiny_batch(self, config: OverfitVerificationConfigDTO) -> OverfitVerificationResult:
        device = torch.device("cpu")
        self.to(device)
        self.train()

        x_list = [list(s.features) for s in config.samples]
        y_list = [s.label for s in config.samples]

        inputs = torch.tensor(x_list, dtype=torch.float32, device=device)
        targets = torch.tensor(y_list, dtype=torch.long, device=device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.learning_rate, weight_decay=1e-4)

        initial_loss = 0.0
        final_loss = 0.0
        steps = 0

        for step in range(config.max_steps):
            optimizer.zero_grad()
            logits = self(inputs)
            loss = criterion(logits, targets)

            if step == 0:
                initial_loss = float(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            final_loss = float(loss.item())
            steps += 1
            if final_loss <= config.target_loss_threshold:
                break

        is_overfitted = final_loss <= config.target_loss_threshold
        status = (
            "TINY_BATCH_OVERFIT_PASSED"
            if is_overfitted
            else "TINY_BATCH_OVERFIT_FAILED"
        )

        return OverfitVerificationResult(
            initial_loss=initial_loss,
            final_loss=final_loss,
            steps_taken=steps,
            is_overfitted=is_overfitted,
            status=status,
        )
