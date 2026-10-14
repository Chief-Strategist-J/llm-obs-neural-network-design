import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from src.features.network_spec.types.network_spec_types import (
    NetworkSpecModel,
    TinyBatchOverfitCommand,
)
from src.shared.types.envelope_types import TraceContext


class NetworkSpecRepositoryPort(ABC):
    @abstractmethod
    async def save(self, ctx: TraceContext, model: NetworkSpecModel) -> NetworkSpecModel:
        pass

    @abstractmethod
    async def find_by_id(self, ctx: TraceContext, spec_id: str) -> NetworkSpecModel | None:
        pass


class NeuralComputePort(ABC):
    @abstractmethod
    def fit_and_verify_overfit(
        self, model_spec: NetworkSpecModel, command: TinyBatchOverfitCommand
    ) -> tuple[float, float, int, bool]:
        pass

    @abstractmethod
    def save_checkpoint(self, spec_id: str, checkpoint_dir: str) -> str:
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> bool:
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


class PyTorchNeuralArchitecture(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, layer_count: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        blocks = [ResidualBlock(hidden_dim) for _ in range(max(1, layer_count))]
        self.backbone = nn.Sequential(*blocks)
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
        h = self.backbone(h)
        return self.head(h)


class PyTorchNeuralComputeAdapter(NeuralComputePort):
    def __init__(self) -> None:
        self._active_model: PyTorchNeuralArchitecture | None = None
        self._active_optimizer: torch.optim.Optimizer | None = None

    def fit_and_verify_overfit(
        self, model_spec: NetworkSpecModel, command: TinyBatchOverfitCommand
    ) -> tuple[float, float, int, bool]:
        device = torch.device("cpu")
        net = PyTorchNeuralArchitecture(
            input_dim=model_spec.input_dim,
            hidden_dim=model_spec.hidden_dim,
            num_classes=model_spec.num_classes,
            layer_count=model_spec.layer_count,
        ).to(device)
        net.train()

        x_list = [list(s.features) for s in command.samples]
        y_list = [s.label for s in command.samples]

        inputs = torch.tensor(x_list, dtype=torch.float32, device=device)
        targets = torch.tensor(y_list, dtype=torch.long, device=device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.05, weight_decay=1e-4)

        initial_loss = 0.0
        final_loss = 0.0
        steps = 0

        for step in range(command.max_steps):
            optimizer.zero_grad()
            logits = net(inputs)
            loss = criterion(logits, targets)

            if step == 0:
                initial_loss = float(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            final_loss = float(loss.item())
            steps += 1
            if final_loss <= command.target_loss_threshold:
                break

        is_overfit = final_loss <= command.target_loss_threshold
        if is_overfit:
            self._active_model = net
            self._active_optimizer = optimizer

        return initial_loss, final_loss, steps, is_overfit

    def save_checkpoint(self, spec_id: str, checkpoint_dir: str) -> str:
        if self._active_model is None:
            raise ValueError("No active trained model to persist.")
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"{spec_id}.pt")
        torch.save(
            {
                "spec_id": spec_id,
                "model_state_dict": self._active_model.state_dict(),
                "optimizer_state_dict": (
                    self._active_optimizer.state_dict() if self._active_optimizer else None
                ),
            },
            path,
        )
        return path

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        if not os.path.exists(checkpoint_path):
            return False
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if self._active_model is not None and "model_state_dict" in checkpoint:
            self._active_model.load_state_dict(checkpoint["model_state_dict"])
            if self._active_optimizer and checkpoint.get("optimizer_state_dict"):
                self._active_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            return True
        return False
