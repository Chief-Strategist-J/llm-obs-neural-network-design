import os
import hashlib
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Sequence
from src.features.network_spec.contracts.network_spec_types import (
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
    def fit_dataset(
        self,
        model_spec: NetworkSpecModel,
        train_samples: list[tuple[list[float], int]],
        val_samples: list[tuple[list[float], int]],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        checkpoint_dir: str,
        loss_type: str = "cross_entropy",
        focal_gamma: float = 2.0,
        class_weights: Sequence[float] | None = None,
    ) -> tuple[int, float, float, float, str, str]:
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


class SubconsciousLatentNetwork(nn.Module):
    def __init__(self, hidden_dim: int, layer_count: int) -> None:
        super().__init__()
        self.subconscious_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(max(1, layer_count // 2))]
        )
        self.latent_norm = nn.LayerNorm(hidden_dim)
        self.latent_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        subtext = self.subconscious_blocks(x)
        gate = self.latent_gate(subtext)
        return self.latent_norm(subtext * gate + x)


class PyTorchNeuralArchitecture(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, layer_count: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        blocks = [ResidualBlock(hidden_dim) for _ in range(max(1, layer_count))]
        self.surface_backbone = nn.Sequential(*blocks)
        self.subconscious_network = SubconsciousLatentNetwork(hidden_dim, layer_count)

        self.intent_head = nn.Linear(hidden_dim, num_classes)
        self.subconscious_head = nn.Linear(hidden_dim, max(2, num_classes))
        self.cognitive_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_head = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self, x: torch.Tensor, return_all_heads: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_enc = self.encoder(x)
        h_surface = self.surface_backbone(h_enc)
        h_subconscious = self.subconscious_network(h_enc)

        logits_intent = self.intent_head(h_surface)
        logits_subconscious = self.subconscious_head(h_subconscious)

        fused = torch.cat([h_surface, h_subconscious], dim=-1)
        h_fused = torch.relu(self.fusion_norm(self.cognitive_fusion(fused)))
        fused_decision = self.fusion_head(h_fused)

        if return_all_heads:
            return logits_intent, logits_subconscious, fused_decision
        return fused_decision


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
            intent_logits, sub_logits, fused_logits = net(inputs, return_all_heads=True)
            sub_targets = targets % max(2, model_spec.num_classes)

            loss_intent = criterion(intent_logits, targets)
            loss_sub = criterion(sub_logits, sub_targets)
            loss_fused = criterion(fused_logits, targets)

            loss = loss_fused + 0.3 * loss_intent + 0.2 * loss_sub

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

    def fit_dataset(
        self,
        model_spec: NetworkSpecModel,
        train_samples: list[tuple[list[float], int]],
        val_samples: list[tuple[list[float], int]],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        checkpoint_dir: str,
        loss_type: str = "cross_entropy",
        focal_gamma: float = 2.0,
        class_weights: Sequence[float] | None = None,
    ) -> tuple[int, float, float, float, str, str]:
        from src.features.adaptive_loss.infrastructure.adapters.torch_loss_modules import (
            FocalLoss,
            ClassBalancedLoss,
        )

        device = torch.device("cpu")
        net = PyTorchNeuralArchitecture(
            input_dim=model_spec.input_dim,
            hidden_dim=model_spec.hidden_dim,
            num_classes=model_spec.num_classes,
            layer_count=model_spec.layer_count,
        ).to(device)
        net.train()

        if loss_type == "focal_loss":
            criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        elif loss_type == "class_balanced" and class_weights is not None:
            criterion = ClassBalancedLoss(class_weights=class_weights, gamma=focal_gamma)
        else:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        sub_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        x_train = torch.tensor([s[0] for s in train_samples], dtype=torch.float32, device=device)
        y_train = torch.tensor([s[1] for s in train_samples], dtype=torch.long, device=device)
        x_val = torch.tensor([s[0] for s in val_samples], dtype=torch.float32, device=device)
        y_val = torch.tensor([s[1] for s in val_samples], dtype=torch.long, device=device)

        train_ds = torch.utils.data.TensorDataset(x_train, y_train)
        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=max(1, batch_size), shuffle=True
        )

        final_train_loss = 0.0
        for _ in range(epochs):
            net.train()
            epoch_loss = 0.0
            batch_count = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                intent_logits, sub_logits, fused_logits = net(batch_x, return_all_heads=True)
                sub_targets = batch_y % max(2, model_spec.num_classes)

                loss_fused = criterion(fused_logits, batch_y)
                loss_intent = criterion(intent_logits, batch_y)
                loss_sub = sub_criterion(sub_logits, sub_targets)

                loss = loss_fused + 0.3 * loss_intent + 0.2 * loss_sub
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += float(loss.item())
                batch_count += 1
            final_train_loss = epoch_loss / max(1, batch_count)

        net.eval()
        with torch.no_grad():
            val_logits = net(x_val)
            val_loss = float(criterion(val_logits, y_val).item())
            preds = torch.argmax(val_logits, dim=-1)
            correct = (preds == y_val).sum().item()
            val_accuracy = float(correct) / max(1, len(val_samples))

        self._active_model = net
        self._active_optimizer = optimizer

        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f"{model_spec.id}.pt")
        torch.save(
            {
                "spec_id": model_spec.id,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": final_train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            },
            ckpt_path,
        )

        hasher = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            hasher.update(f.read())
        sha256_hex = hasher.hexdigest()

        return epochs, final_train_loss, val_loss, val_accuracy, ckpt_path, sha256_hex

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
