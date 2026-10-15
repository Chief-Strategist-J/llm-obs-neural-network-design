import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Sequence[float] | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ClassBalancedLoss(nn.Module):
    def __init__(
        self,
        class_weights: Sequence[float],
        loss_type: str = "focal",
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weights = torch.tensor(class_weights, dtype=torch.float32)
        self.loss_type = loss_type
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)

        if self.loss_type == "focal":
            ce_loss = F.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
            weights_t = self.weights.gather(0, targets)
            cb_loss = weights_t * focal_loss
        else:
            ce_loss = F.cross_entropy(logits, targets, weight=self.weights, reduction="none")
            cb_loss = ce_loss

        if self.reduction == "mean":
            return cb_loss.mean()
        elif self.reduction == "sum":
            return cb_loss.sum()
        return cb_loss
