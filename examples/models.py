from __future__ import annotations

import torch
from torch import nn


class RegressionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class Classifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.linear(x)}


class MultiInputRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        x = torch.stack((left, right), dim=1).float()
        return self.linear(x).squeeze(-1)


def mae_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (input - target).abs().mean()


def rounded_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true == y_pred.round()).float().mean().item()
