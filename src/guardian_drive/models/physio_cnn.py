from __future__ import annotations

import torch
from torch import nn


class PhysioCNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        kernel_size: int = 7,
        dropout: float = 0.2,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in hidden_channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            ]
            c_in = c_out
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c_in, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.head(z)
