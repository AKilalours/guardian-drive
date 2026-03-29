"""Tiny MLP for per-window feature vectors.

This is a pragmatic bridge between rule-based heuristics and a full temporal model.
It runs fast on Pi 5 and is easy to calibrate.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.sigmoid(logits).squeeze(-1)
