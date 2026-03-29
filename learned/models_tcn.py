"""Lightweight Temporal ConvNet (TCN) for windowed physiological features.

Why this exists:
- Rule-based scoring is fine for demos, but a real system needs learned models
  that generalize across subjects and handle noisy sensors.

This TCN is tiny on purpose so it can run on Raspberry Pi 5.

Inputs:
- sequences of per-step feature vectors (e.g., 10s step, 30s window => 3 steps)
- Output:
- probability of drowsy/fatigued (Task B)

Note: training requires a real dataset (WESAD or your own seatbelt recordings).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, dilation=dilation, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, kernel_size=k, dilation=dilation, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = self.net(x)
        # crop to match input length (causal-ish)
        y = y[..., : x.shape[-1]]
        return y + self.down(x)


class TinyTCN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, levels: int = 3):
        super().__init__()
        blocks = []
        c_in = n_features
        for i in range(levels):
            blocks.append(TCNBlock(c_in, hidden, k=3, dilation=2**i))
            c_in = hidden
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_seq):
        """x_seq: (B, T, F)"""
        x = x_seq.transpose(1, 2)  # (B, F, T)
        z = self.tcn(x)
        logits = self.head(z)
        return torch.sigmoid(logits).squeeze(-1)
