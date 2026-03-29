"""Dataset utilities for learned models.

We keep this minimal and explicit.

Expected data layout for real recordings (recommended):
- One session per file, stored as NPZ or Parquet
- Contains timestamped streams for each sensor
- Contains labels for segments (drowsy/alert, arrhythmia/normal, crash events)

This repo ships a synthetic dataset generator so the training code runs end-to-end,
*but synthetic metrics are not real performance*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class SyntheticTaskBDataset:
    """Generate feature sequences that mimic alert vs drowsy.

    This is only for code-path validation.
    """

    n_samples: int = 4000
    seq_len: int = 6
    n_features: int = 10
    seed: int = 7

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        y = rng.integers(0, 2, size=(self.n_samples,)).astype(np.float32)
        X = rng.normal(0, 1, size=(self.n_samples, self.seq_len, self.n_features)).astype(np.float32)
        # Inject weak signal: drowsy has lower "activity" and higher "resp irregularity" proxies
        X[y == 1, :, 0] -= 0.8  # activity
        X[y == 1, :, 1] += 0.6  # resp irregularity
        return X, y
