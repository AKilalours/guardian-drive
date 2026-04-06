from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class TaskBInference:
    """
    Task B learned inference stub.
    The trained task_b_tcn.pt uses a custom TCN architecture that requires
    raw multi-channel physiological sequences, not the flat feature vector
    passed here. Task B runtime is handled by models/task_b.py webcam +
    physio rule path instead.
    """
    weights_path: Optional[Path] = None

    def __post_init__(self):
        if self.weights_path is None:
            env = os.getenv("GD_TASK_B_MLP_WEIGHTS", "").strip()
            self.weights_path = Path(env) if env else None
        self._model = None

    def _load(self, n_features: int):
        return None  # TCN requires sequence input — not compatible with flat features

    def predict(self, x: np.ndarray) -> Optional[float]:
        return None  # signals models/task_b.py to use rule-based + webcam path
