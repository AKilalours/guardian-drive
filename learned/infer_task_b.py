from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np


@dataclass
class TaskBInference:
    """Optional learned inference for Task B.

    If weights are not configured, .predict() returns None.

    Configure:
      export GD_TASK_B_MLP_WEIGHTS=artifacts/task_b_mlp.pt
    """

    weights_path: Optional[Path] = None

    def __post_init__(self):
        if self.weights_path is None:
            env = os.getenv("GD_TASK_B_MLP_WEIGHTS", "").strip()
            self.weights_path = Path(env) if env else None
        self._model = None

    def _load(self, n_features: int):
        try:
            import torch
            from .mlp import TinyMLP
        except Exception:
            return None

        if not self.weights_path or not self.weights_path.exists():
            return None

        model = TinyMLP(n_features=n_features)
        state = torch.load(self.weights_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        self._model = model
        return model

    def predict(self, x: np.ndarray) -> Optional[float]:
        """x: (n_features,)"""
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        n_features = x.shape[1]
        if self._model is None:
            m = self._load(n_features)
            if m is None:
                return None
        import torch
        with torch.no_grad():
            p = float(self._model(torch.from_numpy(x))[0].item())
        return float(np.clip(p, 0.0, 1.0))
