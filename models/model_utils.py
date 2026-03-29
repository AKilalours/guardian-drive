from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import joblib


def default_artifacts_dir() -> Path:
    return Path(os.getenv("GD_ARTIFACTS_DIR", "artifacts"))


def load_joblib(path: Path):
    try:
        if path.exists():
            return joblib.load(path)
    except Exception:
        return None
    return None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
