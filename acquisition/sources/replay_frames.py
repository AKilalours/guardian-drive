"""Replay SensorFrame windows from a JSONL file produced by SensorFrame.to_json()."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator
import numpy as np

from acquisition.models import SensorFrame, TaskLabel

def iter_sensorframes(path: Path) -> Iterator[SensorFrame]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            obj=json.loads(line)
            yield SensorFrame(
                session_id=obj.get("session_id",""),
                subject_id=obj.get("subject_id",""),
                timestamp=float(obj.get("timestamp",0.0)),
                window_sec=float(obj.get("window_sec",30.0)),
                label=TaskLabel(obj.get("label","unknown")) if obj.get("label") in [e.value for e in TaskLabel] else TaskLabel.UNKNOWN,
                ecg=np.asarray(obj.get("ecg")) if obj.get("ecg") is not None else None,
                eda=np.asarray(obj.get("eda")) if obj.get("eda") is not None else None,
                respiration=np.asarray(obj.get("respiration")) if obj.get("respiration") is not None else None,
                accel=np.asarray(obj.get("accel")) if obj.get("accel") is not None else None,
                gyro=np.asarray(obj.get("gyro")) if obj.get("gyro") is not None else None,
                temperature=obj.get("temperature"),
                alcohol=obj.get("alcohol"),
                belt_tension=obj.get("belt_tension"),
            )
