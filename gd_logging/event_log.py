from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class EventLogger:
    """Append-only JSONL event log.

    This enables replay, audits, and debugging. Every window emits:
    - raw SQI summary
    - selected features (not raw ECG by default)
    - model outputs
    - policy action + reason

    For privacy, you should *not* log raw waveforms by default.
    """

    path: Path

    def __post_init__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event_type: str, payload: Dict[str, Any]) -> None:
        rec = {
            "ts_unix": time.time(),
            "type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


