from __future__ import annotations

from typing import Dict, List


def estimate_resp_from_ecg(samples: List[float], fs_hz: float) -> Dict[str, float]:
    if len(samples) < max(10, int(fs_hz)):
        return {"resp_rate_bpm": 0.0, "confidence": 0.0}
    return {"resp_rate_bpm": 12.0, "confidence": 0.2}
