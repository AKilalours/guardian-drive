from __future__ import annotations

from typing import Dict, List


def compute_contact_quality(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"contact_score": 0.0, "signal_range": 0.0, "dc_spread": 0.0}

    smin = min(samples)
    smax = max(samples)
    signal_range = float(smax - smin)
    mean = sum(samples) / len(samples)
    dc_spread = sum(abs(x - mean) for x in samples) / len(samples)

    score = 0.0
    if signal_range > 0.02:
        score += 0.45
    if dc_spread > 0.005:
        score += 0.35
    if signal_range > 0.05:
        score += 0.20

    return {
        "contact_score": max(0.0, min(1.0, score)),
        "signal_range": signal_range,
        "dc_spread": dc_spread,
    }
