from __future__ import annotations

import math
from typing import Dict, List


def rr_from_peak_times(peak_times_sec: List[float]) -> List[float]:
    if len(peak_times_sec) < 2:
        return []
    rr = []
    for i in range(1, len(peak_times_sec)):
        dt = peak_times_sec[i] - peak_times_sec[i - 1]
        if 0.3 <= dt <= 2.0:
            rr.append(dt)
    return rr


def compute_hrv(rr_sec: List[float]) -> Dict[str, float]:
    if not rr_sec:
        return {"hr_bpm": 0.0, "rmssd": 0.0, "sdnn": 0.0, "n_rr": 0}

    mean_rr = sum(rr_sec) / len(rr_sec)
    hr_bpm = 60.0 / mean_rr if mean_rr > 0 else 0.0

    if len(rr_sec) >= 2:
        diffs = [rr_sec[i] - rr_sec[i - 1] for i in range(1, len(rr_sec))]
        rmssd = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    else:
        rmssd = 0.0

    if len(rr_sec) >= 2:
        mean = mean_rr
        sdnn = math.sqrt(sum((x - mean) ** 2 for x in rr_sec) / len(rr_sec))
    else:
        sdnn = 0.0

    return {
        "hr_bpm": hr_bpm,
        "rmssd": rmssd,
        "sdnn": sdnn,
        "n_rr": len(rr_sec),
    }
