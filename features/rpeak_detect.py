from __future__ import annotations

from typing import Dict, List


def detect_rpeaks(samples: List[float], fs_hz: float) -> Dict[str, object]:
    if len(samples) < 5:
        return {"peak_indices": [], "peak_times_sec": []}

    mx = max(samples) if samples else 0.0
    thr = mx * 0.60
    refractory = max(1, int(0.25 * fs_hz))

    peaks: List[int] = []
    last_i = -refractory
    for i in range(1, len(samples) - 1):
        if i - last_i < refractory:
            continue
        if samples[i] > thr and samples[i] >= samples[i - 1] and samples[i] >= samples[i + 1]:
            peaks.append(i)
            last_i = i

    return {
        "peak_indices": peaks,
        "peak_times_sec": [i / fs_hz for i in peaks],
    }
