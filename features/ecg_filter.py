from __future__ import annotations

from typing import List


def detrend_mean(samples: List[float]) -> List[float]:
    if not samples:
        return []
    mean = sum(samples) / len(samples)
    return [x - mean for x in samples]
