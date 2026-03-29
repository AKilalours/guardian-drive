from __future__ import annotations

from typing import Dict, List


def compute_motion_artifact_score(samples: List[float]) -> Dict[str, float]:
    if len(samples) < 4:
        return {"motion_score": 1.0, "hf_proxy": 0.0}

    diffs = [abs(samples[i] - samples[i - 1]) for i in range(1, len(samples))]
    hf_proxy = sum(diffs) / max(len(diffs), 1)
    motion_score = min(1.0, hf_proxy * 20.0)
    return {"motion_score": motion_score, "hf_proxy": hf_proxy}
