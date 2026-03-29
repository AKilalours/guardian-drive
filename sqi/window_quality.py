from __future__ import annotations

from typing import Dict, List

from sqi.contact_quality import compute_contact_quality
from sqi.motion_artifact import compute_motion_artifact_score


def compute_window_quality(samples: List[float]) -> Dict[str, object]:
    c = compute_contact_quality(samples)
    m = compute_motion_artifact_score(samples)

    score = 0.65 * c["contact_score"] + 0.35 * (1.0 - m["motion_score"])
    abstain = score < 0.40

    return {
        "overall_score": max(0.0, min(1.0, score)),
        "abstain": abstain,
        "contact": c,
        "motion": m,
        "summary": "usable" if not abstain else "poor",
    }
