from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass
class StrokeScreenInput:
    face_present: bool = False
    face_confidence: float = 0.0
    mouth_asymmetry: float = 0.0   # 0..1, higher = worse
    eye_asymmetry: float = 0.0     # 0..1, higher = worse
    speech_prompt_completed: bool = False
    speech_slur_score: float = 0.0  # 0..1, higher = worse
    response_latency_sec: float = 0.0
    arm_drift_score: float = 0.0    # optional, 0..1
    manual_trigger: bool = False

    @classmethod
    def from_metrics(cls, metrics: Optional[Dict[str, object]]) -> "StrokeScreenInput":
        metrics = metrics or {}
        return cls(
            face_present=bool(metrics.get("face_present", False)),
            face_confidence=float(metrics.get("face_confidence", 0.0) or 0.0),
            mouth_asymmetry=float(metrics.get("mouth_asymmetry", 0.0) or 0.0),
            eye_asymmetry=float(metrics.get("eye_asymmetry", 0.0) or 0.0),
            speech_prompt_completed=bool(metrics.get("speech_prompt_completed", False)),
            speech_slur_score=float(metrics.get("speech_slur_score", 0.0) or 0.0),
            response_latency_sec=float(metrics.get("response_latency_sec", 0.0) or 0.0),
            arm_drift_score=float(metrics.get("arm_drift_score", 0.0) or 0.0),
            manual_trigger=bool(metrics.get("manual_trigger", False)),
        )


@dataclass
class StrokeScreenResult:
    task: str = "task_d_stroke_screen"
    label: str = "no_concern"
    score: float = 0.0
    severity: str = "none"
    rationale: List[str] = field(default_factory=list)
    should_prompt_fast_exam: bool = False
    recommended_action: str = "monitor"
    details: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class StrokeScreenTask:
    """
    This is not stroke diagnosis. It is a conservative screening layer.

    It should only say things like:
      - no concern
      - prompt FAST exam
      - possible neurologic emergency

    It should never claim confirmation.
    """

    def __init__(
        self,
        *,
        prompt_threshold: float = 0.42,
        emergency_threshold: float = 0.72,
        high_latency_sec: float = 3.0,
        very_high_latency_sec: float = 6.0,
    ) -> None:
        self.prompt_threshold = prompt_threshold
        self.emergency_threshold = emergency_threshold
        self.high_latency_sec = high_latency_sec
        self.very_high_latency_sec = very_high_latency_sec

    def run(self, inp: StrokeScreenInput) -> StrokeScreenResult:
        rationale: List[str] = []
        score = 0.0

        if inp.manual_trigger:
            score += 0.18
            rationale.append("manual neurologic concern trigger")

        if inp.face_present and inp.face_confidence >= 0.5:
            face_asym = max(inp.mouth_asymmetry, inp.eye_asymmetry)
            if face_asym >= 0.30:
                score += 0.28
                rationale.append("facial asymmetry elevated")
            if face_asym >= 0.55:
                score += 0.18
                rationale.append("facial asymmetry strongly elevated")
        else:
            rationale.append("face evidence weak or unavailable")

        if inp.speech_prompt_completed:
            if inp.speech_slur_score >= 0.30:
                score += 0.22
                rationale.append("speech abnormality detected")
            if inp.speech_slur_score >= 0.60:
                score += 0.15
                rationale.append("speech abnormality strongly elevated")
        else:
            if inp.response_latency_sec >= self.high_latency_sec:
                score += 0.16
                rationale.append("response latency elevated")
            if inp.response_latency_sec >= self.very_high_latency_sec:
                score += 0.12
                rationale.append("response latency severely elevated")

        if inp.arm_drift_score >= 0.35:
            score += 0.18
            rationale.append("possible arm drift")
        if inp.arm_drift_score >= 0.60:
            score += 0.10
            rationale.append("arm drift strongly elevated")

        # Require multi-signal concern before going hard.
        modalities = 0
        if max(inp.mouth_asymmetry, inp.eye_asymmetry) >= 0.30:
            modalities += 1
        if inp.speech_slur_score >= 0.30 or inp.response_latency_sec >= self.high_latency_sec:
            modalities += 1
        if inp.arm_drift_score >= 0.35:
            modalities += 1
        if inp.manual_trigger:
            modalities += 1

        if modalities < 2:
            score *= 0.72
            rationale.append("single-signal concern discounted")

        score = _clip01(score)

        if score >= self.emergency_threshold:
            label = "possible_neurologic_emergency"
            severity = "high"
            action = "emergency"
            prompt_fast = True
        elif score >= self.prompt_threshold:
            label = "prompt_fast_exam"
            severity = "moderate"
            action = "prompt_and_monitor"
            prompt_fast = True
        else:
            label = "no_concern"
            severity = "none"
            action = "monitor"
            prompt_fast = False

        return StrokeScreenResult(
            label=label,
            score=score,
            severity=severity,
            rationale=rationale,
            should_prompt_fast_exam=prompt_fast,
            recommended_action=action,
            details={
                "face_present": inp.face_present,
                "face_confidence": inp.face_confidence,
                "mouth_asymmetry": inp.mouth_asymmetry,
                "eye_asymmetry": inp.eye_asymmetry,
                "speech_prompt_completed": inp.speech_prompt_completed,
                "speech_slur_score": inp.speech_slur_score,
                "response_latency_sec": inp.response_latency_sec,
                "arm_drift_score": inp.arm_drift_score,
                "manual_trigger": inp.manual_trigger,
                "modalities": modalities,
            },
        )
