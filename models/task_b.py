from __future__ import annotations

"""
Guardian Drive — Task B: Drowsiness / Fatigue Risk Screening

Runtime paths:
1) Optional learned physio model (if configured)
2) Physio rule baseline
3) Real webcam heuristic path
4) Webcam + physio fusion

Honest limits:
- This is not clinically validated.
- This is a real runtime signal path when webcam metrics are present.
- It does not make stroke detection real.
"""

import numpy as np

from acquisition.models import FeatureBundle, DrowsinessResult
from learned.infer_task_b import TaskBInference


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _first_float(*values, default=None):
    for v in values:
        out = _safe_float(v, None)
        if out is not None:
            return out
    return default


class DrowsinessScreener:
    def __init__(self):
        self._learned = TaskBInference()

    def _feature_vector(self, fb: FeatureBundle) -> np.ndarray:
        return np.array(
            [
                float(fb.ecg.hr_bpm or 0.0),
                float(fb.ecg.hrv_rmssd or 0.0),
                float(fb.ecg.hrv_sdnn or 0.0),
                float(fb.resp.rate_bpm or 0.0),
                float(fb.resp.irregularity or 0.0),
                float(fb.eda.scl_mean or 0.0),
                float(fb.imu.posture_score or 0.0),
                float(fb.sqi.motion_level),
                float(fb.sqi.ecg_quality),
                float(fb.sqi.eda_contact),
                float(fb.sqi.resp_quality),
                float(fb.sqi.belt_quality),
            ],
            dtype=np.float32,
        )

    def _get_webcam_metrics(self, fb: FeatureBundle):
        wm = getattr(fb, "webcam_metrics", None)
        if wm is None:
            return None

        if hasattr(wm, "to_dict"):
            try:
                wm = wm.to_dict()
            except Exception:
                return None

        if not isinstance(wm, dict):
            return None

        if not bool(wm.get("available", False)):
            return None

        return wm

    def _predict_from_webcam(self, fb: FeatureBundle) -> DrowsinessResult | None:
        wm = self._get_webcam_metrics(fb)
        if wm is None:
            return None

        face_detected = bool(wm.get("face_detected", False))
        if not face_detected:
            return None

        provided_score = _first_float(wm.get("drowsy_score"), default=None)
        ear = _first_float(wm.get("ear"), default=None)
        perclos = _first_float(wm.get("perclos_30s"), wm.get("perclos"), default=None)
        blink_rate = _first_float(wm.get("blink_rate_30s"), default=None)
        mouth_ratio = _first_float(wm.get("mouth_open_ratio"), default=None)
        yawns = _first_float(wm.get("yawn_events_30s"), wm.get("yawns"), default=0.0)
        eyes_closed = wm.get("eyes_closed", None)
        note = str(wm.get("note", "") or "")

        cues: dict[str, float] = {}
        weights: dict[str, float] = {}

        if provided_score is not None:
            cues["webcam_model"] = _clip01(provided_score)
            weights["webcam_model"] = 2.2

        if perclos is not None:
            # 0.10 = low concern, 0.40+ = strong concern
            cues["perclos"] = _clip01((perclos - 0.10) / 0.30)
            weights["perclos"] = 2.7

        if ear is not None:
            # Lower EAR means greater likelihood of eye closure
            cues["ear"] = _clip01((0.24 - ear) / 0.07)
            weights["ear"] = 1.7

        if eyes_closed is True:
            cues["eyes_closed"] = 0.85
            weights["eyes_closed"] = 0.9

        if mouth_ratio is not None:
            cues["mouth_ratio"] = _clip01((mouth_ratio - 0.20) / 0.18)
            weights["mouth_ratio"] = 0.8

        if yawns is not None and yawns > 0:
            cues["yawns"] = _clip01(float(yawns) / 3.0)
            weights["yawns"] = 1.4

        # Blink rate is noisy; use lightly.
        if blink_rate is not None:
            if blink_rate < 6.0:
                cues["blink_rate"] = _clip01((6.0 - blink_rate) / 6.0)
                weights["blink_rate"] = 0.4
            elif blink_rate > 35.0:
                cues["blink_rate"] = _clip01((blink_rate - 35.0) / 20.0)
                weights["blink_rate"] = 0.3

        if not cues:
            out = DrowsinessResult(
                score=0.0,
                confidence=0.25,
                abstained=False,
                reason=f"webcam: face detected but no usable cues ({note or 'no metrics'})",
            )
            setattr(out, "features_used", ["webcam_face_only"])
            return out

        total_w = sum(weights.values())
        score = sum(cues[k] * weights[k] for k in cues) / max(total_w, 1e-9)

        conf = min(0.95, 0.45 + 0.07 * len(cues))
        if perclos is not None and perclos >= 0.30:
            conf = max(conf, 0.78)
        if perclos is not None and perclos >= 0.40:
            conf = max(conf, 0.86)
        if provided_score is not None:
            conf = max(conf, 0.75)

        out = DrowsinessResult(
            score=float(_clip01(score)),
            confidence=float(conf),
            abstained=False,
            reason="webcam: " + "; ".join(f"{k}={cues[k]:.2f}" for k in sorted(cues.keys())),
        )
        setattr(out, "features_used", list(sorted(cues.keys())))
        return out

    def _predict_from_physio(self, fb: FeatureBundle) -> DrowsinessResult | None:
        # If physio is globally unusable, webcam may still work.
        if fb.sqi.abstain:
            return None

        p = self._learned.predict(self._feature_vector(fb))
        if p is not None:
            out = DrowsinessResult(
                abstained=False,
                score=float(p),
                confidence=float(fb.sqi.overall_confidence),
                reason="learned_physio: score from configured weights",
            )
            setattr(
                out,
                "features_used",
                [
                    "hr",
                    "hrv_rmssd",
                    "hrv_sdnn",
                    "resp_rate",
                    "resp_irr",
                    "eda_scl_mean",
                    "posture",
                    "motion",
                    "sqi_ecg",
                    "sqi_eda",
                    "sqi_resp",
                    "belt",
                ],
            )
            return out

        r = DrowsinessResult()
        r.abstained = False
        contributions: dict[str, tuple[float, float]] = {}

        if fb.sqi.ecg_usable and fb.ecg.hr_bpm is not None and fb.ecg.hr_bpm <= 65.0:
            s = _clip01((65.0 - fb.ecg.hr_bpm) / 12.0)
            r.hr_contrib = s
            contributions["hr"] = (s, 1.3)

        if fb.sqi.resp_usable and fb.resp.rate_bpm is not None and fb.resp.rate_bpm <= 12.0:
            s = _clip01((12.0 - fb.resp.rate_bpm) / 5.0)
            if fb.resp.shallow_flag:
                s = _clip01(s + 0.15)
            r.resp_contrib = s
            contributions["resp"] = (s, 1.2)

        if fb.sqi.eda_usable and fb.eda.scl_mean is not None and fb.eda.scl_mean <= 2.8:
            s = _clip01((2.8 - fb.eda.scl_mean) / 1.4)
            r.eda_contrib = s
            contributions["eda"] = (s, 1.0)

        if fb.imu.posture_score <= 0.75:
            s = _clip01((0.75 - fb.imu.posture_score) / 0.45)
            r.imu_contrib = s
            contributions["posture"] = (s, 1.1)

        if not contributions:
            out = DrowsinessResult(
                abstained=False,
                score=0.0,
                confidence=float(fb.sqi.overall_confidence) * 0.5,
                reason="physio: no fatigue signals above thresholds",
            )
            setattr(out, "features_used", [])
            return out

        total_w = sum(w for _, w in contributions.values())
        total_s = sum(s * w for s, w in contributions.values())
        r.score = float(_clip01(total_s / total_w))
        r.confidence = float(fb.sqi.overall_confidence * min(len(contributions) / 3.0, 1.0))
        r.reason = "physio: " + "; ".join(f"{k}={s:.2f}" for k, (s, _) in contributions.items())
        setattr(r, "features_used", list(contributions.keys()))
        return r

    def _fuse(
        self,
        physio: DrowsinessResult | None,
        webcam: DrowsinessResult | None,
    ) -> DrowsinessResult | None:
        if physio is None and webcam is None:
            return None

        if physio is None:
            webcam.reason = f"{webcam.reason}; mode=webcam_only"
            return webcam

        if webcam is None:
            physio.reason = f"{physio.reason}; mode=physio_only"
            return physio

        webcam_w = 0.65
        physio_w = 0.35

        if webcam.confidence < 0.55:
            webcam_w = 0.50
            physio_w = 0.50

        score = _clip01(webcam_w * webcam.score + physio_w * physio.score)

        # Agreement bonus
        if webcam.score >= 0.65 and physio.score >= 0.40:
            score = _clip01(score + 0.08)

        conf = _clip01(max(webcam.confidence, physio.confidence))
        if webcam.score >= 0.75 and physio.score >= 0.40:
            conf = max(conf, 0.88)

        out = DrowsinessResult(
            abstained=False,
            score=float(score),
            confidence=float(conf),
            reason=(
                f"fusion: webcam={webcam.score:.2f} (conf={webcam.confidence:.2f}) + "
                f"physio={physio.score:.2f} (conf={physio.confidence:.2f})"
            ),
        )
        out.hr_contrib = float(getattr(physio, "hr_contrib", 0.0))
        out.resp_contrib = float(getattr(physio, "resp_contrib", 0.0))
        out.eda_contrib = float(getattr(physio, "eda_contrib", 0.0))
        out.imu_contrib = float(getattr(physio, "imu_contrib", 0.0))

        setattr(
            out,
            "features_used",
            list(
                dict.fromkeys(
                    list(getattr(webcam, "features_used", []))
                    + list(getattr(physio, "features_used", []))
                )
            ),
        )
        return out

    def predict(self, fb: FeatureBundle) -> DrowsinessResult:
        webcam = self._predict_from_webcam(fb)
        physio = self._predict_from_physio(fb)

        fused = self._fuse(physio=physio, webcam=webcam)
        if fused is not None:
            return fused

        if fb.sqi.abstain:
            return DrowsinessResult(
                score=0.0,
                confidence=0.0,
                abstained=True,
                reason=f"Abstain: {fb.sqi.summary()} and no webcam metrics",
            )

        return DrowsinessResult(
            score=0.0,
            confidence=0.0,
            abstained=True,
            reason="Task B unavailable: no usable physio or webcam inputs",
        )
