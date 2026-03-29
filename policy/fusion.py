from __future__ import annotations

"""
Guardian Drive — Task fusion

What this does now:
- Task A: real PTB-XL runtime if waveform-compatible ECG exists.
- Task B: uses models.task_b as the authoritative drowsiness scorer.
- Task C: can be disabled, baseline, or honest no-op for unimplemented "real" mode.

What this does NOT do:
- It does not fake a real crash model.
- It does not double-count webcam evidence.
- It does not pretend your old proxy-CNN Task B image model is wired into runtime if it is not.
"""

import os
from typing import Any, Dict, Optional

from acquisition.models import (
    FeatureBundle,
    RiskState,
    DriverState,
    AlertLevel,
    ArrhythmiaClass,
    CrashSeverity,
    DrowsinessResult,
    CrashResult,
)
from models.task_a import ArrhythmiaScreener
from models.task_b import DrowsinessScreener
from models.task_c import CrashDetector


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _safe_float(x, default=None):
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


class FusionEngine:
    def __init__(self):
        self.enable_task_a = _env_bool("GD_ENABLE_TASK_A", True)
        self.enable_task_b = _env_bool("GD_ENABLE_TASK_B", True)
        self.enable_task_c = _env_bool("GD_ENABLE_TASK_C", True)

        self.task_a_mode = _env_str("GD_TASK_A_RUNTIME_MODE", "auto")
        self.task_b_mode = _env_str("GD_TASK_B_RUNTIME_MODE", "auto")
        self.task_c_mode = _env_str("GD_TASK_C_RUNTIME_MODE", "auto")

        self._a = ArrhythmiaScreener() if self.enable_task_a else None
        self._b = DrowsinessScreener() if self.enable_task_b else None
        self._c = CrashDetector() if self.enable_task_c else None

    def status_dict(self) -> dict:
        a = {
            "enabled": bool(self.enable_task_a),
            "requested_mode": self.task_a_mode,
            "artifact_path": str(getattr(self._a, "_model_path", "")) if self._a else None,
            "artifact_kind": getattr(self._a, "_artifact_kind", "disabled") if self._a else "disabled",
            "source": getattr(self._a, "_source", "disabled") if self._a else "disabled",
            "note": (
                "Task A uses PTB-XL runtime path only when ECG waveform is attached to FeatureBundle/ECGFeatures."
                if self._a else
                "Task A disabled."
            ),
        }

        b = {
            "enabled": bool(self.enable_task_b),
            "requested_mode": self.task_b_mode,
            "runtime_integrated": bool(self.enable_task_b),
            "note": (
                "Task B runtime is driven by models.task_b using webcam, physio, or both depending on runtime availability."
                if self.enable_task_b else
                "Task B disabled."
            ),
        }

        c_mode = self.task_c_mode.strip().lower()
        c_real_requested = c_mode in {"real", "real_crash", "dataset_real"}

        c = {
            "enabled": bool(self.enable_task_c),
            "requested_mode": self.task_c_mode,
            "artifact_path": str(getattr(self._c, "_model_path", "")) if self._c else None,
            "note": (
                "Task C real mode requested, but current runtime/task_c.py is still baseline/simulated."
                if (self.enable_task_c and c_real_requested) else
                "Task C runtime uses current models.task_c implementation."
                if self.enable_task_c else
                "Task C disabled."
            ),
        }

        return {
            "task_a": a,
            "task_b": b,
            "task_c": c,
            "claim_guardrail": {
                "full_system_real": False,
                "task_a_real_training": bool(a["source"] == "PTB-XL"),
                "task_b_runtime_live": bool(self.enable_task_b),
                "task_c_fully_real": False,
                "medical_grade": False,
            },
        }

    def _predict_task_a(self, fb: FeatureBundle):
        if not self.enable_task_a or self._a is None:
            return None

        out = self._a.predict(fb)
        if out is not None:
            out.details = dict(getattr(out, "details", {}) or {})
            out.details.update({
                "ecg_source": str(getattr(fb.ecg, "source", "") or ""),
                "ecg_fs_hz": float(getattr(fb.ecg, "fs_hz", 0.0) or 0.0),
                "ecg_samples": 0 if getattr(fb.ecg, "samples", None) is None else int(len(fb.ecg.samples)),
                "sqi_overall": float(fb.sqi.overall_confidence),
                "seat_ecg_quality": float(getattr(fb.sqi, "seat_ecg_quality", 0.0) or 0.0),
            })
        return out

    def _predict_task_b_physio(self, fb: FeatureBundle) -> Optional[DrowsinessResult]:
        if not self.enable_task_b or self._b is None:
            return None
        try:
            if hasattr(self._b, "_predict_from_physio"):
                return self._b._predict_from_physio(fb)
            return self._b.predict(fb)
        except Exception as e:
            return DrowsinessResult(
                score=0.0,
                confidence=0.0,
                abstained=True,
                reason=f"Task B physiologic scorer failed: {e}",
            )

    def _predict_task_b_webcam(self, fb: FeatureBundle) -> Optional[DrowsinessResult]:
        if not self.enable_task_b or self._b is None:
            return None
        try:
            if hasattr(self._b, "_predict_from_webcam"):
                return self._b._predict_from_webcam(fb)
            return None
        except Exception as e:
            return DrowsinessResult(
                score=0.0,
                confidence=0.0,
                abstained=True,
                reason=f"Task B webcam scorer failed: {e}",
            )

    def _fuse_task_b(self, physio, webcam):
        if self._b is not None and hasattr(self._b, "_fuse"):
            return self._b._fuse(physio=physio, webcam=webcam)

        if physio is None and webcam is None:
            return None
        if physio is None:
            return webcam
        if webcam is None:
            return physio
        if physio.abstained and not webcam.abstained:
            return webcam
        if webcam.abstained and not physio.abstained:
            return physio

        return DrowsinessResult(
            score=_clamp01((float(physio.score) + float(webcam.score)) / 2.0),
            confidence=_clamp01(max(float(physio.confidence), float(webcam.confidence))),
            abstained=bool(physio.abstained and webcam.abstained),
            reason=f"fallback_fuse: physio={physio.score:.2f}, webcam={webcam.score:.2f}",
        )

    def _attach_drowsy_details(self, fb: FeatureBundle, out: Optional[DrowsinessResult], mode_used: str) -> Optional[DrowsinessResult]:
        if out is None:
            return None

        wm: Dict[str, Any] = dict(fb.webcam_metrics or {})
        yawn_count = _safe_float(wm.get("yawn_count_30s", wm.get("yawn_events_30s")), 0.0) or 0.0
        yawn_ratio = _safe_float(wm.get("yawn_ratio"), 0.0) or 0.0
        yawn_score = _clamp01(max(yawn_count / 3.0, yawn_ratio * 2.5))

        out.details = dict(getattr(out, "details", {}) or {})
        out.details.update({
            "mode_used": mode_used,
            "webcam_available": bool(wm.get("available", False)),
            "webcam_score": float(_safe_float(wm.get("drowsy_score"), 0.0) or 0.0),
            "vision_score": float(_safe_float(wm.get("drowsy_score"), 0.0) or 0.0),
            "perclos": float(_safe_float(wm.get("perclos_30s", wm.get("perclos")), 0.0) or 0.0),
            "blink_rate": float(_safe_float(wm.get("blink_rate_30s"), 0.0) or 0.0),
            "ear": float(_safe_float(wm.get("ear"), 0.0) or 0.0),
            "eyes_closed": bool(wm.get("eyes_closed", False)),
            "eyes_closed_sec": float(_safe_float(wm.get("eyes_closed_sec", wm.get("closure_sec")), 0.0) or 0.0),
            "yawn_count": float(yawn_count),
            "yawn_score": float(yawn_score),
            "mouth_open": bool(wm.get("mouth_open", False)),
            "posture_score": float(getattr(fb.imu, "posture_score", 0.0) or 0.0),
            "hr_bpm": float(getattr(fb.ecg, "hr_bpm", 0.0) or 0.0),
            "rmssd_ms": float(getattr(fb.ecg, "hrv_rmssd", 0.0) or 0.0),
            "sdnn_ms": float(getattr(fb.ecg, "hrv_sdnn", 0.0) or 0.0),
            "hr_contrib": float(getattr(out, "hr_contrib", 0.0) or 0.0),
            "hrv_contrib": float(getattr(out, "hrv_contrib", 0.0) or 0.0),
            "resp_contrib": float(getattr(out, "resp_contrib", 0.0) or 0.0),
            "eda_contrib": float(getattr(out, "eda_contrib", 0.0) or 0.0),
            "imu_contrib": float(getattr(out, "imu_contrib", 0.0) or 0.0),
            "sqi_overall": float(fb.sqi.overall_confidence),
            "seat_ecg_quality": float(getattr(fb.sqi, "seat_ecg_quality", 0.0) or 0.0),
        })
        return out

    def _predict_task_b(self, fb: FeatureBundle):
        if not self.enable_task_b or self._b is None:
            return None

        mode = self.task_b_mode.strip().lower()

        if mode in {"disabled", "off", "none"}:
            return None

        if mode in {"proxy", "proxy_subjectsplit", "proxy_cnn", "cnn", "vision_proxy"}:
            return DrowsinessResult(
                score=0.0,
                confidence=0.0,
                abstained=True,
                reason=(
                    f"Requested Task B mode '{self.task_b_mode}' is not integrated into runtime. "
                    "This is not your extracted-frame CNN inference path."
                ),
                details={"mode_used": "unsupported_proxy"},
            )

        if mode in {"webcam", "webcam_only"}:
            out = self._predict_task_b_webcam(fb)
            if out is None:
                out = DrowsinessResult(
                    score=0.0,
                    confidence=0.0,
                    abstained=True,
                    reason="Webcam-only Task B requested but webcam metrics are unavailable.",
                )
            return self._attach_drowsy_details(fb, out, "webcam_only")

        if mode in {"physio", "physio_only"}:
            out = self._predict_task_b_physio(fb)
            return self._attach_drowsy_details(fb, out, "physio_only")

        if mode in {"fusion", "webcam_physio_fusion"}:
            physio = self._predict_task_b_physio(fb)
            webcam = self._predict_task_b_webcam(fb)
            out = self._fuse_task_b(physio, webcam)
            return self._attach_drowsy_details(fb, out, "explicit_fusion")

        # auto = let models.task_b decide based on actual available runtime inputs
        out = self._b.predict(fb)
        return self._attach_drowsy_details(fb, out, "auto")

    def _predict_task_c(self, fb: FeatureBundle):
        if not self.enable_task_c or self._c is None:
            return None

        mode = self.task_c_mode.strip().lower()

        if mode in {"disabled", "off", "none"}:
            return None

        if mode in {"real", "real_crash", "dataset_real"}:
            return CrashResult(
                detected=False,
                severity=CrashSeverity.NONE,
                confidence=0.0,
                latency_ms=0.0,
                g_peak=0.0,
                jerk_peak=0.0,
                belt_tension=0.0,
                belt_corroborated=False,
                reason="Requested real Task C mode, but current runtime/task_c.py is still baseline/simulated.",
                details={"mode_used": "real_requested_but_unimplemented"},
            )

        out = self._c.predict(fb)
        if out is not None:
            out.details = dict(getattr(out, "details", {}) or {})
            out.details.update({
                "imu_crash_g_peak": float(getattr(fb.imu, "crash_g_peak", 0.0) or 0.0),
                "imu_crash_jerk": float(getattr(fb.imu, "crash_jerk", 0.0) or 0.0),
                "belt_tension": float(fb.belt_tension or 0.0),
            })
        return out

    def run(self, fb: FeatureBundle) -> RiskState:
        rs = RiskState(
            timestamp=fb.timestamp,
            session_id=fb.session_id,
            overall_conf=fb.sqi.overall_confidence,
            abstained=fb.sqi.abstain,
        )

        rs.arrhythmia = self._predict_task_a(fb)
        rs.drowsiness = self._predict_task_b(fb)
        rs.crash = self._predict_task_c(fb)

        if rs.crash and rs.crash.detected and rs.crash.severity == CrashSeverity.SEVERE:
            rs.driver_state = DriverState.ESCALATE
            rs.alert_level = AlertLevel.ESCALATE

        elif (
            rs.arrhythmia
            and not rs.arrhythmia.abstained
            and rs.arrhythmia.cls not in {ArrhythmiaClass.NORMAL, ArrhythmiaClass.NOISY, ArrhythmiaClass.UNKNOWN}
        ):
            rs.driver_state = DriverState.AT_RISK
            rs.alert_level = AlertLevel.CAUTION

        elif rs.drowsiness and not rs.drowsiness.abstained and rs.drowsiness.score >= 0.55:
            rs.driver_state = DriverState.AT_RISK
            rs.alert_level = AlertLevel.CAUTION

        elif fb.sqi.abstain:
            rs.driver_state = DriverState.NORMAL
            rs.alert_level = AlertLevel.INACTIVE

        else:
            rs.driver_state = DriverState.NORMAL
            rs.alert_level = AlertLevel.NOMINAL

        return rs
