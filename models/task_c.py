from __future__ import annotations

"""Guardian Drive v4.1 — Task C: Crash Detection (IMU)

Two-mode:
1) baseline rule detector
2) optional learned sklearn model if artifacts exist

Env (optional):
  GD_TASK_C_MODEL=artifacts/task_c_model.joblib
"""

import os
import time
import numpy as np
from pathlib import Path

from acquisition.models import FeatureBundle, CrashResult, CrashSeverity
from .model_utils import default_artifacts_dir, load_joblib


def _get(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default


class CrashDetector:
    def __init__(self):
        model_path = os.getenv("GD_TASK_C_MODEL", "").strip()
        if not model_path:
            model_path = str(default_artifacts_dir() / "task_c_model.joblib")
        self._model_path = Path(model_path)
        self._model = load_joblib(self._model_path)

    def _x(self, fb: FeatureBundle) -> np.ndarray:
        imu = fb.imu
        g = float(_get(imu, ["crash_g_peak", "g_peak", "g"], 0.0) or 0.0)
        j = float(_get(imu, ["crash_jerk", "jerk"], 0.0) or 0.0)
        m = float(_get(imu, ["motion_rms", "motion"], 0.0) or 0.0)
        return np.array([g, j, m], dtype=np.float32)

    def _predict_learned(self, fb: FeatureBundle) -> CrashResult | None:
        if self._model is None:
            return None
        try:
            x = self._x(fb).reshape(1, -1)
            proba = float(self._model.predict_proba(x)[0, 1])
            detected = proba >= 0.5
            g = float(_get(fb.imu, ["crash_g_peak", "g_peak", "g"], 0.0) or 0.0)
            sev = CrashSeverity.SEVERE if g >= 7.5 else CrashSeverity.MILD if g >= 4.0 else CrashSeverity.NONE
            return CrashResult(
                detected=detected,
                severity=sev if detected else CrashSeverity.NONE,
                confidence=proba if detected else 1.0 - proba,
                g_peak=g,
                latency_ms=0.0,
                reason=("learned: proba>=0.5" if detected else "learned: proba<0.5"),
            )
        except Exception:
            return None

    def predict(self, fb: FeatureBundle) -> CrashResult:
        t0 = time.perf_counter()
        imu = fb.imu

        # baseline flags if present
        g_peak = float(_get(imu, ["crash_g_peak", "g_peak", "g"], 0.0) or 0.0)
        jerk   = float(_get(imu, ["crash_jerk", "jerk"], 0.0) or 0.0)
        flag   = bool(_get(imu, ["crash_flag", "flag"], False))

        # learned first if available
        learned = self._predict_learned(fb)
        if learned is not None:
            learned.latency_ms = (time.perf_counter() - t0) * 1000.0
            return learned

        # baseline thresholds
        detected = flag or (g_peak >= 4.0 and jerk >= 10.0)
        if not detected:
            return CrashResult(
                detected=False,
                severity=CrashSeverity.NONE,
                confidence=0.95,
                g_peak=g_peak,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                reason="baseline: below thresholds",
            )

        severity = CrashSeverity.SEVERE if g_peak >= 7.5 else CrashSeverity.MILD
        conf = 0.98 if severity == CrashSeverity.SEVERE else 0.90
        return CrashResult(
            detected=True,
            severity=severity,
            confidence=conf,
            g_peak=g_peak,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            reason="baseline: thresholds triggered",
        )
