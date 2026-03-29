from __future__ import annotations

"""
Guardian Drive — Task A: Arrhythmia Screening

Supported artifact formats
--------------------------
1) Synthetic/demo sklearn pipeline
   - plain sklearn object with predict_proba(...)
   - trained from models/train_task_a.py

2) Real PTB-XL artifact bundle
   - dict with keys:
       model, threshold, task, source, feature_spec
   - trained from models/train_task_a_real_ptbxl.py

Important
---------
- PTB-XL runtime inference requires an ECG waveform window to be present at runtime.
- If waveform data is unavailable, this screener returns an honest abstain unless
  GD_TASK_A_ALLOW_BASELINE_FALLBACK=1 is set.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np

from acquisition.models import FeatureBundle, ArrhythmiaResult, ArrhythmiaClass
from .model_utils import default_artifacts_dir, load_joblib


def _get(obj: Any, names, default=None):
    if obj is None:
        return default
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_1d_float_array(x):
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _resample_linear(x: np.ndarray, out_len: int) -> np.ndarray:
    if x.size == out_len:
        return x.astype(np.float32, copy=False)
    xp = np.linspace(0.0, 1.0, num=x.size, dtype=np.float32)
    xnew = np.linspace(0.0, 1.0, num=out_len, dtype=np.float32)
    return np.interp(xnew, xp, x.astype(np.float32, copy=False)).astype(np.float32)


class ArrhythmiaScreener:
    def __init__(self):
        self._allow_baseline_fallback = _env_bool("GD_TASK_A_ALLOW_BASELINE_FALLBACK", False)

        self._model_path = self._resolve_model_path()
        self._artifact = load_joblib(self._model_path)

        self._model = None
        self._threshold = 0.5
        self._artifact_kind = "none"
        self._source = "unknown"
        self._feature_spec = {}

        if isinstance(self._artifact, dict) and "model" in self._artifact:
            self._model = self._artifact.get("model")
            self._threshold = float(self._artifact.get("threshold", 0.5))
            self._artifact_kind = "ptbxl_bundle"
            self._source = str(self._artifact.get("source", "unknown"))
            self._feature_spec = dict(self._artifact.get("feature_spec", {}) or {})
        elif self._artifact is not None:
            self._model = self._artifact
            self._artifact_kind = "plain_pipeline"
            self._source = "synthetic_or_unknown"

    def _resolve_model_path(self) -> Path:
        env_path = os.getenv("GD_TASK_A_MODEL", "").strip()
        if env_path:
            return Path(env_path)

        art = default_artifacts_dir()
        preferred = [
            art / "task_a_model_real_ptbxl.joblib",
            art / "task_a_model_simulated.joblib",
            art / "task_a_model.joblib",
        ]
        for p in preferred:
            if p.exists():
                return p
        return preferred[0]

    # -------------------------------------------------------------------------
    # Synthetic/demo feature path
    # -------------------------------------------------------------------------

    def _feature_vector_synthetic(self, fb: FeatureBundle) -> np.ndarray:
        ecg = fb.ecg
        hr = float(_get(ecg, ["hr_bpm", "hr"], 0.0) or 0.0)
        rr_irr = float(_get(ecg, ["rr_irregularity", "rr_irr", "rr_irreg"], 0.0) or 0.0)
        rr_rmssd = float(_get(ecg, ["rr_rmssd", "hrv_rmssd"], 0.0) or 0.0)
        rr_sdnn = float(_get(ecg, ["rr_sdnn", "hrv_sdnn"], 0.0) or 0.0)
        p_frac = float(_get(ecg, ["p_wave_fraction", "p_frac", "p_wave_frac"], 0.0) or 0.0)
        return np.array([hr, rr_irr, rr_rmssd, rr_sdnn, p_frac], dtype=np.float32)

    # -------------------------------------------------------------------------
    # PTB-XL runtime waveform path
    # -------------------------------------------------------------------------

    def _extract_runtime_waveform(self, fb: FeatureBundle):
        """
        Try multiple possible storage locations for the runtime ECG waveform.

        This file alone does NOT magically make Task A runtime-real.
        Your runtime pipeline must actually attach waveform data somewhere
        accessible here. If not, this returns (None, None) and Task A abstains.
        """
        ecg = getattr(fb, "ecg", None)

        candidates = [
            _get(ecg, ["lead_ii", "lead2", "waveform", "signal", "samples", "window", "raw", "raw_signal"], None),
            _get(fb, ["ecg_signal", "ecg_samples", "raw_ecg", "ecg_window", "ecg_waveform"], None),
        ]

        for cand in candidates:
            arr = _as_1d_float_array(cand)
            if arr is not None and arr.size >= 64:
                fs = float(
                    _get(ecg, ["fs_hz", "fs", "sampling_rate_hz", "sampling_rate"], 100.0) or 100.0
                )
                return arr, fs

        return None, None

    def _feature_vector_ptbxl_runtime(self, fb: FeatureBundle) -> np.ndarray | None:
        x, fs = self._extract_runtime_waveform(fb)
        if x is None:
            return None

        # Trainer used a 10-second low-rate rhythm record.
        # Best effort: use last 10 seconds if enough data exists.
        target_secs = 10.0
        need = int(round(fs * target_secs))
        if need > 0 and x.size >= need:
            x = x[-need:]

        x = x.astype(np.float32, copy=False)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x - np.mean(x)

        std = float(np.std(x))
        if std < 1e-6:
            std = 1.0
        x = x / std

        # Exact same compact feature construction used in train_task_a_real_ptbxl.py
        x_250 = _resample_linear(x, 250)
        fft_mag = np.abs(np.fft.rfft(x_250))[:128].astype(np.float32)
        dx = np.diff(x_250)

        stats = np.array(
            [
                float(np.mean(x_250)),
                float(np.std(x_250)),
                float(np.min(x_250)),
                float(np.max(x_250)),
                float(np.percentile(x_250, 5)),
                float(np.percentile(x_250, 95)),
                float(np.mean(np.abs(dx))) if dx.size else 0.0,
                float(np.std(dx)) if dx.size else 0.0,
                float(np.max(np.abs(dx))) if dx.size else 0.0,
            ],
            dtype=np.float32,
        )

        feat = np.concatenate([x_250, fft_mag, stats], axis=0)
        return feat.astype(np.float32, copy=False)

    def _subclass_from_runtime_signals(self, fb: FeatureBundle, proba: float) -> tuple[ArrhythmiaClass, str]:
        ecg = fb.ecg
        hr = float(_get(ecg, ["hr_bpm", "hr"], 0.0) or 0.0)
        rr_irr = float(_get(ecg, ["rr_irregularity", "rr_irr", "rr_irreg"], 0.0) or 0.0)
        p_frac = float(_get(ecg, ["p_wave_fraction", "p_frac", "p_wave_frac"], 0.0) or 0.0)

        if proba < self._threshold:
            return ArrhythmiaClass.NORMAL, f"real_ptbxl: prob<{self._threshold:.3f}"

        if rr_irr >= 0.18 and p_frac <= 0.25:
            return ArrhythmiaClass.AFIB, "real_ptbxl: high RR irregularity + low P-wave fraction"
        if hr >= 110:
            return ArrhythmiaClass.TACHYCARDIA, "real_ptbxl: high HR"
        if hr <= 50:
            return ArrhythmiaClass.BRADYCARDIA, "real_ptbxl: low HR"

        # Your enum does not have OTHER. UNKNOWN is the only honest fallback.
        return ArrhythmiaClass.UNKNOWN, "real_ptbxl: abnormal rhythm probability elevated but subtype unresolved"

    def _predict_ptbxl_bundle(self, fb: FeatureBundle) -> ArrhythmiaResult | None:
        if self._model is None:
            return None

        x = self._feature_vector_ptbxl_runtime(fb)
        if x is None:
            return ArrhythmiaResult(
                cls=ArrhythmiaClass.NOISY,
                confidence=0.0,
                hr_bpm=float(_get(fb.ecg, ["hr_bpm", "hr"], 0.0) or 0.0),
                rr_irr=float(_get(fb.ecg, ["rr_irregularity", "rr_irr", "rr_irreg"], 0.0) or 0.0),
                p_frac=float(_get(fb.ecg, ["p_wave_fraction", "p_frac", "p_wave_frac"], 1.0) or 1.0),
                abstained=True,
                reason="real_ptbxl artifact loaded but runtime ECG waveform is unavailable/incompatible",
                features_used=[],
            )

        try:
            proba = float(self._model.predict_proba(x.reshape(1, -1))[0, 1])
            cls, reason = self._subclass_from_runtime_signals(fb, proba)

            return ArrhythmiaResult(
                cls=cls,
                confidence=proba if cls != ArrhythmiaClass.NORMAL else (1.0 - proba),
                hr_bpm=float(_get(fb.ecg, ["hr_bpm", "hr"], 0.0) or 0.0),
                rr_irr=float(_get(fb.ecg, ["rr_irregularity", "rr_irr", "rr_irreg"], 0.0) or 0.0),
                p_frac=float(_get(fb.ecg, ["p_wave_fraction", "p_frac", "p_wave_frac"], 1.0) or 1.0),
                abstained=False,
                reason=reason,
                features_used=["ecg_waveform_250", "fft_128", "stats_9"],
            )
        except Exception as e:
            return ArrhythmiaResult(
                cls=ArrhythmiaClass.NOISY,
                confidence=0.0,
                hr_bpm=float(_get(fb.ecg, ["hr_bpm", "hr"], 0.0) or 0.0),
                rr_irr=float(_get(fb.ecg, ["rr_irregularity", "rr_irr", "rr_irreg"], 0.0) or 0.0),
                p_frac=float(_get(fb.ecg, ["p_wave_fraction", "p_frac", "p_wave_frac"], 1.0) or 1.0),
                abstained=True,
                reason=f"real_ptbxl inference failed: {type(e).__name__}",
                features_used=["ecg_waveform_250", "fft_128", "stats_9"],
            )

    def _predict_plain_pipeline(self, fb: FeatureBundle) -> ArrhythmiaResult | None:
        if self._model is None:
            return None

        try:
            x = self._feature_vector_synthetic(fb).reshape(1, -1)
            proba = float(self._model.predict_proba(x)[0, 1])

            ecg = fb.ecg
            hr = float(_get(ecg, ["hr_bpm", "hr"], 0.0) or 0.0)
            rr_irr = float(_get(ecg, ["rr_irregularity", "rr_irr", "rr_irreg"], 0.0) or 0.0)
            p_frac = float(_get(ecg, ["p_wave_fraction", "p_frac", "p_wave_frac"], 1.0) or 1.0)

            if proba < 0.5:
                cls = ArrhythmiaClass.NORMAL
                reason = "synthetic: prob<0.5"
            elif rr_irr >= 0.18 and p_frac <= 0.25:
                cls = ArrhythmiaClass.AFIB
                reason = "synthetic: high RR irregularity + low P-wave fraction"
            elif hr >= 110:
                cls = ArrhythmiaClass.TACHYCARDIA
                reason = "synthetic: high HR"
            elif hr <= 50:
                cls = ArrhythmiaClass.BRADYCARDIA
                reason = "synthetic: low HR"
            else:
                cls = ArrhythmiaClass.UNKNOWN
                reason = "synthetic: abnormal rhythm probability elevated but subtype unresolved"

            return ArrhythmiaResult(
                cls=cls,
                confidence=proba if cls != ArrhythmiaClass.NORMAL else (1.0 - proba),
                hr_bpm=hr,
                rr_irr=rr_irr,
                p_frac=p_frac,
                abstained=False,
                reason=reason,
                features_used=["hr_bpm", "rr_irregularity", "rr_rmssd", "rr_sdnn", "p_wave_fraction"],
            )
        except Exception:
            return None

    def _predict_baseline(self, fb: FeatureBundle) -> ArrhythmiaResult:
        ecg = fb.ecg
        hr = float(_get(ecg, ["hr_bpm", "hr"], 0.0) or 0.0)
        rr_irr = float(_get(ecg, ["rr_irregularity", "rr_irr", "rr_irreg"], 0.0) or 0.0)
        p_frac = float(_get(ecg, ["p_wave_fraction", "p_frac", "p_wave_frac"], 1.0) or 1.0)

        if rr_irr >= 0.20 and p_frac <= 0.25:
            cls = ArrhythmiaClass.AFIB
            conf = 0.85
            reason = "baseline: irregular RR + low P-wave fraction"
        elif hr >= 120:
            cls = ArrhythmiaClass.TACHYCARDIA
            conf = 0.80
            reason = "baseline: HR>=120 bpm"
        elif hr <= 45:
            cls = ArrhythmiaClass.BRADYCARDIA
            conf = 0.80
            reason = "baseline: HR<=45 bpm"
        elif rr_irr >= 0.30:
            cls = ArrhythmiaClass.NOISY
            conf = 0.60
            reason = "baseline: extreme RR irregularity -> likely artifact"
        else:
            cls = ArrhythmiaClass.NORMAL
            conf = 0.90
            reason = "baseline: within normal limits"

        return ArrhythmiaResult(
            cls=cls,
            confidence=conf,
            hr_bpm=hr,
            rr_irr=rr_irr,
            p_frac=p_frac,
            abstained=False,
            reason=reason,
            features_used=["hr_bpm", "rr_irregularity", "p_wave_fraction"],
        )

    def predict(self, fb: FeatureBundle) -> ArrhythmiaResult:
        if fb.sqi.abstain:
            return ArrhythmiaResult(
                cls=ArrhythmiaClass.NOISY,
                confidence=0.0,
                hr_bpm=float(_get(fb.ecg, ["hr_bpm", "hr"], 0.0) or 0.0),
                rr_irr=float(_get(fb.ecg, ["rr_irregularity", "rr_irr", "rr_irreg"], 0.0) or 0.0),
                p_frac=float(_get(fb.ecg, ["p_wave_fraction", "p_frac", "p_wave_frac"], 1.0) or 1.0),
                abstained=True,
                reason="SQI abstain",
                features_used=[],
            )

        if self._artifact_kind == "ptbxl_bundle":
            pred = self._predict_ptbxl_bundle(fb)
            if pred is not None:
                if pred.abstained and self._allow_baseline_fallback:
                    return self._predict_baseline(fb)
                return pred

        if self._artifact_kind == "plain_pipeline":
            pred = self._predict_plain_pipeline(fb)
            if pred is not None:
                return pred

        return self._predict_baseline(fb)
