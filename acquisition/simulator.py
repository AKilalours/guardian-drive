from __future__ import annotations

"""Synthetic data simulator for Guardian Drive™ v4.0.

This repo is a *pipeline demo* (SQI gating → features → task models → fusion → policy).
The simulator produces deterministic multi-sensor windows for a handful of scenarios so
you can run the full stack on any machine without external datasets.

It is NOT a physiology model and should not be used for medical claims.
"""

import math
import time
from typing import Dict, Iterator, Optional

import numpy as np

from .models import (
    SensorFrame, TaskLabel, FS_ECG, FS_EDA, FS_RESP, FS_IMU,
)

# ---------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------
# Notes:
# - `label` is the ground-truth TaskLabel for the window (used in evaluation/tests).
# - Arrhythmia subtype is *not* stored in SensorFrame; it emerges from features.
# - Values are tuned so the rule-based baseline behaves predictably.

SCENARIO_PARAMS: Dict[str, dict] = {
    # Baseline
    "normal":       dict(hr=72,  rr_cv=0.03, resp_bpm=14, eda_level=3.5, posture=0.95,
                          crash=None, label=TaskLabel.NORMAL, belt=True, artifacts=0.0),

    # Arrhythmias
    "afib":         dict(hr=95,  rr_cv=0.30, resp_bpm=15, eda_level=3.8, posture=0.95,
                          crash=None, label=TaskLabel.ARRHYTHMIA, belt=True, artifacts=0.05),
    "tachycardia":  dict(hr=180, rr_cv=0.06, resp_bpm=18, eda_level=4.2, posture=0.95,
                          crash=None, label=TaskLabel.ARRHYTHMIA, belt=True, artifacts=0.05),
    "bradycardia":  dict(hr=35,  rr_cv=0.05, resp_bpm=10, eda_level=2.8, posture=0.95,
                          crash=None, label=TaskLabel.ARRHYTHMIA, belt=True, artifacts=0.05),

    # Fatigue / drowsiness
    "drowsy":       dict(hr=55,  rr_cv=0.02, resp_bpm=8,  eda_level=1.8, posture=0.25,
                          crash=None, label=TaskLabel.DROWSY, belt=True, artifacts=0.05),
    "fatigued":     dict(hr=54,  rr_cv=0.03, resp_bpm=11, eda_level=2.0, posture=0.45,
                          crash=None, label=TaskLabel.FATIGUED, belt=True, artifacts=0.05),
    "stressed":     dict(hr=92,  rr_cv=0.04, resp_bpm=18, eda_level=6.0, posture=0.90,
                          crash=None, label=TaskLabel.NORMAL, belt=True, artifacts=0.05),

    # Impairment (screening only): elevated alcohol score + irregular respiration.
    # NOTE: this is a synthetic proxy for demo/testing; real impairment requires real data.
    "impaired":     dict(hr=78,  rr_cv=0.05, resp_bpm=20, eda_level=4.5, posture=0.85,
                          crash=None, label=TaskLabel.NORMAL, belt=True, artifacts=0.10,
                          alcohol_mean=0.75, alcohol_std=0.08),

    # Crashes
    "crash_mild":   dict(hr=80,  rr_cv=0.03, resp_bpm=16, eda_level=3.5, posture=0.90,
                          crash="mild",   label=TaskLabel.CRASH_MILD, belt=True, artifacts=0.15),
    "crash_severe": dict(hr=85,  rr_cv=0.03, resp_bpm=16, eda_level=3.5, posture=0.90,
                          crash="severe", label=TaskLabel.CRASH_SEVERE, belt=True, artifacts=0.20),

    # Bad signal / belt off / motion
    "artifact":     dict(hr=75,  rr_cv=0.10, resp_bpm=14, eda_level=3.0, posture=0.90,
                          crash=None, label=TaskLabel.ARTIFACT, belt=False, artifacts=1.0),
}


def _stable_seed(s: str) -> int:
    """Stable seed across runs and Python versions."""
    import hashlib
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


class GuardianSimulator:
    """Deterministic synthetic multi-sensor generator.

    API:
      - window(t0, win_sec) -> SensorFrame
      - stream(win_sec, step_sec) -> iterator[SensorFrame]
    """

    def __init__(
        self,
        scenario: str,
        duration: float,
        subject_id: str = "sim_00",
        session_id: Optional[str] = None,
        inject_artifacts: bool = False,
    ):
        if scenario not in SCENARIO_PARAMS:
            raise ValueError(f"Unknown scenario: {scenario}. Use SCENARIO_PARAMS keys.")
        self.scenario = scenario
        self.duration = float(duration)
        self.subject_id = subject_id
        self.session_id = session_id or f"{scenario}_{int(time.time())}"
        self.inject_artifacts = bool(inject_artifacts)

        prm = SCENARIO_PARAMS[scenario]
        self._rng = np.random.default_rng(_stable_seed(f"{subject_id}:{scenario}:{self.session_id}"))
        # light variability between subjects
        self._hr_offset = float(self._rng.normal(0, 2.0))
        self._prm = prm

    def window(self, start_sec: float, win_sec: float) -> SensorFrame:
        prm = self._prm
        win_sec = float(win_sec)

        # per-window RNG keeps stream stable regardless of iteration order
        rng = np.random.default_rng(_stable_seed(f"{self.session_id}:{start_sec:.3f}:{win_sec:.3f}"))

        hr = max(25.0, prm["hr"] + self._hr_offset + float(rng.normal(0, 1.0)))
        rr_cv = float(prm["rr_cv"])
        artifacts = float(prm["artifacts"]) + (0.15 if self.inject_artifacts else 0.0)
        crash = prm["crash"]

        ecg = _gen_ecg_spike_train(hr_bpm=hr, win_sec=win_sec, rr_cv=rr_cv, artifacts=artifacts, rng=rng)
        eda = _gen_eda(level=float(prm["eda_level"]), win_sec=win_sec, artifacts=artifacts, rng=rng)
        resp = _gen_resp(rate_bpm=float(prm["resp_bpm"]), win_sec=win_sec, artifacts=artifacts, rng=rng)
        accel, gyro, belt_tension = _gen_imu(
            win_sec=win_sec,
            crash=crash,
            posture=float(prm["posture"]),
            artifacts=artifacts,
            belt_on=bool(prm["belt"]),
            rng=rng,
        )

        # occasional channel dropouts under high artifacts
        if artifacts >= 0.8:
            if rng.random() < 0.5: ecg = None
            if rng.random() < 0.5: eda = None
            if rng.random() < 0.4: resp = None

        alc_mu = float(prm.get("alcohol_mean", 0.0))
        alc_sd = float(prm.get("alcohol_std", 0.02))

        frame = SensorFrame(
            session_id=self.session_id,
            subject_id=self.subject_id,
            timestamp=time.monotonic(),
            window_sec=win_sec,
            label=prm["label"],
            ecg=ecg,
            eda=eda,
            respiration=resp,
            accel=accel,
            gyro=gyro,
            temperature=float(36.6 + rng.normal(0, 0.15)),
            alcohol=float(np.clip(rng.normal(alc_mu, alc_sd), 0.0, 1.0)),
            belt_tension=float(belt_tension),
        )
        return frame

    def stream(self, win: float, step: float) -> Iterator[SensorFrame]:
        win = float(win)
        step = float(step)
        if win <= 0 or step <= 0:
            raise ValueError("win and step must be positive")
        t = 0.0
        while t + win <= self.duration + 1e-9:
            yield self.window(t, win)
            t += step


# ---------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------

def _gen_ecg_spike_train(
    hr_bpm: float,
    win_sec: float,
    rr_cv: float,
    artifacts: float,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    n = int(FS_ECG * win_sec)
    if n <= 0:
        return None

    # baseline noise
    noise_std = 40.0 + 600.0 * float(np.clip(artifacts, 0, 1))
    x = rng.normal(0.0, noise_std, size=n)

    # RR intervals
    mean_rr = 60.0 / max(hr_bpm, 1e-6)
    t = 0.0
    spike_amp = 1100.0
    while t < win_sec:
        rr = max(0.25, float(rng.normal(mean_rr, rr_cv * mean_rr)))
        t += rr
        idx = int(t * FS_ECG)
        if 0 <= idx < n:
            x[idx] += spike_amp + float(rng.normal(0, 80.0))
            # QRS-ish width (3 samples)
            if idx + 1 < n: x[idx + 1] += spike_amp * 0.45
            if idx + 2 < n: x[idx + 2] += spike_amp * 0.20

    # artifact bursts
    if artifacts > 0.3 and rng.random() < 0.4:
        b0 = int(rng.integers(0, max(1, n - 200)))
        x[b0 : b0 + 200] += rng.normal(0.0, 1500.0 * artifacts, size=200)

    return x.astype(np.float32)


def _gen_eda(level: float, win_sec: float, artifacts: float, rng: np.random.Generator) -> Optional[np.ndarray]:
    n = int(FS_EDA * win_sec)
    if n <= 0:
        return None
    t = np.linspace(0, win_sec, n, endpoint=False)
    drift = 0.06 * np.sin(2 * math.pi * t / max(win_sec, 1e-6))
    noise = rng.normal(0.0, 0.10 + 0.8 * artifacts, size=n)
    x = level + drift + noise
    return x.astype(np.float32)


def _gen_resp(rate_bpm: float, win_sec: float, artifacts: float, rng: np.random.Generator) -> Optional[np.ndarray]:
    n = int(FS_RESP * win_sec)
    if n <= 0:
        return None
    t = np.linspace(0, win_sec, n, endpoint=False)
    f = max(rate_bpm / 60.0, 0.05)
    amp = 1.0 - 0.7 * float(np.clip(artifacts, 0, 1))
    x = amp * np.sin(2 * math.pi * f * t + float(rng.uniform(0, 2*math.pi)))
    x += rng.normal(0.0, 0.05 + 0.25 * artifacts, size=n)
    return x.astype(np.float32)


def _gen_imu(
    win_sec: float,
    crash: Optional[str],
    posture: float,
    artifacts: float,
    belt_on: bool,
    rng: np.random.Generator,
):
    n = int(FS_IMU * win_sec)
    if n <= 0:
        return None, None, 0.0

    # Baseline: near gravity on z-axis, small noise. Posture affects x tilt.
    tilt = float(np.clip(1.0 - posture, 0.0, 0.7))
    ax_mean = 0.6 * tilt
    az_mean = 1.0 - 0.2 * tilt

    accel = np.stack([
        rng.normal(ax_mean, 0.02 + 0.25 * artifacts, size=n),
        rng.normal(0.0,    0.02 + 0.25 * artifacts, size=n),
        rng.normal(az_mean,0.02 + 0.25 * artifacts, size=n),
    ], axis=1)

    gyro = rng.normal(0.0, 0.8 + 10.0 * artifacts, size=(n,3))

    # Crash spike
    belt_tension = 0.65 if belt_on else 0.05
    if crash is not None:
        idx = n // 2
        g_peak = 5.0 if crash == "mild" else 10.0
        for k in range(5):
            if 0 <= idx + k < n:
                accel[idx + k, 0] += g_peak
                accel[idx + k, 2] -= g_peak * 0.2
        belt_tension = 0.75 if crash == "mild" else 0.95

    return accel.astype(np.float32), gyro.astype(np.float32), float(belt_tension)
