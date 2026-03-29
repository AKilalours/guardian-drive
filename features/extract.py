from __future__ import annotations

"""Feature extraction.

Converts raw SensorFrame windows into a typed FeatureBundle.

Key fixes:
- preserve raw ECG waveform for real runtime Task A models
- accept either legacy ECG or live seat ECG
- compute seat-ECG local quality and feed it back into SQI
- attach webcam metrics to FeatureBundle for Task B fusion
- avoid hidden dynamic attributes where possible
"""

import numpy as np
from scipy.signal import find_peaks

from acquisition.models import (
    SensorFrame,
    SQIState,
    FeatureBundle,
    FS_ECG,
    FS_RESP,
    FS_IMU,
)
from sqi.window_quality import compute_window_quality
from features.ecg_filter import detrend_mean
from features.rpeak_detect import detect_rpeaks
from features.hrv_live import rr_from_peak_times, compute_hrv
from features.resp_estimate import estimate_resp_from_ecg


def _safe_lin_slope(y: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 1e-9:
        return 0.0
    return float(((x * (y - y.mean())).sum()) / denom)


def _resp_rate_fft(x: np.ndarray, fs: float) -> float | None:
    if x.size < int(fs * 6):
        return None

    x = np.asarray(x, dtype=float)
    x = x - float(np.mean(x))
    w = np.hanning(x.size)
    y = np.fft.rfft(x * w)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    psd = (np.abs(y) ** 2)

    fmin, fmax = 0.06, 0.60  # 3.6–36 bpm
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return None

    f_peak = float(freqs[mask][np.argmax(psd[mask])])
    return float(f_peak * 60.0)


def _adaptive_ecg_peaks(x: np.ndarray, fs: float):
    """
    Rough, scale-free ECG peak detector.
    This is not clinical-grade. It is only for baseline HR / RR proxies.
    """
    if x.size < max(64, int(2 * fs)):
        return np.array([], dtype=int)

    xc = np.asarray(x, dtype=float)
    xc = xc - float(np.median(xc))

    xa = np.abs(xc)

    mad = float(np.median(np.abs(xa - np.median(xa)))) + 1e-6
    sigma = 1.4826 * mad
    p95 = float(np.percentile(xa, 95))
    thr = max(p95 * 0.55, sigma * 2.5, 1e-6)

    min_dist = max(1, int(0.25 * fs))  # ~240 bpm upper bound
    peaks, _ = find_peaks(xa, height=thr, distance=min_dist)
    return peaks


def _select_ecg_source(frame: SensorFrame):
    """Prefer live seat ECG when present; otherwise fall back to legacy ECG."""
    if frame.seat_ecg is not None and len(frame.seat_ecg) > 10:
        fs_hz = float(frame.seat_ecg_fs_hz or FS_ECG)
        meta = dict(frame.seat_ecg_meta or {})
        return np.asarray(frame.seat_ecg, dtype=np.float32).reshape(-1), fs_hz, "seat_ecg", meta

    if frame.ecg is not None and len(frame.ecg) > 10:
        return np.asarray(frame.ecg, dtype=np.float32).reshape(-1), float(FS_ECG), "legacy_ecg", {}

    return None, None, "", {}


def extract_features(frame: SensorFrame, sqi: SQIState, window_sec: float) -> FeatureBundle:
    fb = FeatureBundle(
        timestamp=frame.timestamp,
        window_sec=float(window_sec),
        session_id=frame.session_id,
        subject_id=frame.subject_id,
        label=frame.label,
        sqi=sqi,
    )

    fb.temperature = frame.temperature
    fb.alcohol = frame.alcohol
    fb.belt_tension = frame.belt_tension
    fb.webcam_metrics = dict(frame.webcam_metrics or {}) if frame.webcam_metrics else None

    # ---------------- ECG / Seat ECG ----------------
    ecg_x, ecg_fs, ecg_source, ecg_meta = _select_ecg_source(frame)

    if ecg_x is not None:
        ecg_x = np.nan_to_num(ecg_x, nan=0.0, posinf=0.0, neginf=0.0)

        # Preserve waveform for downstream runtime models.
        fb.ecg.samples = ecg_x.copy()
        fb.ecg.waveform = ecg_x.copy()
        fb.ecg.signal = ecg_x.copy()
        fb.ecg.lead_ii = ecg_x.copy()
        fb.ecg.fs_hz = float(ecg_fs)
        fb.ecg.sampling_rate_hz = float(ecg_fs)
        fb.ecg.source = ecg_source

        if ecg_source == "seat_ecg":
            fb.seat_ecg_status = dict(ecg_meta or {})

            quality = compute_window_quality(ecg_x.tolist())
            fb.sqi.seat_ecg_quality = float(quality["overall_score"])
            fb.sqi.seat_contact = bool(ecg_meta.get("contact_ok", quality["contact"]["contact_score"] >= 0.40))
            fb.sqi.seat_motion = float(ecg_meta.get("motion_score", quality["motion"]["motion_score"]))
            fb.sqi.ecg_quality = max(float(fb.sqi.ecg_quality), float(quality["overall_score"]))

            fb.ecg.details.update({
                "seat_quality": quality,
                "seat_meta": dict(ecg_meta or {}),
            })

        # Light detrend before peak finding.
        x_filt = np.asarray(detrend_mean(ecg_x.tolist()), dtype=np.float32)

        # Primary peak detector from your new scaffold.
        rp = detect_rpeaks(x_filt.tolist(), float(ecg_fs))
        peak_times = list(rp.get("peak_times_sec", []) or [])

        # Fallback to adaptive detector if scaffold detector is too weak.
        if len(peak_times) < 3:
            peaks = _adaptive_ecg_peaks(x_filt, float(ecg_fs))
            peak_times = [float(i) / float(ecg_fs) for i in peaks]

        rr = rr_from_peak_times(peak_times)
        hrv = compute_hrv(rr)

        if hrv["n_rr"] >= 2:
            fb.ecg.hr_bpm = float(hrv["hr_bpm"])
            fb.ecg.hrv_rmssd = float(hrv["rmssd"] * 1000.0)  # seconds -> ms
            fb.ecg.hrv_sdnn = float(hrv["sdnn"] * 1000.0)    # seconds -> ms

            rr_mean = max(float(np.mean(rr)), 1e-6)
            irr = float(np.clip(np.std(rr) / rr_mean, 0.0, 1.0))
            fb.ecg.rr_irregularity = irr
            fb.ecg.p_wave_fraction = float(np.clip(1.0 - 2.5 * irr, 0.15, 1.0))
            fb.ecg.qrs_duration_ms = float(90.0 + 40.0 * irr)

        fb.ecg.details.update({
            "source": ecg_source,
            "fs_hz": float(ecg_fs),
            "n_samples": int(len(ecg_x)),
            "n_peaks": int(len(peak_times)),
            "peak_times_sec": peak_times[:32],
        })

        # Optional placeholder respiration estimate from ECG only if no direct resp exists.
        if (frame.respiration is None or len(frame.respiration) <= 10) and ecg_source == "seat_ecg":
            resp_est = estimate_resp_from_ecg(x_filt.tolist(), float(ecg_fs))
            if float(resp_est.get("confidence", 0.0)) > 0.0:
                fb.resp.rate_bpm = float(resp_est.get("resp_rate_bpm", 0.0) or 0.0)
                fb.resp.irregularity = float(np.clip(1.0 - float(resp_est.get("confidence", 0.0)), 0.0, 1.0))

    # ---------------- EDA ----------------
    if sqi.eda_usable and frame.eda is not None and len(frame.eda) > 10:
        x = np.asarray(frame.eda, dtype=float)
        fb.eda.scl_mean = float(np.mean(x))
        fb.eda.scl_slope = _safe_lin_slope(x)

        dx = np.diff(x)
        if dx.size:
            scr = np.sum(dx > (np.mean(dx) + 2.5 * np.std(dx)))
        else:
            scr = 0
        fb.eda.scr_rate_per_min = float(scr / max(window_sec, 1e-6) * 60.0)
        fb.eda.scr_amplitude = float(np.clip(np.std(x) * 1.2, 0.0, 5.0))

    # ---------------- Respiration ----------------
    if sqi.resp_usable and frame.respiration is not None and len(frame.respiration) > 10:
        x = np.asarray(frame.respiration, dtype=float)

        rate = _resp_rate_fft(x, FS_RESP)
        fb.resp.rate_bpm = float(rate) if rate is not None else None

        x0 = x - float(np.mean(x))
        spec = np.abs(np.fft.rfft(x0 * np.hanning(x0.size))) ** 2
        if spec.size > 1:
            peak = float(np.max(spec[1:]))
            total = float(np.sum(spec[1:])) + 1e-9
            concentration = peak / total
            fb.resp.irregularity = float(np.clip(1.0 - concentration * 3.0, 0.0, 1.0))

        fb.resp.amplitude_mean = float(np.mean(np.abs(x)))
        fb.resp.shallow_flag = bool(fb.resp.amplitude_mean < 0.35)

    # ---------------- IMU ----------------
    if frame.accel is not None and len(frame.accel) > 0:
        a = np.asarray(frame.accel, dtype=float)
        if a.ndim == 2 and a.shape[1] == 3:
            mag = np.sqrt((a * a).sum(axis=1))

            fb.imu.accel_rms = float(np.sqrt(np.mean((mag - np.mean(mag)) ** 2)))

            dt = 1.0 / FS_IMU
            da = np.diff(a, axis=0) / dt
            jerk = np.sqrt((da * da).sum(axis=1)) if da.size else np.array([0.0])
            fb.imu.jerk_peak = float(np.max(jerk))

            g_peak = float(np.max(mag) - 1.0)
            fb.imu.crash_g_peak = float(max(0.0, g_peak))
            fb.imu.crash_jerk = fb.imu.jerk_peak
            fb.imu.crash_flag = bool(fb.imu.crash_g_peak >= 4.0)

            ax_mean = float(np.mean(a[:, 0]))
            fb.imu.posture_score = float(np.clip(1.0 - abs(ax_mean) / 0.8, 0.0, 1.0))

    return fb
