from __future__ import annotations

"""
Stream real SensorFrame windows from the WESAD dataset (UCI).

This provides real physiological windows for Guardian Drive runtime demos.
It is useful for engineering validation, not for medical claims.

Expected layout:
  <WESAD_ROOT>/
    S2/S2.pkl
    S3/S3.pkl
    ...

Server env:
  GD_WESAD_ROOT=/path/to/WESAD
  GD_WESAD_SUBJECT=S2
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from scipy.signal import resample_poly

from acquisition.models import (
    SensorFrame,
    TaskLabel,
    FS_ECG,
    FS_EDA,
    FS_RESP,
    FS_IMU,
    FS_TEMP,
)


# Standard WESAD chest sampling rates
FS_WESAD_ECG = 700
FS_WESAD_RESP = 700
FS_WESAD_ACC = 700
FS_WESAD_EDA = 4
FS_WESAD_TEMP = 4


def _as_path(x) -> Path:
    return x if isinstance(x, Path) else Path(x)


def _find_subject_pkl(wesad_root, subject: str) -> Path:
    wesad_root = _as_path(wesad_root)

    p1 = wesad_root / subject / f"{subject}.pkl"
    if p1.exists():
        return p1

    p2 = wesad_root / f"{subject}.pkl"
    if p2.exists():
        return p2

    raise FileNotFoundError(
        f"WESAD subject pickle not found for {subject}. "
        f"Tried: {p1} and {p2}"
    )


def _pick_present(mapping, *keys):
    if mapping is None:
        return None
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return None


def _to_1d(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.reshape(-1)


def _to_2d_acc(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 2 and arr.shape[0] == 3:
        return arr.T.astype(np.float32, copy=False)
    return arr.reshape(-1, 3).astype(np.float32, copy=False)


def _resample_1d(x: Optional[np.ndarray], fs_in: int, fs_out: int) -> Optional[np.ndarray]:
    if x is None:
        return None

    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if fs_in == fs_out:
        return arr.astype(np.float32, copy=False)

    import math
    g = math.gcd(fs_in, fs_out)
    up = fs_out // g
    down = fs_in // g
    return resample_poly(arr, up, down).astype(np.float32, copy=False)


def _resample_2d(x: Optional[np.ndarray], fs_in: int, fs_out: int) -> Optional[np.ndarray]:
    if x is None:
        return None

    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for resampling, got shape={arr.shape}")

    if fs_in == fs_out:
        return arr.astype(np.float32, copy=False)

    cols = [
        _resample_1d(arr[:, i], fs_in, fs_out)
        for i in range(arr.shape[1])
    ]
    return np.stack(cols, axis=1).astype(np.float32, copy=False)


def _safe_tail_value(x: Optional[np.ndarray]) -> Optional[float]:
    if x is None or len(x) == 0:
        return None
    return float(x[-1])


@dataclass
class WesadStream:
    subject: str
    session_id: str
    ecg: np.ndarray
    eda: Optional[np.ndarray]
    resp: Optional[np.ndarray]
    accel: Optional[np.ndarray]
    temp: Optional[np.ndarray]
    labels: Optional[np.ndarray] = None
    duration_sec: float = 0.0


def load_wesad_chest(wesad_root, subject: str) -> WesadStream:
    wesad_root = _as_path(wesad_root)
    pkl_path = _find_subject_pkl(wesad_root, subject)

    with pkl_path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected WESAD pickle structure in {pkl_path}: {type(data)}")

    sig = _pick_present(data, "signal", "signals")
    if sig is None:
        raise ValueError(f"WESAD pickle missing signal/signals block: {pkl_path}")
    if not isinstance(sig, dict):
        raise ValueError(f"WESAD signal block is not a dict in {pkl_path}")

    chest = _pick_present(sig, "chest", "Chest")
    if chest is None:
        raise ValueError(f"WESAD pickle missing chest/Chest block: {pkl_path}")
    if not isinstance(chest, dict):
        raise ValueError(f"WESAD chest block is not a dict in {pkl_path}")

    ecg = _pick_present(chest, "ECG", "ecg")
    eda = _pick_present(chest, "EDA", "eda")
    resp = _pick_present(chest, "Resp", "RESP", "resp", "Respiration")
    acc = _pick_present(chest, "ACC", "acc")
    temp = _pick_present(chest, "Temp", "TEMP", "temp")

    if ecg is None:
        raise ValueError(f"WESAD chest ECG missing for {subject}")
    if acc is None:
        raise ValueError(f"WESAD chest ACC missing for {subject}")

    ecg = _to_1d(ecg).astype(np.float32, copy=False)
    eda = _to_1d(eda).astype(np.float32, copy=False) if eda is not None else None
    resp = _to_1d(resp).astype(np.float32, copy=False) if resp is not None else None
    temp = _to_1d(temp).astype(np.float32, copy=False) if temp is not None else None
    accel = _to_2d_acc(acc)

    labels = _pick_present(data, "label", "labels")
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)

    ecg_r = _resample_1d(ecg, FS_WESAD_ECG, FS_ECG)
    eda_r = _resample_1d(eda, FS_WESAD_EDA, FS_EDA) if eda is not None else None
    resp_r = _resample_1d(resp, FS_WESAD_RESP, FS_RESP) if resp is not None else None
    acc_r = _resample_2d(accel, FS_WESAD_ACC, FS_IMU)
    temp_r = _resample_1d(temp, FS_WESAD_TEMP, FS_TEMP) if temp is not None else None

    duration_sec = float(len(ecg_r) / FS_ECG)

    return WesadStream(
        subject=subject,
        session_id=f"wesad_{subject}",
        ecg=ecg_r,
        eda=eda_r,
        resp=resp_r,
        accel=acc_r,
        temp=temp_r,
        labels=labels,
        duration_sec=duration_sec,
    )


def iter_wesad_sensorframes(
    wesad_root,
    subject: str,
    window_sec: float = 30.0,
    step_sec: float = 10.0,
    max_windows: Optional[int] = None,
) -> Iterator[SensorFrame]:
    wesad_root = _as_path(wesad_root)
    ws = load_wesad_chest(wesad_root, subject)

    win = float(window_sec)
    step = float(step_sec)

    if win <= 0:
        raise ValueError("window_sec must be > 0")
    if step <= 0:
        raise ValueError("step_sec must be > 0")

    ecg_win = int(round(win * FS_ECG))
    ecg_step = int(round(step * FS_ECG))

    if ecg_win <= 0 or ecg_step <= 0:
        raise ValueError("Computed ECG window/step must be positive")

    def _slice_1d(x: Optional[np.ndarray], fs: int, i0_ecg: int, dur_sec: float) -> Optional[np.ndarray]:
        if x is None:
            return None
        start = int(round(i0_ecg * (fs / FS_ECG)))
        length = int(round(dur_sec * fs))
        end = start + length
        if start >= len(x):
            return None
        return x[start:min(end, len(x))]

    def _slice_2d(x: Optional[np.ndarray], fs: int, i0_ecg: int, dur_sec: float) -> Optional[np.ndarray]:
        if x is None:
            return None
        start = int(round(i0_ecg * (fs / FS_ECG)))
        length = int(round(dur_sec * fs))
        end = start + length
        if start >= len(x):
            return None
        return x[start:min(end, len(x)), :]

    i0 = 0
    k = 0

    while i0 + ecg_win <= len(ws.ecg):
        if max_windows is not None and k >= int(max_windows):
            break

        ecg_win_arr = _slice_1d(ws.ecg, FS_ECG, i0, win)
        eda_win_arr = _slice_1d(ws.eda, FS_EDA, i0, win)
        resp_win_arr = _slice_1d(ws.resp, FS_RESP, i0, win)
        accel_win_arr = _slice_2d(ws.accel, FS_IMU, i0, win)
        temp_win_arr = _slice_1d(ws.temp, FS_TEMP, i0, win)

        frame = SensorFrame(
            session_id=ws.session_id,
            subject_id=ws.subject,
            timestamp=float(i0 / FS_ECG),
            window_sec=win,
            label=TaskLabel.UNKNOWN,
            ecg=ecg_win_arr,
            eda=eda_win_arr,
            respiration=resp_win_arr,
            accel=accel_win_arr,
            gyro=None,
            temperature=_safe_tail_value(temp_win_arr),
            alcohol=None,
            belt_tension=1.0,
        )

        yield frame
        i0 += ecg_step
        k += 1
