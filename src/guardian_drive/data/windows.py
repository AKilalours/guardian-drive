from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from guardian_drive.data.wesad import SR_CHEST_HZ, SR_WRIST_HZ, load_subject_raw
from guardian_drive.quality.sqi import compute_sqi


def _reduce_ratio(up: int, down: int) -> tuple[int, int]:
    g = int(np.gcd(up, down))
    return up // g, down // g


def resample_1d(x: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    if src_hz == dst_hz:
        return x.astype(np.float32, copy=False)
    up, down = _reduce_ratio(dst_hz, src_hz)
    y = resample_poly(x, up, down)
    return y.astype(np.float32)


def resample_labels_nearest(labels: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    if src_hz == dst_hz:
        return labels.astype(np.int64, copy=False)
    n_src = labels.size
    n_dst = int(round(n_src * (dst_hz / src_hz)))
    if n_dst <= 1:
        raise ValueError("Resampled labels too short")
    idx = np.linspace(0, n_src - 1, num=n_dst)
    idx = np.round(idx).astype(np.int64)
    return labels[idx]


def window_stack(
    signals: dict[str, np.ndarray],
    labels: np.ndarray,
    channel_names: list[str],
    sr_hz: int,
    window_sec: float,
    stride_sec: float,
    label_strategy: str = "majority",
) -> tuple[np.ndarray, np.ndarray]:
    win_len = int(round(window_sec * sr_hz))
    stride = int(round(stride_sec * sr_hz))
    if win_len <= 0 or stride <= 0:
        raise ValueError("window/stride must be positive")

    L = min([signals[c].shape[0] for c in channel_names] + [labels.shape[0]])
    if L < win_len:
        raise ValueError(f"Sequence too short for window: L={L} win_len={win_len}")

    Xs: list[np.ndarray] = []
    ys: list[int] = []
    for start in range(0, L - win_len + 1, stride):
        end = start + win_len
        w = np.stack([signals[c][start:end] for c in channel_names], axis=0)
        w_labels = labels[start:end]
        if label_strategy == "center":
            y = int(w_labels[len(w_labels) // 2])
        elif label_strategy == "majority":
            vals, cnts = np.unique(w_labels, return_counts=True)
            y = int(vals[int(np.argmax(cnts))])
        else:
            raise ValueError(f"Unknown label_strategy={label_strategy}")
        Xs.append(w)
        ys.append(y)

    return np.stack(Xs, axis=0).astype(np.float32), np.asarray(ys, dtype=np.int64)


@dataclass
class ProcessedSubject:
    subject: str
    X: np.ndarray
    y: np.ndarray
    sqi: np.ndarray


def preprocess_wesad_subject(
    *,
    wesad_root: str | Path,
    subject: str,
    use_chest: bool,
    use_wrist: bool,
    chest_signals: Iterable[str],
    wrist_signals: Iterable[str],
    resample_hz: int,
    window_sec: float,
    stride_sec: float,
    label_strategy: str,
    sqi_rules: dict[str, float] | None,
) -> ProcessedSubject:
    raw = load_subject_raw(
        wesad_root,
        subject,
        chest_signals=chest_signals if use_chest else [],
        wrist_signals=wrist_signals if use_wrist else [],
    )

    signals: dict[str, np.ndarray] = {}
    channel_names: list[str] = []

    if use_chest:
        for k, x in raw.chest.items():
            signals[k] = resample_1d(x, SR_CHEST_HZ, resample_hz)
            channel_names.append(k)

    if use_wrist:
        for k, x in raw.wrist.items():
            sr = SR_WRIST_HZ.get(k)
            if sr is None:
                raise KeyError(f"Unknown wrist signal sampling rate for {k}")
            signals[k] = resample_1d(x, sr, resample_hz)
            channel_names.append(k)

    if not channel_names:
        raise ValueError("No signals selected")

    labels_rs = resample_labels_nearest(raw.labels, SR_CHEST_HZ, resample_hz)

    X, y = window_stack(
        signals=signals,
        labels=labels_rs,
        channel_names=channel_names,
        sr_hz=resample_hz,
        window_sec=window_sec,
        stride_sec=stride_sec,
        label_strategy=label_strategy,
    )

    sqi = compute_sqi(X, channel_names=channel_names, rules=sqi_rules or {})
    return ProcessedSubject(subject=subject, X=X, y=y, sqi=sqi)


def save_processed(
    subj: ProcessedSubject, out_path: str | Path, channel_names: list[str], sr_hz: int
) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        X=subj.X,
        y=subj.y,
        sqi=subj.sqi,
        channel_names=np.asarray(channel_names, dtype=object),
        sr_hz=np.asarray([sr_hz], dtype=np.int64),
        subject=np.asarray([subj.subject], dtype=object),
    )


def load_processed(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}
