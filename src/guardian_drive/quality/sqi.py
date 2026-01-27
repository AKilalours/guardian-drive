from __future__ import annotations

import numpy as np


def _clip_fraction_mad(x: np.ndarray, k: float = 20.0) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-8
    thr = k * mad
    return float(np.mean(np.abs(x - med) > thr))


def _sigmoid01(x: np.ndarray, k: float = 10.0) -> np.ndarray:
    """
    Vectorized sigmoid in (0,1), stable enough for moderate k.
    x can be scalar or ndarray.
    """
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-k * x))


def compute_sqi(X: np.ndarray, channel_names: list[str], rules: dict[str, float]) -> np.ndarray:
    """
    Continuous SQI in [0,1]. Higher is better.

    Penalizes:
      - missing/flat channels
      - low ECG std
      - high ECG clip fraction (MAD-based)
    """
    if X.ndim != 3:
        raise ValueError("X must be (N,C,T)")
    N, C, _ = X.shape
    if len(channel_names) != C:
        raise ValueError("channel_names length must match X.shape[1]")

    missing_channel_max_frac = float(rules.get("missing_channel_max_frac", 0.25))
    ecg_min_std = float(rules.get("ecg_min_std", 0.005))
    ecg_max_clip_frac = float(rules.get("ecg_max_clip_frac", 0.05))

    # start at 1.0 for each window
    q = np.ones((N,), dtype=np.float32)

    # ---- missing/flat penalty (continuous) ----
    chan_std = np.std(np.nan_to_num(X, nan=0.0), axis=2)
    chan_nan_frac = np.mean(np.isnan(X), axis=2)
    missing = (chan_std < 1e-12) | (chan_nan_frac > 0.0)
    missing_frac = np.mean(missing, axis=1).astype(np.float32)  # (N,)

    # margin > 0 means "better than threshold"
    miss_margin = (missing_channel_max_frac - missing_frac) / (missing_channel_max_frac + 1e-8)
    miss_score = _sigmoid01(miss_margin, k=8.0)
    miss_score = np.clip(miss_score, 0.0, 1.0).astype(np.float32)
    q *= miss_score

    # ---- ECG penalties ----
    if "ECG" in channel_names:
        idx = channel_names.index("ECG")
        ecg = X[:, idx, :]

        # std score: higher std => better
        ecg_std = np.std(ecg, axis=1).astype(np.float32)  # (N,)
        std_margin = (ecg_std - ecg_min_std) / (ecg_min_std + 1e-8)
        std_score = _sigmoid01(std_margin, k=6.0)
        std_score = np.clip(std_score, 0.0, 1.0).astype(np.float32)
        q *= std_score

        # clip score: lower clip frac => better
        clip_fracs = np.array([_clip_fraction_mad(ecg[i]) for i in range(N)], dtype=np.float32)
        clip_margin = (ecg_max_clip_frac - clip_fracs) / (ecg_max_clip_frac + 1e-8)
        clip_score = _sigmoid01(clip_margin, k=8.0)
        clip_score = np.clip(clip_score, 0.0, 1.0).astype(np.float32)
        q *= clip_score

    return np.clip(q, 0.0, 1.0).astype(np.float32)
