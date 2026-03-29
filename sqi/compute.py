from __future__ import annotations

"""Signal Quality Index (SQI) computation.

Goal: decide when the system should ABSTAIN rather than hallucinate confidence.

In a production system, SQI would be learned/calibrated (per-channel classifiers,
contact impedance, electrode-off detection, motion artifacts, etc.).

Here we use lightweight heuristics that are deterministic and good enough for
the pipeline demo + unit tests.
"""

import numpy as np

from acquisition.models import SensorFrame, SQIState


def compute_sqi(frame: SensorFrame) -> SQIState:
    belt_tension = float(frame.belt_tension) if frame.belt_tension is not None else 0.0
    belt_worn = bool(belt_tension > 0.20)

    # Motion from IMU (0..1)
    motion = 0.0
    if frame.accel is not None and len(frame.accel) > 0:
        # remove gravity-ish mean
        a = np.asarray(frame.accel, dtype=float)
        mag = np.sqrt((a * a).sum(axis=1))
        motion = float(np.clip(np.std(mag - np.mean(mag)) / 1.5, 0.0, 1.0))

    # ECG quality from simple SNR proxy (0..1)
    ecg_q = 0.0
    if frame.ecg is not None and len(frame.ecg) > 10:
        x = np.asarray(frame.ecg, dtype=float)
        # Estimate noise without letting QRS spikes dominate the standard deviation.
        p = np.percentile(np.abs(x), 95)
        noise = np.std(np.clip(x, -p, p)) + 1e-6
        snr = (np.max(np.abs(x)) + 1e-6) / noise
        # Map SNR to a 0..1 quality score (tuned for spike-train ECG).
        ecg_q = float(np.clip((snr - 5.0) / 10.0, 0.0, 1.0))
        # penalize high motion (motion artifacts)
        ecg_q *= float(np.clip(1.0 - 1.2 * motion, 0.0, 1.0))

    # EDA contact (0..1): stable derivative implies good contact
    eda_q = 0.0
    if frame.eda is not None and len(frame.eda) > 10:
        x = np.asarray(frame.eda, dtype=float)
        dx = np.diff(x)
        noise = float(np.std(dx))
        eda_q = float(np.clip(1.0 - noise / 1.2, 0.0, 1.0))
        eda_q *= float(np.clip(1.0 - 1.0 * motion, 0.0, 1.0))

    # Resp quality (0..1): enough variance / amplitude and belt worn
    resp_q = 0.0
    if frame.respiration is not None and len(frame.respiration) > 10:
        x = np.asarray(frame.respiration, dtype=float)
        amp = float(np.std(x))
        resp_q = float(np.clip(amp / 0.40, 0.0, 1.0))
        resp_q *= 1.0 if belt_worn else 0.0

    return SQIState(
        ecg_quality=ecg_q,
        eda_contact=eda_q,
        resp_quality=resp_q,
        motion_level=motion,
        belt_worn=belt_worn,
        belt_quality=float(np.clip(belt_tension, 0.0, 1.0)),
    )
