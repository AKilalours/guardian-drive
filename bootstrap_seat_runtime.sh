#!/usr/bin/env bash
set -euo pipefail

mkdir -p acquisition sqi features server config docs tests

cat > acquisition/ring_buffer.py <<'PY'
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Tuple


@dataclass
class TimedSample:
    t: float
    x: float


class RingBuffer:
    def __init__(self, maxlen: int = 4096) -> None:
        self._buf: Deque[TimedSample] = deque(maxlen=maxlen)

    def append(self, t: float, x: float) -> None:
        self._buf.append(TimedSample(float(t), float(x)))

    def extend(self, samples: Iterable[Tuple[float, float]]) -> None:
        for t, x in samples:
            self.append(t, x)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def to_lists(self) -> Tuple[List[float], List[float]]:
        ts = [s.t for s in self._buf]
        xs = [s.x for s in self._buf]
        return ts, xs

    def last_n(self, n: int) -> Tuple[List[float], List[float]]:
        items = list(self._buf)[-int(n):]
        return [s.t for s in items], [s.x for s in items]
PY

cat > acquisition/seat_ecg_node.py <<'PY'
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from acquisition.ring_buffer import RingBuffer


@dataclass
class SeatECGStatus:
    connected: bool = False
    fs_hz: float = 250.0
    contact_ok: bool = False
    motion_score: float = 0.0
    last_packet_ts: float = 0.0
    packet_count: int = 0
    dropped_packets: int = 0
    source: str = "seat_ecg_stub"

    def to_dict(self) -> Dict[str, object]:
        return {
            "connected": self.connected,
            "fs_hz": self.fs_hz,
            "contact_ok": self.contact_ok,
            "motion_score": self.motion_score,
            "last_packet_ts": self.last_packet_ts,
            "packet_count": self.packet_count,
            "dropped_packets": self.dropped_packets,
            "source": self.source,
        }


@dataclass
class SeatECGWindow:
    ts: List[float] = field(default_factory=list)
    ecg: List[float] = field(default_factory=list)
    fs_hz: float = 250.0
    meta: Dict[str, object] = field(default_factory=dict)


class SeatECGNode:
    def __init__(self, max_seconds: float = 60.0, fs_hz: float = 250.0) -> None:
        self.fs_hz = float(fs_hz)
        self.max_seconds = float(max_seconds)
        self.buf = RingBuffer(maxlen=max(1024, int(self.max_seconds * self.fs_hz * 1.2)))
        self.status = SeatECGStatus(connected=False, fs_hz=self.fs_hz)
        self._last_seq: Optional[int] = None

    def ingest_packet(self, packet: Dict[str, object]) -> None:
        t0 = float(packet.get("t0", time.time()))
        fs_hz = float(packet.get("fs_hz", self.fs_hz))
        samples = list(packet.get("samples", []) or [])
        seq = packet.get("seq", None)
        contact_ok = bool(packet.get("contact_ok", False))
        motion_score = float(packet.get("motion_score", 0.0) or 0.0)

        if seq is not None and self._last_seq is not None and int(seq) != self._last_seq + 1:
            self.status.dropped_packets += max(0, int(seq) - self._last_seq - 1)
        if seq is not None:
            self._last_seq = int(seq)

        dt = 1.0 / fs_hz
        timed = [(t0 + i * dt, float(x)) for i, x in enumerate(samples)]
        self.buf.extend(timed)

        self.status.connected = True
        self.status.fs_hz = fs_hz
        self.status.contact_ok = contact_ok
        self.status.motion_score = motion_score
        self.status.last_packet_ts = time.time()
        self.status.packet_count += 1

    def latest_window(self, seconds: float = 10.0) -> SeatECGWindow:
        n = max(1, int(float(seconds) * self.status.fs_hz))
        ts, xs = self.buf.last_n(n)
        return SeatECGWindow(
            ts=ts,
            ecg=xs,
            fs_hz=self.status.fs_hz,
            meta=self.status.to_dict(),
        )

    def snapshot(self) -> Dict[str, object]:
        return self.status.to_dict()
PY

cat > sqi/contact_quality.py <<'PY'
from __future__ import annotations

from typing import Dict, List


def compute_contact_quality(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"contact_score": 0.0, "signal_range": 0.0, "dc_spread": 0.0}

    smin = min(samples)
    smax = max(samples)
    signal_range = float(smax - smin)
    mean = sum(samples) / len(samples)
    dc_spread = sum(abs(x - mean) for x in samples) / len(samples)

    score = 0.0
    if signal_range > 0.02:
        score += 0.45
    if dc_spread > 0.005:
        score += 0.35
    if signal_range > 0.05:
        score += 0.20

    return {
        "contact_score": max(0.0, min(1.0, score)),
        "signal_range": signal_range,
        "dc_spread": dc_spread,
    }
PY

cat > sqi/motion_artifact.py <<'PY'
from __future__ import annotations

from typing import Dict, List


def compute_motion_artifact_score(samples: List[float]) -> Dict[str, float]:
    if len(samples) < 4:
        return {"motion_score": 1.0, "hf_proxy": 0.0}

    diffs = [abs(samples[i] - samples[i - 1]) for i in range(1, len(samples))]
    hf_proxy = sum(diffs) / max(len(diffs), 1)
    motion_score = min(1.0, hf_proxy * 20.0)
    return {"motion_score": motion_score, "hf_proxy": hf_proxy}
PY

cat > sqi/window_quality.py <<'PY'
from __future__ import annotations

from typing import Dict, List

from sqi.contact_quality import compute_contact_quality
from sqi.motion_artifact import compute_motion_artifact_score


def compute_window_quality(samples: List[float]) -> Dict[str, object]:
    c = compute_contact_quality(samples)
    m = compute_motion_artifact_score(samples)

    score = 0.65 * c["contact_score"] + 0.35 * (1.0 - m["motion_score"])
    abstain = score < 0.40

    return {
        "overall_score": max(0.0, min(1.0, score)),
        "abstain": abstain,
        "contact": c,
        "motion": m,
        "summary": "usable" if not abstain else "poor",
    }
PY

cat > features/ecg_filter.py <<'PY'
from __future__ import annotations

from typing import List


def detrend_mean(samples: List[float]) -> List[float]:
    if not samples:
        return []
    mean = sum(samples) / len(samples)
    return [x - mean for x in samples]
PY

cat > features/rpeak_detect.py <<'PY'
from __future__ import annotations

from typing import Dict, List


def detect_rpeaks(samples: List[float], fs_hz: float) -> Dict[str, object]:
    if len(samples) < 5:
        return {"peak_indices": [], "peak_times_sec": []}

    mx = max(samples) if samples else 0.0
    thr = mx * 0.60
    refractory = max(1, int(0.25 * fs_hz))

    peaks: List[int] = []
    last_i = -refractory
    for i in range(1, len(samples) - 1):
        if i - last_i < refractory:
            continue
        if samples[i] > thr and samples[i] >= samples[i - 1] and samples[i] >= samples[i + 1]:
            peaks.append(i)
            last_i = i

    return {
        "peak_indices": peaks,
        "peak_times_sec": [i / fs_hz for i in peaks],
    }
PY

cat > features/hrv_live.py <<'PY'
from __future__ import annotations

import math
from typing import Dict, List


def rr_from_peak_times(peak_times_sec: List[float]) -> List[float]:
    if len(peak_times_sec) < 2:
        return []
    rr = []
    for i in range(1, len(peak_times_sec)):
        dt = peak_times_sec[i] - peak_times_sec[i - 1]
        if 0.3 <= dt <= 2.0:
            rr.append(dt)
    return rr


def compute_hrv(rr_sec: List[float]) -> Dict[str, float]:
    if not rr_sec:
        return {"hr_bpm": 0.0, "rmssd": 0.0, "sdnn": 0.0, "n_rr": 0}

    mean_rr = sum(rr_sec) / len(rr_sec)
    hr_bpm = 60.0 / mean_rr if mean_rr > 0 else 0.0

    if len(rr_sec) >= 2:
        diffs = [rr_sec[i] - rr_sec[i - 1] for i in range(1, len(rr_sec))]
        rmssd = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    else:
        rmssd = 0.0

    if len(rr_sec) >= 2:
        mean = mean_rr
        sdnn = math.sqrt(sum((x - mean) ** 2 for x in rr_sec) / len(rr_sec))
    else:
        sdnn = 0.0

    return {
        "hr_bpm": hr_bpm,
        "rmssd": rmssd,
        "sdnn": sdnn,
        "n_rr": len(rr_sec),
    }
PY

cat > features/resp_estimate.py <<'PY'
from __future__ import annotations

from typing import Dict, List


def estimate_resp_from_ecg(samples: List[float], fs_hz: float) -> Dict[str, float]:
    if len(samples) < max(10, int(fs_hz)):
        return {"resp_rate_bpm": 0.0, "confidence": 0.0}
    return {"resp_rate_bpm": 12.0, "confidence": 0.2}
PY

cat > server/routes_sensor.py <<'PY'
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional

router = APIRouter(prefix="/api/sensor", tags=["sensor"])

SEAT_NODE = None


def bind_seat_node(node) -> None:
    global SEAT_NODE
    SEAT_NODE = node


class SeatPacket(BaseModel):
    t0: float
    fs_hz: float = 250.0
    samples: List[float] = Field(default_factory=list)
    contact_ok: bool = False
    motion_score: float = 0.0
    seq: Optional[int] = None


@router.post("/seat_ecg")
def post_seat_ecg(packet: SeatPacket):
    if SEAT_NODE is None:
        return {"ok": False, "error": "seat node not bound"}
    SEAT_NODE.ingest_packet(packet.model_dump())
    return {"ok": True, "n": len(packet.samples)}


@router.get("/seat_ecg/status")
def get_seat_ecg_status():
    if SEAT_NODE is None:
        return {"ok": False, "error": "seat node not bound"}
    return {"ok": True, "status": SEAT_NODE.snapshot()}
PY

cat > config/seat_runtime.yaml <<'YAML'
seat_ecg:
  enabled: true
  fs_hz: 250
  window_sec: 10
  source: "seat_ecg_stub"
  contact_threshold: 0.40
  motion_threshold: 0.70

pipeline:
  use_webcam: true
  use_phone_gps: true
  use_rest_routing: true

claims:
  mode_label: "research prototype"
  allow_medical_grade_claims: false
  allow_vehicle_control_claims: false
YAML

cat > docs/REPO_ADAPTATION_PLAN.md <<'MD'
# Repo Adaptation Plan

## Goal
Integrate seat-based ECG ingestion into the existing Guardian Drive repo without pretending the system is production-ready.

## Files added
- acquisition/seat_ecg_node.py
- acquisition/ring_buffer.py
- sqi/contact_quality.py
- sqi/motion_artifact.py
- sqi/window_quality.py
- features/ecg_filter.py
- features/rpeak_detect.py
- features/hrv_live.py
- features/resp_estimate.py
- server/routes_sensor.py
- config/seat_runtime.yaml

## Next required patch points
1. `features/extract.py`
   - consume `SeatECGNode.latest_window(...)`
   - compute quality, filtered signal, peaks, HRV
   - populate your existing `FeatureBundle.ecg`

2. `main.py`
   - instantiate `SeatECGNode`
   - bind it to the server route with `bind_seat_node(node)`
   - replace or augment simulator path with live seat ECG windowing

3. `policy/fusion.py`
   - make sure Task A / Task B can consume live HR / HRV / SQI from the seat path

4. `ui/`
   - add seat ECG connection status panel
   - add signal quality indicator
   - add live waveform preview only when SQI is usable

## Honest claims after this stage
- live seat ECG ingestion path
- real-time multimodal prototype
- signal-quality-aware runtime pipeline

## Claims still not honest
- medical grade
- cardiac arrest detection
- production-ready emergency system
- real vehicle control
MD

cat > tests/test_ring_buffer.py <<'PY'
from acquisition.ring_buffer import RingBuffer


def test_ring_buffer_basic():
    rb = RingBuffer(maxlen=3)
    rb.append(1.0, 10.0)
    rb.append(2.0, 20.0)
    rb.append(3.0, 30.0)
    rb.append(4.0, 40.0)
    ts, xs = rb.to_lists()
    assert ts == [2.0, 3.0, 4.0]
    assert xs == [20.0, 30.0, 40.0]
PY

cat > tests/test_hrv_live.py <<'PY'
from features.hrv_live import rr_from_peak_times, compute_hrv


def test_hrv_live_basic():
    rr = rr_from_peak_times([0.0, 1.0, 2.0, 3.0])
    out = compute_hrv(rr)
    assert out["n_rr"] == 3
    assert 59.0 <= out["hr_bpm"] <= 61.0
PY

cat > tests/test_window_quality.py <<'PY'
from sqi.window_quality import compute_window_quality


def test_window_quality_nonempty():
    samples = [0.0, 0.01, 0.03, 0.01, -0.02, 0.0, 0.02]
    out = compute_window_quality(samples)
    assert "overall_score" in out
    assert "abstain" in out
PY

echo "Scaffold created successfully."
echo "Next: patch features/extract.py, policy/fusion.py, and main.py"
