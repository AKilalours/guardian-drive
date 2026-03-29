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
