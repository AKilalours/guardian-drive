from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PulseStep:
    on_ms: int
    off_ms: int


@dataclass(frozen=True)
class HapticPattern:
    name: str
    steps: List[PulseStep]
    cooldown_sec: float
    description: str = ""


PATTERNS: Dict[str, HapticPattern] = {
    "caution_drowsy": HapticPattern(
        name="caution_drowsy",
        steps=[PulseStep(220, 180), PulseStep(220, 0)],
        cooldown_sec=8.0,
        description="Two short pulses for caution-level drowsiness.",
    ),
    "pull_over_drowsy": HapticPattern(
        name="pull_over_drowsy",
        steps=[PulseStep(350, 150), PulseStep(350, 150), PulseStep(350, 0)],
        cooldown_sec=5.0,
        description="Three strong pulses for pull-over drowsiness.",
    ),
    "urgent_cardiac": HapticPattern(
        name="urgent_cardiac",
        steps=[PulseStep(600, 140), PulseStep(600, 140), PulseStep(600, 0)],
        cooldown_sec=4.0,
        description="Heavy urgent pattern for severe cardiac risk.",
    ),
    "neurologic_emergency": HapticPattern(
        name="neurologic_emergency",
        steps=[PulseStep(500, 180), PulseStep(300, 120), PulseStep(500, 0)],
        cooldown_sec=4.0,
        description="Distinct urgent pattern for possible neurologic emergency.",
    ),
}


@dataclass
class HapticEvent:
    ts: float
    pattern: str
    accepted: bool
    reason: str
    meta: Dict[str, object] = field(default_factory=dict)


class BaseSeatHaptics:
    def emit(self, pattern_name: str, *, reason: str = "", meta: Optional[Dict[str, object]] = None) -> HapticEvent:
        raise NotImplementedError


class ConsoleSeatHaptics(BaseSeatHaptics):
    """
    Honest stub.
    This does not drive hardware. It rate-limits and logs patterns so the rest
    of the runtime can behave like a real actuator pipeline.
    """

    def __init__(self) -> None:
        self._last_emit_ts: Dict[str, float] = {}

    def emit(self, pattern_name: str, *, reason: str = "", meta: Optional[Dict[str, object]] = None) -> HapticEvent:
        now = time.time()
        meta = meta or {}

        if pattern_name not in PATTERNS:
            evt = HapticEvent(ts=now, pattern=pattern_name, accepted=False, reason=f"unknown_pattern:{pattern_name}", meta=meta)
            logger.warning("seat_haptics unknown pattern=%s", pattern_name)
            return evt

        pattern = PATTERNS[pattern_name]
        last = self._last_emit_ts.get(pattern_name, 0.0)
        if now - last < pattern.cooldown_sec:
            evt = HapticEvent(ts=now, pattern=pattern_name, accepted=False, reason="cooldown", meta=meta)
            logger.debug("seat_haptics cooldown pattern=%s", pattern_name)
            return evt

        self._last_emit_ts[pattern_name] = now
        step_str = " ".join(f"[{s.on_ms}ms on/{s.off_ms}ms off]" for s in pattern.steps)
        logger.warning("[SEAT_HAPTIC] pattern=%s reason=%s %s meta=%s", pattern_name, reason, step_str, meta)
        return HapticEvent(ts=now, pattern=pattern_name, accepted=True, reason=reason or "ok", meta=meta)


try:  # optional only
    import serial  # type: ignore
except Exception:  # pragma: no cover
    serial = None


class SerialSeatHaptics(BaseSeatHaptics):
    """
    Optional serial/GPIO bridge.
    Device protocol is intentionally simple:
      HAPTIC <pattern_name>\n
    Replace with your real belt/seat MCU protocol when you actually have it.
    """

    def __init__(self, port: str, baudrate: int = 115200, *, fallback: Optional[BaseSeatHaptics] = None) -> None:
        self.port = port
        self.baudrate = baudrate
        self.fallback = fallback or ConsoleSeatHaptics()
        self._last_emit_ts: Dict[str, float] = {}
        self._ser = None
        self._open_if_possible()

    def _open_if_possible(self) -> None:
        if serial is None:
            logger.warning("pyserial not available; SerialSeatHaptics will fall back to console")
            return
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=1)
            logger.info("Seat haptics serial opened port=%s baud=%s", self.port, self.baudrate)
        except Exception as e:  # pragma: no cover
            logger.warning("Could not open serial seat haptics on %s: %s", self.port, e)
            self._ser = None

    def emit(self, pattern_name: str, *, reason: str = "", meta: Optional[Dict[str, object]] = None) -> HapticEvent:
        now = time.time()
        meta = meta or {}
        if pattern_name not in PATTERNS:
            return HapticEvent(ts=now, pattern=pattern_name, accepted=False, reason=f"unknown_pattern:{pattern_name}", meta=meta)

        pattern = PATTERNS[pattern_name]
        last = self._last_emit_ts.get(pattern_name, 0.0)
        if now - last < pattern.cooldown_sec:
            return HapticEvent(ts=now, pattern=pattern_name, accepted=False, reason="cooldown", meta=meta)
        self._last_emit_ts[pattern_name] = now

        if self._ser is None:
            return self.fallback.emit(pattern_name, reason=reason, meta=meta)

        try:
            self._ser.write(f"HAPTIC {pattern_name}\n".encode("utf-8"))
            self._ser.flush()
            logger.warning("[SEAT_HAPTIC_SERIAL] pattern=%s reason=%s meta=%s", pattern_name, reason, meta)
            return HapticEvent(ts=now, pattern=pattern_name, accepted=True, reason=reason or "ok", meta=meta)
        except Exception as e:  # pragma: no cover
            logger.exception("Seat haptics serial write failed: %s", e)
            return self.fallback.emit(pattern_name, reason=f"serial_error:{e}", meta=meta)


def build_default_haptics() -> BaseSeatHaptics:
    return ConsoleSeatHaptics()
