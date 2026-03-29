"""
Guardian Drive™ v4.0 — Core Data Models
Single source of truth for all typed data structures.

Design principles:
- Typed dataclasses throughout (no raw dicts in hot paths)
- SQI is a first-class citizen with explicit .abstain property
- Every PolicyAction carries a log_reason for traceability
- JSON-serializable for structured logging and session replay
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

# ─── Sample rates (Hz) ───────────────────────────────────────
FS_ECG  = 250
FS_EDA  = 16
FS_RESP = 50
FS_IMU  = 100
FS_TEMP = 5
FS_ALC  = 10
FS_BELT = 50


# ─── Enumerations ────────────────────────────────────────────

class TaskLabel(Enum):
    NORMAL       = "normal"
    ARRHYTHMIA   = "arrhythmia"
    DROWSY       = "drowsy"
    FATIGUED     = "fatigued"
    CRASH_MILD   = "crash_mild"
    CRASH_SEVERE = "crash_severe"
    ARTIFACT     = "artifact"
    UNKNOWN      = "unknown"


class DriverState(Enum):
    NORMAL   = "normal"
    AT_RISK  = "at_risk"
    ALERT    = "alert"
    ESCALATE = "escalate"


class AlertLevel(Enum):
    INACTIVE = 0   # SQI abstain — cannot decide
    NOMINAL  = 1
    ADVISORY = 2
    CAUTION  = 3
    PULLOVER = 4
    ESCALATE = 5


class ArrhythmiaClass(Enum):
    NORMAL      = "normal"
    AFIB        = "afib"
    TACHYCARDIA = "tachycardia"
    BRADYCARDIA = "bradycardia"
    NOISY       = "noisy"      # abstain — SQI below threshold
    UNKNOWN     = "unknown"


class CrashSeverity(Enum):
    NONE   = 0
    MILD   = 1
    SEVERE = 2


# ─── Raw sensor frame ─────────────────────────────────────────

@dataclass
class SensorFrame:
    """One acquisition window. Arrays may be None if channel absent/failed."""
    session_id:   str       = ""
    subject_id:   str       = ""
    timestamp:    float     = field(default_factory=time.monotonic)
    window_sec:   float     = 30.0
    label:        TaskLabel = TaskLabel.UNKNOWN

    ecg:          Optional[np.ndarray] = None   # legacy ECG / simulator ECG
    eda:          Optional[np.ndarray] = None
    respiration:  Optional[np.ndarray] = None
    accel:        Optional[np.ndarray] = None
    gyro:         Optional[np.ndarray] = None

    # new live seat path
    seat_ecg:        Optional[np.ndarray] = None
    seat_ecg_fs_hz:  Optional[float] = None
    seat_ecg_meta:   Dict[str, Any] = field(default_factory=dict)

    # live webcam bridge
    webcam_metrics: Optional[Dict[str, Any]] = None

    temperature:  Optional[float] = None
    alcohol:      Optional[float] = None
    belt_tension: Optional[float] = None

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {}
        d["session_id"]   = self.session_id
        d["subject_id"]   = self.subject_id
        d["timestamp"]    = self.timestamp
        d["window_sec"]   = self.window_sec
        d["label"]        = self.label.value
        d["temperature"]  = self.temperature
        d["alcohol"]      = self.alcohol
        d["belt_tension"] = self.belt_tension
        d["seat_ecg_fs_hz"] = self.seat_ecg_fs_hz
        d["seat_ecg_meta"] = dict(self.seat_ecg_meta or {})
        d["webcam_metrics"] = dict(self.webcam_metrics or {}) if self.webcam_metrics else None

        for k in ("ecg", "eda", "respiration", "accel", "gyro", "seat_ecg"):
            v = getattr(self, k)
            d[k] = v.tolist() if v is not None else None

        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ─── Signal Quality Index ─────────────────────────────────────

@dataclass
class SQIState:
    """
    Per-channel quality probabilities. ABSTAIN when too few channels usable.
    This is a feature, not a bug: the system must say 'I don't know'
    rather than making confident errors.
    """
    timestamp:        float = field(default_factory=time.monotonic)

    # legacy channels
    ecg_quality:      float = 0.0
    eda_contact:      float = 0.0
    resp_quality:     float = 0.0
    motion_level:     float = 0.0   # 0=still, 1=extreme
    belt_worn:        bool  = False
    belt_quality:     float = 0.0

    # new seat ECG path
    seat_ecg_quality: float = 0.0
    seat_contact:     bool  = False
    seat_motion:      float = 0.0

    @property
    def ecg_usable(self) -> bool:
        effective_quality = max(float(self.ecg_quality), float(self.seat_ecg_quality))
        effective_motion = max(float(self.motion_level), float(self.seat_motion))
        has_interface = bool(self.belt_worn or self.seat_contact)
        return effective_quality >= 0.55 and effective_motion < 0.40 and has_interface

    @property
    def eda_usable(self) -> bool:
        return self.eda_contact >= 0.45 and self.motion_level < 0.40

    @property
    def resp_usable(self) -> bool:
        return self.resp_quality >= 0.50 and self.belt_worn

    @property
    def imu_usable(self) -> bool:
        return True  # IMU always present; motion_level captures its reliability

    @property
    def abstain(self) -> bool:
        """True = refuse to make any safety-critical decision this window."""
        n = sum([self.ecg_usable, self.eda_usable, self.resp_usable, self.imu_usable])
        return n < 2

    @property
    def overall_confidence(self) -> float:
        effective_ecg_q = max(float(self.ecg_quality), float(self.seat_ecg_quality))
        effective_motion = max(float(self.motion_level), float(self.seat_motion))
        interface_ok = 1.0 if (self.belt_worn or self.seat_contact) else 0.0
        wv = [
            (effective_ecg_q,             1.5),
            (self.eda_contact,            1.0),
            (self.resp_quality,           1.0),
            (1.0 - effective_motion,      1.0),
            (interface_ok,                0.8),
        ]
        return float(sum(v * w for v, w in wv) / sum(w for _, w in wv))

    def summary(self) -> str:
        flags = []
        if not self.ecg_usable:
            flags.append("ECG↓")
        if self.seat_contact and self.seat_ecg_quality < 0.55:
            flags.append("SeatECG↓")
        if not self.eda_usable:
            flags.append("EDA↓")
        if not self.resp_usable:
            flags.append("RESP↓")
        if not self.belt_worn and not self.seat_contact:
            flags.append("NoInterface")
        if max(self.motion_level, self.seat_motion) > 0.35:
            flags.append("Motion")
        if self.abstain:
            flags.append("ABSTAIN")
        return f"SQI={self.overall_confidence:.2f} [{', '.join(flags) or 'OK'}]"

    def to_dict(self) -> dict:
        flags = []
        if not self.ecg_usable:
            flags.append("ECG↓")
        if self.seat_contact and self.seat_ecg_quality < 0.55:
            flags.append("SeatECG↓")
        if not self.eda_usable:
            flags.append("EDA↓")
        if not self.resp_usable:
            flags.append("RESP↓")
        if not self.belt_worn and not self.seat_contact:
            flags.append("NoInterface")
        if max(self.motion_level, self.seat_motion) > 0.35:
            flags.append("Motion")
        if self.abstain:
            flags.append("ABSTAIN")

        return {
            "timestamp": float(self.timestamp),
            "overall_confidence": float(self.overall_confidence),
            "abstain": bool(self.abstain),
            "flags": ",".join(flags) if flags else "OK",
            "ecg_quality": float(self.ecg_quality),
            "seat_ecg_quality": float(self.seat_ecg_quality),
            "eda_contact": float(self.eda_contact),
            "resp_quality": float(self.resp_quality),
            "motion_level": float(self.motion_level),
            "seat_motion": float(self.seat_motion),
            "belt_worn": bool(self.belt_worn),
            "belt_quality": float(self.belt_quality),
            "seat_contact": bool(self.seat_contact),
            "ecg_usable": bool(self.ecg_usable),
            "eda_usable": bool(self.eda_usable),
            "resp_usable": bool(self.resp_usable),
            "imu_usable": bool(self.imu_usable),
        }


# ─── Feature bundles ──────────────────────────────────────────

@dataclass
class ECGFeatures:
    hr_bpm:           Optional[float] = None
    hrv_rmssd:        Optional[float] = None
    hrv_sdnn:         Optional[float] = None
    rr_irregularity:  float           = 0.0
    p_wave_fraction:  float           = 1.0
    qrs_duration_ms:  Optional[float] = None
    st_elev_mv:       float           = 0.0
    ectopic_fraction: float           = 0.0

    samples:          Optional[np.ndarray] = None
    waveform:         Optional[np.ndarray] = None
    signal:           Optional[np.ndarray] = None
    lead_ii:          Optional[np.ndarray] = None

    fs_hz:            float = float(FS_ECG)
    sampling_rate_hz: float = float(FS_ECG)
    source:           str   = ""
    details:          Dict[str, Any] = field(default_factory=dict)


@dataclass
class EDAFeatures:
    scl_mean:         Optional[float] = None
    scl_slope:        Optional[float] = None
    scr_rate_per_min: Optional[float] = None
    scr_amplitude:    Optional[float] = None


@dataclass
class RespFeatures:
    rate_bpm:       Optional[float] = None
    irregularity:   float           = 0.0
    amplitude_mean: Optional[float] = None
    shallow_flag:   bool            = False


@dataclass
class IMUFeatures:
    accel_rms:     float = 0.0
    jerk_peak:     float = 0.0
    posture_score: float = 1.0
    crash_flag:    bool  = False
    crash_g_peak:  float = 0.0
    crash_jerk:    float = 0.0


@dataclass
class FeatureBundle:
    timestamp:    float     = field(default_factory=time.monotonic)
    window_sec:   float     = 30.0
    session_id:   str       = ""
    subject_id:   str       = ""
    label:        TaskLabel = TaskLabel.UNKNOWN
    sqi:          SQIState  = field(default_factory=SQIState)

    ecg:          ECGFeatures  = field(default_factory=ECGFeatures)
    eda:          EDAFeatures  = field(default_factory=EDAFeatures)
    resp:         RespFeatures = field(default_factory=RespFeatures)
    imu:          IMUFeatures  = field(default_factory=IMUFeatures)

    webcam_metrics: Optional[Dict[str, Any]] = None
    seat_ecg_status: Dict[str, Any] = field(default_factory=dict)

    temperature:  Optional[float] = None
    alcohol:      Optional[float] = None
    belt_tension: Optional[float] = None


# ─── Model outputs ─────────────────────────────────────────────

@dataclass
class ArrhythmiaResult:
    timestamp:     float           = field(default_factory=time.monotonic)
    cls:           ArrhythmiaClass = ArrhythmiaClass.UNKNOWN
    confidence:    float           = 0.0
    abstained:     bool            = False
    hr_bpm:        Optional[float] = None
    rr_irr:        float           = 0.0
    p_frac:        float           = 1.0
    reason:        str             = ""
    features_used: List[str]       = field(default_factory=list)
    details:       Dict[str, Any]  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": float(self.timestamp),
            "cls": self.cls.value if hasattr(self.cls, "value") else str(self.cls),
            "confidence": float(self.confidence),
            "abstained": bool(self.abstained),
            "hr_bpm": None if self.hr_bpm is None else float(self.hr_bpm),
            "rr_irr": float(self.rr_irr),
            "p_frac": float(self.p_frac),
            "reason": self.reason,
            "features_used": list(self.features_used),
            "details": dict(self.details),
        }


@dataclass
class DrowsinessResult:
    timestamp:    float = field(default_factory=time.monotonic)
    score:        float = 0.0   # 0=alert 1=asleep
    confidence:   float = 0.0
    abstained:    bool  = False
    hr_contrib:   float = 0.0
    hrv_contrib:  float = 0.0
    resp_contrib: float = 0.0
    eda_contrib:  float = 0.0
    imu_contrib:  float = 0.0
    reason:       str   = ""
    details:      Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": float(self.timestamp),
            "score": float(self.score),
            "confidence": float(self.confidence),
            "abstained": bool(self.abstained),
            "hr_contrib": float(self.hr_contrib),
            "hrv_contrib": float(self.hrv_contrib),
            "resp_contrib": float(self.resp_contrib),
            "eda_contrib": float(self.eda_contrib),
            "imu_contrib": float(self.imu_contrib),
            "reason": self.reason,
            "details": dict(self.details),
        }


@dataclass
class CrashResult:
    timestamp:         float         = field(default_factory=time.monotonic)
    detected:          bool          = False
    severity:          CrashSeverity = CrashSeverity.NONE
    confidence:        float         = 0.0
    latency_ms:        float         = 0.0
    g_peak:            float         = 0.0
    jerk_peak:         float         = 0.0
    belt_tension:      float         = 0.0
    belt_corroborated: bool          = False
    reason:            str           = ""
    details:           Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": float(self.timestamp),
            "detected": bool(self.detected),
            "severity": self.severity.value if hasattr(self.severity, "value") else str(self.severity),
            "confidence": float(self.confidence),
            "latency_ms": float(self.latency_ms),
            "g_peak": float(self.g_peak),
            "jerk_peak": float(self.jerk_peak),
            "belt_tension": float(self.belt_tension),
            "belt_corroborated": bool(self.belt_corroborated),
            "reason": self.reason,
            "details": dict(self.details),
        }


@dataclass
class RiskState:
    timestamp:    float       = field(default_factory=time.monotonic)
    session_id:   str         = ""
    overall_conf: float       = 0.0
    arrhythmia:   Optional[ArrhythmiaResult] = None
    drowsiness:   Optional[DrowsinessResult] = None
    crash:        Optional[CrashResult]      = None
    driver_state: DriverState = DriverState.NORMAL
    alert_level:  AlertLevel  = AlertLevel.NOMINAL
    abstained:    bool        = False


@dataclass
class PolicyAction:
    """Every action is fully traceable via log_reason and corroborated_by."""
    timestamp:         float      = field(default_factory=time.monotonic)
    level:             AlertLevel = AlertLevel.NOMINAL
    voice_message:     str        = ""
    display_message:   str        = ""
    escalate_911:      bool       = False   # simulation only in v4
    hospital_advisory: bool       = False   # advisory, not autopilot
    log_reason:        str        = ""
    persistence_sec:   float      = 0.0
    corroborated_by:   List[str]  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "level": int(self.level.value) if hasattr(self.level, "value") else int(self.level),
            "level_name": self.level.name if hasattr(self.level, "name") else str(self.level),
            "voice_message": self.voice_message,
            "display_message": self.display_message,
            "escalate_911": bool(self.escalate_911),
            "hospital_advisory": bool(self.hospital_advisory),
            "log_reason": self.log_reason,
            "persistence_sec": float(self.persistence_sec),
            "corroborated_by": list(self.corroborated_by),
        }
