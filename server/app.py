from __future__ import annotations

"""
Guardian Drive — Real-Time API + Dashboard Server (LIVE ONLY)

Run:
  GD_LIVE_ONLY=1 GD_ENABLE_SEAT_ECG=1 GD_ENABLE_WEBCAM=1 GD_HOST=0.0.0.0 GD_PORT=8000 python -m server.app

Open on laptop:
  http://127.0.0.1:8000

Open on phone:
  https://<your-current-tunnel>.trycloudflare.com/gps_sender.html
"""

import asyncio
import json
import math
import os
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from acquisition.seat_ecg_node import SeatECGNode
from sqi.compute import compute_sqi
from features.extract import extract_features
from policy.fusion import FusionEngine
from policy.state_machine import SafetyStateMachine
from acquisition.models import (
    AlertLevel,
    ArrhythmiaClass,
    CrashSeverity,
    PolicyAction,
    RiskState,
    SensorFrame,
    TaskLabel,
)
from server.routes_sensor import router as sensor_router, bind_seat_node

from integrations.gps_runtime import RuntimeGPS
from integrations.navigation_local import LocalHospitalNav
from integrations.navigation_osm import OSMHospitalNav
from integrations.telephony_console import ConsoleTelephony
from integrations.telephony_twilio import TwilioTelephony
from integrations.vehicle_sim import NoOpVehicleControl
from integrations.base import DispatchMessage
from integrations.discord_webhook import DiscordNotifier

WEBCAM_ENABLED = os.getenv("GD_ENABLE_WEBCAM", "0").strip().lower() in {"1", "true", "yes", "on"}
GD_LIVE_ONLY = os.getenv("GD_LIVE_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}
SEAT_ECG_ENABLED = os.getenv("GD_ENABLE_SEAT_ECG", "1").strip().lower() in {"1", "true", "yes", "on"}
SEAT_ECG_FS_HZ = float(os.getenv("GD_SEAT_ECG_FS_HZ", "250"))
SEAT_ECG_STALE_SEC = float(os.getenv("GD_SEAT_ECG_STALE_SEC", "3.0"))

try:
    from integrations.vision_webcam import WebcamMonitor
except Exception:
    WebcamMonitor = None

BASE_DIR = Path(__file__).resolve().parent.parent
SERVER_DIR = Path(__file__).resolve().parent
STATIC_DIR = SERVER_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC_DEFAULT = 30.0
STEP_SEC_DEFAULT = 1.0
DURATION_SEC_DEFAULT = 3600.0

NAV_REFRESH_SEC = float(os.getenv("GD_NAV_REFRESH_SEC", "15"))
NAV_MOVE_M = float(os.getenv("GD_NAV_MOVE_M", "75"))

PIPELINE_TASK = None
CAM = None
SEAT_NODE = SeatECGNode(max_seconds=max(120.0, WINDOW_SEC_DEFAULT * 4.0), fs_hz=SEAT_ECG_FS_HZ)


class ScenarioReq(BaseModel):
    scenario: str


class GpsReq(BaseModel):
    lat: float
    lon: float
    accuracy_m: Optional[float] = None
    timestamp_unix: Optional[float] = None


@dataclass
class RuntimeState:
    scenario: str = "live_only"
    window_sec: float = WINDOW_SEC_DEFAULT
    step_sec: float = STEP_SEC_DEFAULT
    duration_sec: float = DURATION_SEC_DEFAULT
    clients: Set[WebSocket] = None


@dataclass
class CompatPolicyAction:
    timestamp: float = field(default_factory=time.monotonic)
    level: AlertLevel = AlertLevel.NOMINAL
    voice_message: str = ""
    display_message: str = ""
    escalate_911: bool = False
    hospital_advisory: bool = False
    log_reason: str = ""
    persistence_sec: float = 0.0
    corroborated_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": float(self.timestamp),
            "level": int(self.level.value),
            "level_name": self.level.name,
            "voice_message": self.voice_message,
            "display_message": self.display_message,
            "escalate_911": bool(self.escalate_911),
            "hospital_advisory": bool(self.hospital_advisory),
            "log_reason": self.log_reason,
            "persistence_sec": float(self.persistence_sec),
            "corroborated_by": list(self.corroborated_by),
        }


state = RuntimeState(clients=set())

LAST = {
    "payload": None,
    "dispatch": None,
    "route": None,
    "gps": None,
}

RUNTIME_INFO = {
    "fusion": None,
    "source": None,
    "source_kind": None,
    "startup_time": time.time(),
    "live_only": GD_LIVE_ONLY,
}

NAV_CACHE = {
    "fix": None,
    "route": None,
    "ts": 0.0,
}


def _print_runtime_banner() -> None:
    print("\n" + "=" * 78)
    print("GUARDIAN DRIVE — LIVE-ONLY MODE ACTIVE")
    print("=" * 78)
    print(f"  live_only:        {GD_LIVE_ONLY}")
    print(f"  seat_ecg_enabled: {SEAT_ECG_ENABLED}")
    print(f"  webcam_enabled:   {WEBCAM_ENABLED and WebcamMonitor is not None}")
    print("  fallback_policy:  NONE")
    print("  behavior:         process live inputs only, otherwise idle")
    print("=" * 78 + "\n")


def _find_static_file(name: str) -> Optional[Path]:
    for p in (STATIC_DIR / name, BASE_DIR / name):
        if p.exists() and p.is_file():
            return p
    return None


def _get_field(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _task_to_policy_block(task: Any, kind: str) -> dict:
    if task is None:
        return {"score": 0.0, "details": {}}

    details = dict(getattr(task, "details", {}) or {})
    score = 0.0

    if kind == "task_a":
        abstained = bool(getattr(task, "abstained", False))
        cls = getattr(task, "cls", None)
        cls_value = getattr(cls, "value", str(cls) if cls is not None else "unknown")
        abnormal = (not abstained) and cls_value not in {"normal", "noisy", "unknown"}
        score = float(getattr(task, "confidence", 0.0) or 0.0) if abnormal else 0.0
        details.update({
            "cls": cls_value,
            "hr_bpm": _get_field(task, "hr_bpm"),
            "rr_irr": float(_get_field(task, "rr_irr", 0.0) or 0.0),
            "p_frac": float(_get_field(task, "p_frac", 0.0) or 0.0),
        })

    elif kind == "task_b":
        score = float(getattr(task, "score", 0.0) or 0.0)
        details.update({
            "hr_contrib": float(getattr(task, "hr_contrib", 0.0) or 0.0),
            "hrv_contrib": float(getattr(task, "hrv_contrib", 0.0) or 0.0),
            "resp_contrib": float(getattr(task, "resp_contrib", 0.0) or 0.0),
            "eda_contrib": float(getattr(task, "eda_contrib", 0.0) or 0.0),
            "imu_contrib": float(getattr(task, "imu_contrib", 0.0) or 0.0),
            "confidence": float(getattr(task, "confidence", 0.0) or 0.0),
        })

    elif kind == "task_c":
        detected = bool(getattr(task, "detected", False))
        sev = getattr(task, "severity", None)
        sev_val = int(getattr(sev, "value", 0) or 0)
        conf = float(getattr(task, "confidence", 0.0) or 0.0)
        if detected and sev_val >= CrashSeverity.SEVERE.value:
            score = max(conf, 0.95)
        elif detected and sev_val >= CrashSeverity.MILD.value:
            score = max(conf, 0.55)
        else:
            score = 0.0
        details.update({
            "detected": detected,
            "severity": sev_val,
            "g_peak": float(getattr(task, "g_peak", 0.0) or 0.0),
            "jerk_peak": float(getattr(task, "jerk_peak", 0.0) or 0.0),
        })

    return {"score": float(score), "details": details}


def _riskstate_to_policy_input(rs: RiskState) -> dict:
    return {
        "task_a": _task_to_policy_block(rs.arrhythmia, "task_a"),
        "task_b": _task_to_policy_block(rs.drowsiness, "task_b"),
        "task_c": _task_to_policy_block(rs.crash, "task_c"),
        "task_d": {"score": 0.0, "details": {}},
    }


def _normalize_policy_action(raw: Any) -> PolicyAction | CompatPolicyAction:
    if raw is None:
        return CompatPolicyAction(
            level=AlertLevel.INACTIVE,
            display_message="No policy output available.",
            log_reason="policy returned None",
        )

    if hasattr(raw, "level") and hasattr(raw, "display_message"):
        return raw

    state_name = _get_field(raw, "state", "nominal")
    summary = str(_get_field(raw, "summary", "") or "Monitoring.")
    reasons = list(_get_field(raw, "reasons", []) or [])
    route_kind = str(_get_field(raw, "route_kind", "none") or "none")
    prepare_dispatch = bool(_get_field(raw, "prepare_dispatch", False))
    notify_contact = bool(_get_field(raw, "notify_contact", False))

    level_map = {
        "nominal": AlertLevel.NOMINAL,
        "advisory": AlertLevel.ADVISORY,
        "caution": AlertLevel.CAUTION,
        "pull_over": AlertLevel.PULLOVER,
        "escalate": AlertLevel.ESCALATE,
    }
    level = level_map.get(state_name, AlertLevel.NOMINAL)

    return CompatPolicyAction(
        timestamp=float(_get_field(raw, "ts", time.monotonic()) or time.monotonic()),
        level=level,
        voice_message=summary,
        display_message=summary,
        escalate_911=bool(level == AlertLevel.ESCALATE or prepare_dispatch or notify_contact),
        hospital_advisory=bool(route_kind == "er"),
        log_reason=" | ".join(str(r) for r in reasons) if reasons else summary,
        persistence_sec=0.0,
        corroborated_by=[str(r) for r in reasons],
    )


def _run_policy(sm: SafetyStateMachine, rs: RiskState) -> PolicyAction | CompatPolicyAction:
    try:
        raw = sm.step(rs)
        return _normalize_policy_action(raw)
    except Exception:
        raw = sm.step(_riskstate_to_policy_input(rs))
        return _normalize_policy_action(raw)


def _dispatch_script(dispatch: DispatchMessage | None) -> str:
    if dispatch is None:
        return "No dispatch available yet."

    meta = dispatch.meta or {}
    loc = meta.get("gps") or {}
    hosp = meta.get("route") or {}
    rest = meta.get("rest_guidance") or {}

    lat = loc.get("lat")
    lon = loc.get("lon")
    hname = hosp.get("destination_name")
    heta = hosp.get("eta_sec")
    dist = hosp.get("distance_m")

    eta_min = round(float(heta) / 60.0, 1) if isinstance(heta, (int, float)) else None
    km = round(float(dist) / 1000.0, 2) if isinstance(dist, (int, float)) else None

    lines = [
        "GUARDIAN DRIVE — HUMAN-IN-THE-LOOP EMERGENCY SCRIPT",
        "",
        "Hi, I need emergency assistance.",
        f"Reason: {dispatch.title}",
        f"Details: {dispatch.body}",
    ]

    if lat is not None and lon is not None:
        lines.append(f"Location: {lat:.6f}, {lon:.6f}")
        lines.append(f"Maps: https://www.google.com/maps?q={lat:.6f},{lon:.6f}")

    if hname:
        extra = []
        if km is not None:
            extra.append(f"{km} km")
        if eta_min is not None:
            extra.append(f"{eta_min} min")
        lines.append(f"Nearest ER (advisory): {hname}" + (" — " + " · ".join(extra) if extra else ""))

    if rest:
        msg = rest.get("message")
        mins = rest.get("recommended_stop_min")
        if msg:
            lines.append(f"Rest guidance: {msg}")
        if mins is not None:
            lines.append(f"Recommended stop duration: {mins} min")

    lines += [
        "",
        "This system is not a medical device. It flags risk patterns and recommends evaluation.",
    ]
    return "\n".join(lines)


def _mk_telephony():
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    tok = os.getenv("TWILIO_AUTH_TOKEN")
    frm = os.getenv("TWILIO_FROM_NUMBER")
    to = os.getenv("GD_EMERGENCY_CONTACT_NUMBER")
    if sid and tok and frm and to:
        try:
            return TwilioTelephony(to_number=to)
        except Exception:
            return ConsoleTelephony()
    return ConsoleTelephony()


def _fix_to_dict(fix):
    if not fix:
        return None
    return {
        "lat": fix.point.lat,
        "lon": fix.point.lon,
        "accuracy_m": fix.accuracy_m,
        "speed_mps": getattr(fix, "speed_mps", None),
        "heading_deg": getattr(fix, "heading_deg", None),
        "timestamp_unix": fix.timestamp_unix,
    }


def _route_to_dict(route):
    if not route:
        return None
    return {
        "destination_name": route.destination_name,
        "lat": route.destination_point.lat,
        "lon": route.destination_point.lon,
        "eta_sec": route.eta_sec,
        "distance_m": route.distance_m,
        "provider": route.provider,
        "notes": route.notes,
    }


def _dispatch_to_dict(dispatch: DispatchMessage | None):
    if dispatch is None:
        return None
    return {
        "title": dispatch.title,
        "body": dispatch.body,
        "meta": dispatch.meta,
    }


def _rough_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2.0))
    y = math.radians(lat2 - lat1)
    return math.sqrt(x * x + y * y) * r


def _maybe_route(fix, nav_osm, nav_local):
    if not fix:
        return None

    now = time.monotonic()
    cached_fix = NAV_CACHE["fix"]
    cached_route = NAV_CACHE["route"]
    cached_ts = NAV_CACHE["ts"]

    need_refresh = cached_route is None
    if not need_refresh:
        if (now - cached_ts) >= NAV_REFRESH_SEC:
            need_refresh = True
        elif cached_fix is None:
            need_refresh = True
        else:
            moved = _rough_distance_m(
                float(cached_fix.point.lat),
                float(cached_fix.point.lon),
                float(fix.point.lat),
                float(fix.point.lon),
            )
            if moved >= NAV_MOVE_M:
                need_refresh = True

    if need_refresh:
        route = nav_osm.nearest_er(fix) or nav_local.nearest_er(fix)
        if route is not None:
            NAV_CACHE["fix"] = fix
            NAV_CACHE["route"] = route
            NAV_CACHE["ts"] = now

    return NAV_CACHE["route"]


def _downsample_1d(arr, n=600):
    if arr is None:
        return None
    x = np.asarray(arr, dtype=np.float32).reshape(-1)
    if x.size < 2:
        return None
    if x.size <= n:
        return x.tolist()
    idx = np.linspace(0, x.size - 1, n).astype(np.int32)
    return x[idx].tolist()


def _feature_bundle_to_ui_features(fb) -> dict:
    return {
        "hr_bpm": None if fb.ecg.hr_bpm is None else float(fb.ecg.hr_bpm),
        "hrv_rmssd": None if fb.ecg.hrv_rmssd is None else float(fb.ecg.hrv_rmssd),
        "hrv_sdnn": None if fb.ecg.hrv_sdnn is None else float(fb.ecg.hrv_sdnn),
        "rr_irregularity": float(fb.ecg.rr_irregularity or 0.0),
        "p_wave_fraction": float(fb.ecg.p_wave_fraction or 0.0),
        "qrs_ms": None if fb.ecg.qrs_duration_ms is None else float(fb.ecg.qrs_duration_ms),
        "st_elev_mv": float(getattr(fb.ecg, "st_elev_mv", 0.0) or 0.0),
        "resp_rate_bpm": None if fb.resp.rate_bpm is None else float(fb.resp.rate_bpm),
        "resp_irregularity": float(fb.resp.irregularity or 0.0),
        "eda_scl_mean": None if fb.eda.scl_mean is None else float(fb.eda.scl_mean),
        "eda_scl_slope": None if fb.eda.scl_slope is None else float(fb.eda.scl_slope),
        "posture_score": float(fb.imu.posture_score or 0.0),
        "accel_rms": float(fb.imu.accel_rms or 0.0),
        "crash_g_peak": float(fb.imu.crash_g_peak or 0.0),
        "crash_jerk": float(fb.imu.crash_jerk or 0.0),
        "temperature": None if fb.temperature is None else float(fb.temperature),
        "alcohol": None if fb.alcohol is None else float(fb.alcohol),
        "belt_tension": None if fb.belt_tension is None else float(fb.belt_tension),
        "ecg_source": str(getattr(fb.ecg, "source", "") or ""),
        "seat_ecg_quality": float(getattr(fb.sqi, "seat_ecg_quality", 0.0) or 0.0),
        "seat_contact": bool(getattr(fb.sqi, "seat_contact", False)),
    }


def _build_previews(frame, fb) -> dict:
    ecg_src = getattr(fb.ecg, "samples", None)
    if ecg_src is None:
        ecg_src = getattr(frame, "seat_ecg", None)
    if ecg_src is None:
        ecg_src = getattr(frame, "ecg", None)

    resp_src = getattr(frame, "respiration", None)

    return {
        "ecg": _downsample_1d(ecg_src, n=700),
        "resp": _downsample_1d(resp_src, n=240),
    }


def _normalize_webcam_metrics(cam_metrics: dict | None) -> dict | None:
    if cam_metrics is None:
        return None

    eyes_closed = cam_metrics.get("eyes_closed")
    drowsy_score = cam_metrics.get("drowsy_score")
    perclos = cam_metrics.get("perclos_30s")
    yawn_count = cam_metrics.get("yawn_count_30s")
    note = cam_metrics.get("note")

    out = dict(cam_metrics)
    out["eyes"] = (
        "closed" if eyes_closed is True else
        "open" if eyes_closed is False else
        None
    )
    out["perclos"] = perclos
    out["yawns"] = yawn_count
    out["drowsy"] = bool(float(drowsy_score) >= 0.70) if drowsy_score is not None else False
    out["status_text"] = note or ("webcam active" if out.get("available") else "webcam unavailable")
    return out


def _rest_guidance(rs) -> dict | None:
    d = getattr(rs, "drowsiness", None)
    if d is None or d.abstained:
        return None

    score = float(d.score)
    if score >= 0.85:
        return {
            "recommended_stop_min": 30,
            "message": "Severe drowsiness detected. Pull over now and rest at least 30 minutes.",
        }
    if score >= 0.60:
        return {
            "recommended_stop_min": 15,
            "message": "Moderate drowsiness detected. Plan a rest stop within the next few minutes.",
        }
    return None


def _build_dispatch(action, rs, fb, fix, route) -> DispatchMessage:
    lines = []
    title = "GUARDIAN DRIVE — URGENT SAFETY EVENT"

    rest = _rest_guidance(rs)

    if rs.crash and rs.crash.detected and rs.crash.severity.value >= 2:
        title = "GUARDIAN DRIVE — SEVERE CRASH DETECTED"
        lines.append(
            f"Crash severity: SEVERE (g_peak={rs.crash.g_peak:.1f}g, conf={rs.crash.confidence:.2f})"
        )

    elif rs.arrhythmia and (not rs.arrhythmia.abstained) and rs.arrhythmia.cls.value != "normal":
        title = "GUARDIAN DRIVE — CARDIAC RISK DETECTED"
        hr_txt = "n/a" if rs.arrhythmia.hr_bpm is None else f"{rs.arrhythmia.hr_bpm:.0f}"
        lines.append(
            f"ECG risk: {rs.arrhythmia.cls.value.upper()} "
            f"(HR={hr_txt} bpm, conf={rs.arrhythmia.confidence:.2f})"
        )

    elif rs.drowsiness and (not rs.drowsiness.abstained):
        title = "GUARDIAN DRIVE — DROWSINESS / FATIGUE ALERT"
        lines.append(
            f"Drowsiness score: {rs.drowsiness.score:.2f} "
            f"(conf={rs.drowsiness.confidence:.2f})"
        )

    if fix:
        acc_txt = "n/a" if fix.accuracy_m is None else str(fix.accuracy_m)
        lines.append(
            f"Location: {fix.point.lat:.6f}, {fix.point.lon:.6f} "
            f"(±{acc_txt} m)"
        )
        lines.append(f"Maps: https://www.google.com/maps?q={fix.point.lat:.6f},{fix.point.lon:.6f}")

    if action.hospital_advisory and route:
        lines.append(
            f"Nearest ER: {route.destination_name} "
            f"({route.destination_point.lat:.6f}, {route.destination_point.lon:.6f})"
        )

    if rest:
        lines.append(rest["message"])
        lines.append(f"Recommended stop duration: {rest['recommended_stop_min']} minutes")

    body = "\n".join(lines) if lines else "Urgent safety event detected."

    meta = {
        "policy_level": action.level.name if hasattr(action.level, "name") else str(action.level),
        "timestamp_monotonic": action.timestamp,
        "gps": _fix_to_dict(fix),
        "route": _route_to_dict(route) if action.hospital_advisory else None,
        "rest_guidance": rest,
    }
    return DispatchMessage(title=title, body=body, meta=meta)


def _build_manual_status_dispatch() -> DispatchMessage | None:
    payload = LAST.get("payload")
    if payload is None:
        return None

    gps = LAST.get("gps")
    route = LAST.get("route")
    rs_task_a = payload.get("task_a") or {}
    rs_task_b = payload.get("task_b") or {}
    policy = payload.get("policy") or {}

    title = "GUARDIAN DRIVE — MANUAL STATUS SHARE"
    lines = [
        f"Policy: {policy.get('level_name', 'unknown')}",
    ]

    if rs_task_a:
        lines.append(
            f"Task A cardiac: {rs_task_a.get('cls')} conf={rs_task_a.get('confidence')}"
        )
    if rs_task_b:
        lines.append(
            f"Task B drowsiness: score={rs_task_b.get('score')} conf={rs_task_b.get('confidence')}"
        )

    if gps:
        lines.append(f"Location: {gps.point.lat:.6f}, {gps.point.lon:.6f}")
        lines.append(f"Maps: https://www.google.com/maps?q={gps.point.lat:.6f},{gps.point.lon:.6f}")

    if route:
        lines.append(f"Nearest ER: {route.destination_name}")

    return DispatchMessage(
        title=title,
        body="\n".join(lines),
        meta={
            "gps": _fix_to_dict(gps),
            "route": _route_to_dict(route),
            "manual": True,
        },
    )


def _runtime_status_payload():
    last = LAST.get("payload") or {}
    fusion_status = RUNTIME_INFO.get("fusion")
    return {
        "ok": True,
        "server": {
            "host_env": os.getenv("GD_HOST", "0.0.0.0"),
            "port_env": int(os.getenv("GD_PORT", "8000")),
            "startup_unix": float(RUNTIME_INFO["startup_time"]),
            "base_dir": str(BASE_DIR),
            "static_dir": str(STATIC_DIR),
        },
        "runtime": {
            "live_only": bool(GD_LIVE_ONLY),
            "selected_source": RUNTIME_INFO.get("source"),
            "selected_source_kind": RUNTIME_INFO.get("source_kind"),
            "scenario": state.scenario,
            "window_sec": state.window_sec,
            "step_sec": state.step_sec,
            "duration_sec": state.duration_sec,
            "webcam_enabled": bool(CAM is not None),
            "seat_ecg_enabled": bool(SEAT_ECG_ENABLED),
            "seat_ecg_status": SEAT_NODE.snapshot(),
            "connected_ws_clients": len(state.clients),
            "fusion": fusion_status,
        },
        "artifacts": {
            "dashboard_exists": bool(_find_static_file("GuardianDrive_Dashboard.html")),
            "gps_sender_exists": bool(_find_static_file("gps_sender.html")),
            "preview_exists": bool(_find_static_file("preview.html")),
        },
        "last": {
            "window_index": last.get("window_index"),
            "scenario": last.get("scenario"),
            "have_payload": LAST.get("payload") is not None,
            "have_dispatch": LAST.get("dispatch") is not None,
            "have_route": LAST.get("route") is not None,
            "have_gps": LAST.get("gps") is not None,
        },
        "claim_guardrail": (
            fusion_status.get("claim_guardrail")
            if isinstance(fusion_status, dict)
            else None
        ),
    }


async def _broadcast(payload: dict) -> None:
    if not state.clients:
        return

    msg = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    dead = []

    for ws in list(state.clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)

    for ws in dead:
        state.clients.discard(ws)


def _latest_cam_metrics() -> dict | None:
    if CAM is None:
        return None
    try:
        return _normalize_webcam_metrics(CAM.latest_metrics().to_dict())
    except Exception as e:
        return {
            "available": False,
            "note": f"webcam metrics failed: {e}",
            "eyes": None,
            "perclos": None,
            "yawns": None,
            "drowsy": False,
        }


def _live_seat_window(window_sec: float):
    if not SEAT_ECG_ENABLED:
        return None, SEAT_NODE.snapshot()

    snap = SEAT_NODE.snapshot()
    if not bool(snap.get("connected", False)):
        return None, snap

    last_packet_ts = float(snap.get("last_packet_ts", 0.0) or 0.0)
    if last_packet_ts <= 0.0:
        return None, snap

    age = time.time() - last_packet_ts
    if age > SEAT_ECG_STALE_SEC:
        return None, snap

    win = SEAT_NODE.latest_window(window_sec)
    min_needed = max(64, int(float(win.fs_hz) * float(window_sec) * 0.60))
    if len(win.ecg) < min_needed:
        return None, snap

    return win, snap


def _build_live_frame(window_sec: float, cam_metrics: dict | None):
    seat_win, seat_snap = _live_seat_window(window_sec)
    has_seat = seat_win is not None
    has_webcam = bool(cam_metrics and cam_metrics.get("available", False))

    if not has_seat and not has_webcam:
        return None, None, None, seat_snap

    frame = SensorFrame(
        session_id="live_runtime",
        subject_id="live",
        timestamp=time.monotonic(),
        window_sec=float(window_sec),
        label=TaskLabel.UNKNOWN,
        webcam_metrics=dict(cam_metrics or {}) if cam_metrics else None,
    )

    if has_seat:
        frame.seat_ecg = np.asarray(seat_win.ecg, dtype=np.float32)
        frame.seat_ecg_fs_hz = float(seat_win.fs_hz)
        frame.seat_ecg_meta = dict(seat_win.meta or {})

    if has_seat and has_webcam:
        source = "seat+webcam"
        source_kind = "live_multimodal"
    elif has_seat:
        source = "seat_live"
        source_kind = "seat_live"
    else:
        source = "webcam_live"
        source_kind = "webcam_live"

    return frame, source, source_kind, seat_snap


def _idle_payload(reason: str, cam_metrics: dict | None, fix, route, seat_status: dict) -> dict:
    return {
        "version": "real-guarded",
        "scenario": "live_only",
        "source_kind": "idle",
        "t_sec": 0.0,
        "window_index": 0,
        "features": {},
        "previews": {"ecg": None, "resp": None},
        "sqi": {
            "overall_confidence": 0.0,
            "abstain": True,
            "flags": "NO_LIVE_SENSOR",
            "ecg_quality": 0.0,
            "seat_ecg_quality": 0.0,
            "eda_contact": 0.0,
            "resp_quality": 0.0,
            "motion_level": 0.0,
            "seat_motion": 0.0,
            "belt_worn": False,
            "belt_quality": 0.0,
            "seat_contact": False,
            "ecg_usable": False,
            "eda_usable": False,
            "resp_usable": False,
            "imu_usable": False,
        },
        "task_a": None,
        "task_b": None,
        "task_c": None,
        "policy": {
            "timestamp": time.monotonic(),
            "level": int(AlertLevel.INACTIVE.value),
            "level_name": AlertLevel.INACTIVE.name,
            "voice_message": "",
            "display_message": reason,
            "escalate_911": False,
            "hospital_advisory": False,
            "log_reason": reason,
            "persistence_sec": 0.0,
            "corroborated_by": [],
        },
        "dispatch": None,
        "gps": _fix_to_dict(fix),
        "route": _route_to_dict(route),
        "webcam": cam_metrics,
        "seat_ecg": seat_status,
        "rest_guidance": None,
        "runtime": {
            "fusion": RUNTIME_INFO.get("fusion"),
        },
    }


async def _pipeline_loop():
    fusion = FusionEngine()
    sm = SafetyStateMachine()
    gps = RuntimeGPS()
    nav_osm = OSMHospitalNav()
    nav_local = LocalHospitalNav(BASE_DIR / "data" / "hospitals.csv")
    tel = _mk_telephony()
    discord = DiscordNotifier.from_env()
    veh = NoOpVehicleControl()

    RUNTIME_INFO["fusion"] = fusion.status_dict()

    last_level = None
    t0 = time.monotonic()
    win_i = 0

    while True:
        cam_metrics = _latest_cam_metrics()
        frame, current_source, current_source_kind, seat_status = _build_live_frame(state.window_sec, cam_metrics)

        fix = gps.read_fix()
        route = _maybe_route(fix, nav_osm, nav_local)

        if frame is None:
            RUNTIME_INFO["source"] = "live_only"
            RUNTIME_INFO["source_kind"] = "idle"

            payload = _idle_payload(
                reason="Waiting for live seat ECG or live webcam data. Simulation fallback is disabled.",
                cam_metrics=cam_metrics,
                fix=fix,
                route=route,
                seat_status=seat_status,
            )
            LAST["payload"] = payload
            LAST["route"] = route
            LAST["gps"] = fix
            await _broadcast(payload)
            await asyncio.sleep(max(0.25, state.step_sec))
            continue

        if RUNTIME_INFO["source_kind"] != current_source_kind:
            RUNTIME_INFO["source"] = current_source
            RUNTIME_INFO["source_kind"] = current_source_kind
            t0 = time.monotonic()
            win_i = 0

        win_i += 1

        sqi = compute_sqi(frame)
        fb = extract_features(frame, sqi, state.window_sec)

        rs = fusion.run(fb)
        action = _run_policy(sm, rs)

        should_prepare_dispatch = (
            bool(action.escalate_911)
            or bool(action.level.value >= AlertLevel.PULLOVER.value)
        )
        dispatch = _build_dispatch(action, rs, fb, fix, route) if should_prepare_dispatch else None

        if action.escalate_911 and dispatch is not None:
            tel.dispatch_simulation(message=dispatch)
            tel.notify_emergency_contact(message=f"{dispatch.title}\n{dispatch.body}", meta=dispatch.meta)

            if discord is not None and last_level != "ESCALATE":
                try:
                    discord.send(title=dispatch.title, body=dispatch.body, meta=dispatch.meta)
                except Exception:
                    pass

        if action.level.value >= AlertLevel.PULLOVER.value:
            veh.request_safe_pull_over(
                reason=action.display_message or "pullover",
                meta={"route": _route_to_dict(route)},
            )

        payload = {
            "version": "real-guarded",
            "scenario": current_source,
            "source_kind": current_source_kind,
            "t_sec": round(time.monotonic() - t0, 3),
            "window_index": win_i,
            "features": _feature_bundle_to_ui_features(fb),
            "previews": _build_previews(frame, fb),
            "sqi": fb.sqi.to_dict(),
            "task_a": rs.arrhythmia.to_dict() if rs.arrhythmia else None,
            "task_b": rs.drowsiness.to_dict() if rs.drowsiness else None,
            "task_c": rs.crash.to_dict() if rs.crash else None,
            "policy": action.to_dict(),
            "dispatch": _dispatch_to_dict(dispatch),
            "gps": _fix_to_dict(fix),
            "route": _route_to_dict(route),
            "webcam": cam_metrics,
            "seat_ecg": seat_status,
            "rest_guidance": _rest_guidance(rs),
            "runtime": {
                "fusion": RUNTIME_INFO["fusion"],
                "live_only": bool(GD_LIVE_ONLY),
            },
        }

        LAST["dispatch"] = dispatch
        LAST["route"] = route
        LAST["gps"] = fix
        LAST["payload"] = payload

        await _broadcast(payload)
        last_level = action.level.name
        await asyncio.sleep(state.step_sec)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CAM, PIPELINE_TASK

    bind_seat_node(SEAT_NODE)
    _print_runtime_banner()

    if WEBCAM_ENABLED and WebcamMonitor is not None:
        try:
            CAM = WebcamMonitor(cam_index=int(os.getenv("GD_WEBCAM_INDEX", "0")))
            print("[GD] Webcam enabled.")
        except Exception as e:
            CAM = None
            print(f"[GD] Webcam failed to start: {e}")

    PIPELINE_TASK = asyncio.create_task(_pipeline_loop())
    try:
        yield
    finally:
        if PIPELINE_TASK is not None:
            PIPELINE_TASK.cancel()
            with suppress(asyncio.CancelledError):
                await PIPELINE_TASK
            PIPELINE_TASK = None

        if CAM is not None and hasattr(CAM, "close"):
            try:
                CAM.close()
            except Exception:
                pass
        CAM = None


app = FastAPI(title="Guardian Drive Server", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(sensor_router)


@app.get("/")
def index():
    p = _find_static_file("GuardianDrive_Dashboard.html")
    if p:
        return FileResponse(str(p))
    return HTMLResponse(
        "<h3>GuardianDrive_Dashboard.html not found</h3>"
        "<p>Put it in server/static/ or repo root.</p>",
        status_code=404,
    )


@app.get("/gps_sender.html")
def gps_sender():
    p = _find_static_file("gps_sender.html")
    if p:
        return FileResponse(str(p))
    return HTMLResponse(
        "<h3>gps_sender.html not found</h3>"
        "<p>Put it in server/static/ or repo root.</p>",
        status_code=404,
    )


@app.get("/preview.html")
def preview():
    p = _find_static_file("preview.html")
    if p:
        return FileResponse(str(p))
    return HTMLResponse(
        "<h3>preview.html not found</h3>"
        "<p>Put it in server/static/ or repo root.</p>",
        status_code=404,
    )


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/healthz")
def healthz():
    return JSONResponse({"ok": True, "ts": time.time()})


@app.get("/api/runtime_status")
def runtime_status():
    return JSONResponse(_runtime_status_payload())


@app.post("/api/scenario")
def set_scenario(req: ScenarioReq):
    return JSONResponse(
        {
            "ok": True,
            "scenario": "live_only",
            "ignored": True,
            "reason": "LIVE-ONLY mode is active. Scenario changes are disabled.",
        }
    )


@app.post("/api/gps")
def update_gps(req: GpsReq):
    RuntimeGPS.set_fix(
        lat=req.lat,
        lon=req.lon,
        accuracy_m=req.accuracy_m,
        timestamp_unix=req.timestamp_unix,
    )
    return JSONResponse({"ok": True})


@app.post("/api/notify_contact")
def notify_contact():
    d = LAST.get("dispatch") or _build_manual_status_dispatch()
    if d is None:
        return JSONResponse({"ok": False, "error": "No status available yet."}, status_code=400)

    tel = _mk_telephony()
    tel.notify_emergency_contact(message=f"{d.title}\n{d.body}", meta=d.meta)
    return JSONResponse({"ok": True})


@app.post("/api/notify_discord")
def notify_discord():
    notifier = DiscordNotifier.from_env()
    if notifier is None:
        return JSONResponse(
            {"ok": False, "error": "GD_DISCORD_WEBHOOK_URL or DISCORD_WEBHOOK_URL not set."},
            status_code=400,
        )

    d = LAST.get("dispatch") or _build_manual_status_dispatch()
    if d is None:
        return JSONResponse({"ok": False, "error": "No status available yet."}, status_code=400)

    ok = notifier.send(title=d.title, body=d.body, meta=d.meta)
    return JSONResponse({"ok": bool(ok)})


@app.get("/api/dispatch_script")
def dispatch_script():
    d = LAST.get("dispatch") or _build_manual_status_dispatch()
    return JSONResponse({"ok": True, "script": _dispatch_script(d)})


@app.get("/api/maps_url")
def maps_url():
    r = LAST.get("route")
    if not r:
        return JSONResponse({"ok": False, "error": "No route available yet."}, status_code=400)
    lat = r.destination_point.lat
    lon = r.destination_point.lon
    name = r.destination_name
    url = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}"
    return JSONResponse({"ok": True, "url": url, "name": name})


@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    state.clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        state.clients.discard(ws)


@app.get("/camera")
def camera_page():
    return HTMLResponse(
        """
        <html>
          <head><title>Guardian Drive — Camera</title></head>
          <body style="font-family: sans-serif; background:#081626; color:#d8e6f8;">
            <h3>Guardian Drive — Webcam</h3>
            <p>If you see video, webcam access is live.</p>
            <img src="/camera.mjpg" style="max-width: 900px; width: 100%; border: 1px solid #2c4a6a; border-radius: 12px;" />
          </body>
        </html>
        """
    )


@app.get("/camera.mjpg")
def camera_mjpeg():
    if CAM is None:
        return JSONResponse(
            {
                "ok": False,
                "error": "Webcam not enabled. Set GD_ENABLE_WEBCAM=1 and fix webcam dependencies."
            },
            status_code=400,
        )

    async def gen():
        boundary = "frame"
        while True:
            jpg = CAM.latest_jpeg()
            if jpg:
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                    + jpg + b"\r\n"
                )
            await asyncio.sleep(0.08)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("GD_PORT", "8000"))
    host = os.getenv("GD_HOST", "0.0.0.0")
    uvicorn.run("server.app:app", host=host, port=port, reload=False)
