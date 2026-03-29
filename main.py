from __future__ import annotations

"""
Guardian Drive™ v4.1 — Live Pipeline Demo (Terminal)

Usage:
  python main.py --scenario crash_severe --dispatch
  python main.py --scenario afib --telephony discord --dispatch
  python main.py --eval
  python main.py --list
  python main.py --scenario normal --seat-from-ecg

Notes on “real” (truth in advertising):
- Telephony can be console, Discord webhook, or Twilio (your configured numbers).
- GPS can be mock or phone-driven (gps_sender.html -> /api/gps) or manual lat/lon.
- Routing can be offline (local haversine) or OSRM (online / self-hosted).
- Vehicle can be noop or OBD-II telemetry (read-only).
- --seat-from-ecg is a bench-test bridge that routes existing ECG windows through the new seat path.

Emergency calling (e.g., 911) is NOT a drop-in feature. This demo escalates to
your configured emergency contact(s) and provides a Google Maps link + nearest
ER advisory.
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from acquisition.simulator import GuardianSimulator, SCENARIO_PARAMS
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
    FS_ECG,
)
from gd_logging.event_log import EventLogger

from integrations.base import DispatchMessage
from integrations.gps_mock import MockGPS
from integrations.navigation_local import LocalHospitalNav
from integrations.navigation_osrm import OSRMHospitalNav
from integrations.telephony_console import ConsoleTelephony
from integrations.vehicle_sim import NoOpVehicleControl

R = "\033[0m"
B = "\033[1m"
C = "\033[96m"
G = "\033[92m"
Y = "\033[93m"
RE = "\033[91m"
DM = "\033[2m"
BRE = "\033[41m"


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


def _make_gps(args):
    if args.gps == "manual":
        if args.lat is None or args.lon is None:
            raise SystemExit("--gps manual requires --lat and --lon")
        try:
            from integrations.gps_manual import ManualGPS
        except Exception as e:
            raise SystemExit(
                f"--gps manual requested but integrations.gps_manual is missing or broken: {e}"
            )
        return ManualGPS(lat=args.lat, lon=args.lon, accuracy_m=20.0)

    if args.gps == "phone":
        try:
            from integrations.gps_phone import PhoneGPS
        except Exception as e:
            raise SystemExit(
                f"--gps phone requested but integrations.gps_phone is missing or broken: {e}"
            )
        return PhoneGPS(server_base=args.gps_server)

    return MockGPS()


def _make_nav(args):
    if args.nav == "osrm":
        return OSRMHospitalNav(Path("data/hospitals.csv"))
    return LocalHospitalNav(Path("data/hospitals.csv"))


def _make_telephony(args):
    if args.telephony == "twilio":
        try:
            from integrations.telephony_twilio import TwilioTelephony
            return TwilioTelephony.from_env()
        except Exception as e:
            print(f"{Y}[warn] Twilio telephony unavailable, falling back to console: {e}{R}")
            return ConsoleTelephony()

    if args.telephony == "discord":
        try:
            from integrations.telephony_discord import DiscordWebhookTelephony
            return DiscordWebhookTelephony.from_env()
        except Exception as e:
            print(f"{Y}[warn] Discord telephony unavailable, falling back to console: {e}{R}")
            return ConsoleTelephony()

    return ConsoleTelephony()


def _make_vehicle(args):
    if args.vehicle == "obd":
        try:
            from integrations.vehicle_obd import OBDVehicleTelemetry
            return OBDVehicleTelemetry(port=args.obd_port)
        except Exception as e:
            print(f"{Y}[warn] OBD vehicle telemetry unavailable, falling back to noop: {e}{R}")
            return NoOpVehicleControl()

    return NoOpVehicleControl()


def bar(v: float, w: int = 20) -> str:
    c = RE if v > .75 else Y if v > .45 else G
    filled = max(0, min(w, int(v * w)))
    return c + "█" * filled + DM + "░" * (w - filled) + R


def _maps_url(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"


class DispatchLimiter:
    """Prevents spamming contacts when ESCALATE persists across windows."""
    def __init__(self, min_interval_sec: float = 60.0):
        self.min_interval_sec = float(min_interval_sec)
        self._last_sent = None

    def allow(self, now_mono: float) -> bool:
        if self._last_sent is None:
            self._last_sent = now_mono
            return True
        if (now_mono - self._last_sent) >= self.min_interval_sec:
            self._last_sent = now_mono
            return True
        return False


def render(fb, rs, action, i: int, sc: str, t_elapsed: float) -> None:
    print("\033[2J\033[H", end="")
    print(B + C + "━" * 70)
    print(f"  GUARDIAN DRIVE™ v4.1  │  {sc.upper():<15}│  Window #{i:<3}│  T+{t_elapsed:.0f}s")
    print("━" * 70 + R)

    sqi = fb.sqi
    sq_color = RE if sqi.abstain else Y if sqi.overall_confidence < 0.6 else G
    print(f"\n{B}  SIGNAL QUALITY{R}  {sq_color}{B}{sqi.summary()}{R}")
    if sqi.abstain:
        print(f"  {Y}  ↳ System abstaining — insufficient signal quality to make decisions{R}")

    ecg_source = str(getattr(fb.ecg, "source", "") or "")
    if ecg_source:
        print(f"  {DM}ECG source: {ecg_source} | seat_q={getattr(sqi, 'seat_ecg_quality', 0.0):.2f}{R}")

    print(f"\n{B}  TASK A — Arrhythmia Screening{R}")
    if rs.arrhythmia:
        a = rs.arrhythmia
        if getattr(a, "abstained", False):
            print(f"  {DM}ABSTAIN: {getattr(a,'reason','')}{R}")
        else:
            ac = RE if a.cls not in (ArrhythmiaClass.NORMAL,) else G
            rr = getattr(a, "rr_irr", None)
            pf = getattr(a, "p_frac", None) if hasattr(a, "p_frac") else getattr(a, "p_wave_fraction", None)
            print(f"  Class:  {ac}{B}{a.cls.value.upper()}{R}  Confidence: {a.confidence:.2f}")
            if getattr(a, "hr_bpm", None) is not None:
                rr_txt = "NA" if rr is None else f"{rr:.3f}"
                pf_txt = "NA" if pf is None else f"{pf:.2f}"
                print(f"  HR: {a.hr_bpm:.0f} bpm  RR-irr: {rr_txt}  P-wave frac: {pf_txt}")
            print(f"  {DM}Reason: {getattr(a,'reason','')[:70]}{R}")

    print(f"\n{B}  TASK B — Drowsiness / Fatigue{R}")
    if rs.drowsiness:
        d = rs.drowsiness
        if getattr(d, "abstained", False):
            print(f"  {DM}ABSTAIN: {getattr(d,'reason','')}{R}")
        else:
            dc = RE if d.score > 0.65 else Y if d.score > 0.40 else G
            print(f"  Score: {bar(d.score)} {dc}{d.score:.2f}{R}  Confidence: {d.confidence:.2f}")
            contribs = [(k, v) for k, v in [
                ("HR", getattr(d, "hr_contrib", 0.0)),
                ("HRV", getattr(d, "hrv_contrib", 0.0)),
                ("Resp", getattr(d, "resp_contrib", 0.0)),
                ("EDA", getattr(d, "eda_contrib", 0.0)),
                ("IMU", getattr(d, "imu_contrib", 0.0)),
            ] if v > 0.1]
            if contribs:
                print(f"  {DM}Contributors: {', '.join(f'{k}={v:.2f}' for k, v in contribs)}{R}")

    print(f"\n{B}  TASK C — Crash Detection{R}")
    if rs.crash:
        c_ = rs.crash
        if getattr(c_, "detected", False):
            sev = getattr(c_, "severity", None)
            sev_val = getattr(sev, "value", 0) if sev is not None else 0
            cc = RE if sev_val >= 2 else Y
            print(
                f"  {cc}{B}CRASH DETECTED — {getattr(sev,'name','UNKNOWN')}{R}  "
                f"g={getattr(c_,'g_peak',0):.1f}  conf={c_.confidence:.2f}  latency={getattr(c_,'latency_ms',0):.0f}ms"
            )
        else:
            print(f"  {G}No crash: {getattr(c_,'reason','')[:60]}{R}")

    print(f"\n{B}  POLICY ACTION{R}")
    lc = BRE + B if action.level == AlertLevel.ESCALATE else RE if action.level.value >= 4 else Y if action.level.value >= 3 else G
    print(f"  {lc}[{action.level.name}]{R}  {action.display_message}")
    if action.escalate_911:
        print(f"  {RE}{B}→ Escalation triggered (contacts + location payload){R}")
    if action.hospital_advisory:
        print(f"  {Y}→ Hospital advisory: navigate to nearest ER{R}")
    print(f"  {DM}Persist: {action.persistence_sec:.1f}s  Corroborated: {action.corroborated_by}{R}")
    print(f"  {DM}Reason:  {action.log_reason[:75]}{R}")
    print(DM + "━" * 70 + R)


def main() -> None:
    p = argparse.ArgumentParser(description="Guardian Drive™ v4.1")
    p.add_argument("--scenario", default="normal", choices=list(SCENARIO_PARAMS))
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--window", type=float, default=30.0)
    p.add_argument("--step", type=float, default=8.0)
    p.add_argument("--list", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--log", default="", help="Write JSONL event log to this path")
    p.add_argument("--dispatch", action="store_true", help="Send dispatch + contact messages on ESCALATE")

    p.add_argument("--telephony", choices=["console", "twilio", "discord"], default="console")
    p.add_argument("--nav", choices=["local", "osrm"], default="local")

    p.add_argument("--gps", choices=["mock", "manual", "phone"], default="mock")
    p.add_argument("--lat", type=float, default=None, help="Manual GPS latitude (required if --gps manual)")
    p.add_argument("--lon", type=float, default=None, help="Manual GPS longitude (required if --gps manual)")
    p.add_argument("--gps-server", default="http://127.0.0.1:8000", help="Server base URL (needed for --gps phone)")

    p.add_argument("--vehicle", choices=["noop", "obd"], default="noop")
    p.add_argument("--obd-port", default=None, help="OBD serial port (e.g. /dev/tty.usbserial-xxxx)")

    p.add_argument(
        "--seat-from-ecg",
        action="store_true",
        help="Route existing ECG windows through the new seat ECG path for bench testing.",
    )

    p.add_argument("--dispatch-min-interval", type=float, default=60.0,
                   help="Seconds between dispatch sends while ESCALATE persists")
    args = p.parse_args()

    if args.list:
        print("\nScenarios:")
        for s, prm in SCENARIO_PARAMS.items():
            print(f"  {s:<16} HR={prm['hr']:<5} label={prm['label'].value}")
        return

    if args.eval:
        from evaluation.runner import run_evaluation
        run_evaluation(save_json=True)
        return

    fusion = FusionEngine()
    sm = SafetyStateMachine()
    logger = EventLogger(Path(args.log)) if args.log else None
    limiter = DispatchLimiter(args.dispatch_min_interval)

    gps = _make_gps(args)
    nav = _make_nav(args)
    tel = _make_telephony(args)
    veh = _make_vehicle(args)

    sim = GuardianSimulator(args.scenario, args.duration, inject_artifacts=False)

    print(f"\n{B}{C}Guardian Drive™ v4.1 — Initializing {args.scenario.upper()}...{R}\n")
    if args.seat_from_ecg:
        print(f"{Y}[bench] --seat-from-ecg enabled: legacy ECG will be routed through seat ECG path.{R}\n")
    time.sleep(0.25)
    t0 = time.monotonic()

    try:
        for i, frame in enumerate(sim.stream(args.window, args.step)):
            if args.seat_from_ecg and getattr(frame, "ecg", None) is not None:
                frame.seat_ecg = np.asarray(frame.ecg, dtype=np.float32).copy()
                frame.seat_ecg_fs_hz = float(FS_ECG)
                frame.seat_ecg_meta = {
                    "contact_ok": True,
                    "motion_score": 0.0,
                    "source": "bench_from_legacy_ecg",
                }

            sqi = compute_sqi(frame)
            fb = extract_features(frame, sqi, args.window)
            rs = fusion.run(fb)
            action = _run_policy(sm, rs)

            fix = gps.get_fix()
            advisory = nav.nearest_er(fix) if fix else None

            if args.dispatch and action.escalate_911:
                now = time.monotonic()
                if limiter.allow(now):
                    gps_payload = None if not fix else {
                        "lat": fix.point.lat,
                        "lon": fix.point.lon,
                        "acc_m": fix.accuracy_m,
                        "maps": _maps_url(fix.point.lat, fix.point.lon),
                    }
                    er_payload = None if not advisory else {
                        "name": advisory.destination_name,
                        "eta_sec": advisory.eta_sec,
                        "distance_m": advisory.distance_m,
                        "provider": getattr(advisory, "provider", "unknown"),
                    }
                    tel.dispatch_simulation(message=DispatchMessage(
                        title="GUARDIAN DRIVE — ESCALATION",
                        body=f"scenario={args.scenario} action={action.level.name} reason={action.log_reason}",
                        meta={
                            "gps": gps_payload,
                            "er": er_payload,
                            "vehicle": veh.snapshot(),
                            "ecg_source": str(getattr(fb.ecg, "source", "") or ""),
                        },
                    ))

            if logger is not None:
                logger.write("window", {
                    "i": i + 1,
                    "scenario": args.scenario,
                    "t": time.monotonic() - t0,
                    "sqi": fb.sqi.to_dict(),
                    "task_a": None if not rs.arrhythmia else rs.arrhythmia.to_dict(),
                    "task_b": None if not rs.drowsiness else rs.drowsiness.to_dict(),
                    "task_c": None if not rs.crash else rs.crash.to_dict(),
                    "action": action.to_dict(),
                    "gps": None if not fix else {"lat": fix.point.lat, "lon": fix.point.lon, "acc_m": fix.accuracy_m},
                    "route": None if not advisory else {
                        "dest": advisory.destination_name,
                        "eta_sec": advisory.eta_sec,
                        "distance_m": advisory.distance_m,
                        "provider": getattr(advisory, "provider", "unknown")
                    },
                    "vehicle": veh.snapshot(),
                    "ecg_source": str(getattr(fb.ecg, "source", "") or ""),
                    "seat_status": dict(getattr(fb, "seat_ecg_status", {}) or {}),
                })

            render(fb, rs, action, i + 1, args.scenario, time.monotonic() - t0)
            time.sleep(0.4)

    finally:
        if hasattr(veh, "close"):
            try:
                veh.close()
            except Exception:
                pass

    print(f"\n{G}{B}Run complete.{R}\n")


if __name__ == "__main__":
    main()
