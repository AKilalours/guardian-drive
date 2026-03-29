"""Guardian Drive UI (Streamlit)

Run:
  streamlit run ui/streamlit_app.py

This UI supports:
- Live simulation (same pipeline as main.py)
- Replay of JSONL logs

It is intentionally "integration-ready": real sensors can be plugged into
acquisition sources later without changing the UI.
"""

from __future__ import annotations

from pathlib import Path
import time
import pandas as pd
import streamlit as st

from acquisition.simulator import GuardianSimulator, SCENARIO_PARAMS
from sqi.compute import compute_sqi
from features.extract import extract_features
from policy.fusion import FusionEngine
from policy.state_machine import SafetyStateMachine
from integrations.gps_mock import MockGPS
from integrations.navigation_local import LocalHospitalNav
from integrations.telephony_console import ConsoleTelephony
from integrations.vehicle_sim import NoOpVehicleControl
from integrations.base import DispatchMessage
from gd_logging.event_log import EventLogger
from replay.loader import load_window_summaries


st.set_page_config(page_title="Guardian Drive", layout="wide")


def _mk_integrations():
    gps = MockGPS()
    nav = LocalHospitalNav(Path("data/hospitals.csv"))
    tel = ConsoleTelephony()
    veh = NoOpVehicleControl()
    return gps, nav, tel, veh


def _render_live():
    st.header("Live Simulation")
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    with c1:
        scenario = st.selectbox("Scenario", options=list(SCENARIO_PARAMS.keys()), index=0)
    with c2:
        duration = st.number_input("Duration (s)", min_value=30, max_value=900, value=120, step=30)
    with c3:
        window = st.number_input("Window (s)", min_value=5, max_value=60, value=30, step=5)
    with c4:
        step = st.number_input("Step (s)", min_value=1, max_value=30, value=8, step=1)

    log_path = Path(st.text_input("Log path (JSONL)", value="runs/latest.jsonl"))
    do_log = st.checkbox("Write event log", value=True)

    run = st.button("Start")
    if not run:
        st.info("Select a scenario and click Start.")
        return

    fusion = FusionEngine()
    sm = SafetyStateMachine()
    sim = GuardianSimulator(scenario, float(duration), inject_artifacts=(scenario == "artifact"))

    gps, nav, tel, veh = _mk_integrations()
    logger = EventLogger(log_path) if do_log else None

    status = st.empty()
    grid = st.container()
    chart = st.empty()

    history = []
    t0 = time.time()

    for i, frame in enumerate(sim.stream(float(window), float(step))):
        sqi = compute_sqi(frame)
        fb = extract_features(frame, sqi, float(window))
        rs = fusion.run(fb)
        action = sm.step(rs)

        fix = gps.get_fix()
        advisory = nav.nearest_er(fix) if (fix and action.hospital_advisory) else None

        if action.escalate_911:
            # Simulation-only dispatch message
            msg = DispatchMessage(
                title="GUARDIAN DRIVE DISPATCH (SIMULATION)",
                body=f"Event={action.level.name} reason={action.log_reason}",
                meta={
                    "scenario": scenario,
                    "gps": None if not fix else {"lat": fix.point.lat, "lon": fix.point.lon, "acc_m": fix.accuracy_m},
                    "er": None if not advisory else {"name": advisory.destination_name, "lat": advisory.destination_point.lat, "lon": advisory.destination_point.lon, "eta_sec": advisory.eta_sec},
                },
            )
            tel.dispatch_simulation(message=msg)
            tel.notify_emergency_contact(message=msg.body, meta=msg.meta)

        if logger is not None:
            logger.write("window", {
                "i": i+1,
                "scenario": scenario,
                "t": time.time() - t0,
                "sqi": fb.sqi.to_dict(),
                "task_a": None if not rs.arrhythmia else rs.arrhythmia.to_dict(),
                "task_b": None if not rs.drowsiness else rs.drowsiness.to_dict(),
                "task_c": None if not rs.crash else rs.crash.to_dict(),
                "action": action.to_dict(),
                "gps": None if not fix else {"lat": fix.point.lat, "lon": fix.point.lon, "acc_m": fix.accuracy_m},
                "er": None if not advisory else {"name": advisory.destination_name, "eta_sec": advisory.eta_sec, "dist_m": advisory.distance_m},
            })

        # UI
        status.markdown(f"**Window {i+1}** · SQI={fb.sqi.overall_confidence:.2f} · Action=**{action.level.name}**")
        with grid:
            cols = st.columns(4)
            cols[0].metric("SQI", f"{fb.sqi.overall_confidence:.2f}", "ABSTAIN" if fb.sqi.abstain else "OK")
            if rs.arrhythmia and not rs.arrhythmia.abstained:
                cols[1].metric("Arrhythmia", rs.arrhythmia.cls.value, f"conf={rs.arrhythmia.confidence:.2f}")
            else:
                cols[1].metric("Arrhythmia", "ABSTAIN" if (rs.arrhythmia and rs.arrhythmia.abstained) else "—")

            if rs.drowsiness and not rs.drowsiness.abstained:
                cols[2].metric("Drowsy score", f"{rs.drowsiness.score:.2f}", f"conf={rs.drowsiness.confidence:.2f}")
            else:
                cols[2].metric("Drowsy score", "ABSTAIN" if (rs.drowsiness and rs.drowsiness.abstained) else "—")

            if rs.crash and rs.crash.detected:
                cols[3].metric("Crash", rs.crash.severity.name, f"g={rs.crash.g_peak:.1f}")
            else:
                cols[3].metric("Crash", "no")

        history.append({
            "t": time.time() - t0,
            "sqi": fb.sqi.overall_confidence,
            "drowsy": 0.0 if not (rs.drowsiness and not rs.drowsiness.abstained) else rs.drowsiness.score,
            "arrhythmia_conf": 0.0 if not (rs.arrhythmia and not rs.arrhythmia.abstained) else rs.arrhythmia.confidence,
        })
        df = pd.DataFrame(history).set_index("t")
        chart.line_chart(df[["sqi","drowsy","arrhythmia_conf"]])
        time.sleep(0.15)

    st.success("Simulation complete")


def _render_replay():
    st.header("Replay")
    path = Path(st.text_input("Replay JSONL", value="runs/latest.jsonl"))
    if not path.exists():
        st.warning("Log file not found.")
        return

    rows = load_window_summaries(path)
    if not rows:
        st.warning("No window events in this log.")
        return

    df = pd.json_normalize(rows)
    st.write(df.head(20))

    # plot some key signals
    if "t" in df.columns:
        df = df.sort_values("t")
        plot_cols = [c for c in ["sqi.overall_confidence", "task_b.score", "task_a.confidence"] if c in df.columns]
        if plot_cols:
            st.line_chart(df.set_index("t")[plot_cols])


def main():
    st.title("Guardian Drive™ — Engineering UI")
    tabs = st.tabs(["Live", "Replay", "Notes"])

    with tabs[0]:
        _render_live()

    with tabs[1]:
        _render_replay()

    with tabs[2]:
        st.markdown(
            """
### Reality check (non-negotiable)
- **Autopilot control** is not implemented here. Any real vehicle control must be OEM-approved.
- **911/PSAP integration** is not implemented. This UI uses **simulation mode** dispatch messages.
- **GPS** is mock by default. Replace with a real provider (gpsd/serial) when you have hardware.

What *is* real here:
- Deterministic pipeline, SQI gating, abstain behavior, traceable policy decisions
- Replayable logs for debugging and evaluation
"""
        )


if __name__ == "__main__":
    main()
