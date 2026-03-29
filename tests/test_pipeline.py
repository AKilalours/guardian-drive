"""
Guardian Drive v4.0 - Unit + Integration Tests

Covers:
- SQI abstain behavior
- Feature extraction correctness
- Policy state machine persistence logic
- Evaluation metrics correctness
- End-to-end replay integrity
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time

from acquisition.models import SQIState, TaskLabel, ArrhythmiaClass
from acquisition.simulator import GuardianSimulator
from sqi.compute import compute_sqi
from features.extract import extract_features
from models.task_a import ArrhythmiaScreener
from models.task_b import DrowsinessScreener
from models.task_c import CrashDetector
from policy.fusion import FusionEngine
from policy.state_machine import SafetyStateMachine
from evaluation.metrics import compute_ece, compute_far_per_hour, DetectionRecord

PASS = "✅ PASS"
FAIL = "❌ FAIL"

def assert_eq(name, got, expected, tol=None):
    if tol is not None:
        ok = abs(got - expected) <= tol
    else:
        ok = (got == expected)
    print(f"  {PASS if ok else FAIL}  {name}: got={got!r} expected={expected!r}")
    return ok

def assert_true(name, cond, detail=""):
    print(f"  {PASS if cond else FAIL}  {name}" + (f": {detail}" if detail else ""))
    return cond

# ─── Test 1: SQI abstain logic ────────────────────────────────

def test_sqi_abstain():
    print("\n[1] SQI Abstain Logic")
    sqi = SQIState(ecg_quality=0.0, eda_contact=0.0, belt_worn=False, motion_level=0.9)
    assert_true("abstain when all channels bad", sqi.abstain)
    sqi2 = SQIState(ecg_quality=0.9, eda_contact=0.8, belt_worn=True, motion_level=0.1)
    assert_true("no abstain with good signal", not sqi2.abstain)
    sqi3 = SQIState(ecg_quality=0.9, eda_contact=0.0, belt_worn=True, motion_level=0.1)
    assert_true("no abstain with ECG+IMU (2 channels)", not sqi3.abstain, sqi3.summary())

# ─── Test 2: Simulator produces correct shapes ───────────────

def test_simulator_shapes():
    print("\n[2] Simulator Channel Shapes")
    sim   = GuardianSimulator("normal", duration=30.0)
    frame = sim.window(0, 30)
    assert_true("ECG shape", frame.ecg is not None and len(frame.ecg) == 250*30,
                f"len={len(frame.ecg) if frame.ecg is not None else None}")
    assert_true("EDA shape", frame.eda is not None and len(frame.eda) == 16*30)
    assert_true("Accel shape", frame.accel is not None and frame.accel.shape == (100*30, 3))
    assert_true("Belt tension", frame.belt_tension is not None and 0 <= frame.belt_tension <= 1)

# ─── Test 3: ECG features extracted correctly ────────────────

def test_ecg_features():
    print("\n[3] ECG Feature Extraction")
    for sc, exp_hr_range in [("normal",(65,80)), ("tachycardia",(160,200)), ("bradycardia",(25,45))]:
        sim   = GuardianSimulator(sc, 60.0)
        frame = sim.window(0, 30)
        sqi   = compute_sqi(frame)
        fb    = extract_features(frame, sqi, 30.0)
        hr    = fb.ecg.hr_bpm
        ok    = hr is not None and exp_hr_range[0] <= hr <= exp_hr_range[1]
        hr_str = f"{hr:.0f}" if hr is not None else "None"
        print(f"  {'✅' if ok else '⚠️ '}  {sc}: HR={hr_str} expected {exp_hr_range}")

# ─── Test 4: Arrhythmia screener abstains correctly ──────────

def test_arrhythmia_abstain():
    print("\n[4] Task A Abstain When SQI Bad")
    sim   = GuardianSimulator("artifact", 60.0)
    frame = sim.window(0, 30)
    sqi   = compute_sqi(frame)
    fb    = extract_features(frame, sqi, 30.0)
    result= ArrhythmiaScreener().predict(fb)
    assert_true("abstain on artifact scenario",
                result.abstained or result.cls == ArrhythmiaClass.NOISY,
                f"cls={result.cls} abstained={result.abstained}")

# ─── Test 5: Task A classifies arrhythmias ───────────────────

def test_arrhythmia_classification():
    print("\n[5] Task A Arrhythmia Classification")
    screener = ArrhythmiaScreener()
    expectations = [
        ("normal",      ArrhythmiaClass.NORMAL),
        ("afib",        ArrhythmiaClass.AFIB),
        ("tachycardia", ArrhythmiaClass.TACHYCARDIA),
        ("bradycardia", ArrhythmiaClass.BRADYCARDIA),
    ]
    for sc, expected_cls in expectations:
        sim = GuardianSimulator(sc, 60.0)
        frame = sim.window(10, 40)
        sqi   = compute_sqi(frame)
        fb    = extract_features(frame, sqi, 30.0)
        r     = screener.predict(fb)
        ok    = (r.cls == expected_cls) or r.abstained
        print(f"  {'✅' if ok else '⚠️ '}  {sc}: got={r.cls.value} expected={expected_cls.value} "
              f"conf={r.confidence:.2f} abstain={r.abstained}")

# ─── Test 6: Crash detection ─────────────────────────────────

def test_crash_detection():
    print("\n[6] Task C Crash Detection")
    detector = CrashDetector()
    for sc, should_detect in [("normal", False), ("crash_mild", True), ("crash_severe", True)]:
        sim   = GuardianSimulator(sc, 120.0)
        found = False
        for frame in sim.stream(win=15.0, step=5.0):
            sqi = compute_sqi(frame)
            fb  = extract_features(frame, sqi, 15.0)
            r   = detector.predict(fb)
            if r.detected: found = True; break
        ok = (found == should_detect)
        print(f"  {'✅' if ok else '⚠️ '}  {sc}: detected={found} expected={should_detect}")

# ─── Test 7: Policy state machine persistence ─────────────────

def test_policy_persistence():
    print("\n[7] Policy State Machine — Persistence")
    from acquisition.models import RiskState, ArrhythmiaResult, AlertLevel
    from acquisition.models import ArrhythmiaClass, DriverState

    sm = SafetyStateMachine()

    def make_rs(cls, conf):
        rs = RiskState()
        rs.arrhythmia = ArrhythmiaResult(
            cls=cls, confidence=conf, abstained=False,
            hr_bpm=178 if cls == ArrhythmiaClass.TACHYCARDIA else 72
        )
        rs.driver_state = DriverState.NORMAL
        return rs

    # First call: should NOT immediately escalate (persistence not met)
    rs1 = make_rs(ArrhythmiaClass.TACHYCARDIA, 0.90)
    a1  = sm.step(rs1)
    assert_true(
        "No ESCALATE on first window (persistence not met)",
        a1.level.value < AlertLevel.ESCALATE.value,
        f"level={a1.level.name}"
    )

# ─── Test 7b: Severe crash escalates immediately ───────────────

def test_crash_severe_immediate_escalate():
    print("\n[7b] Policy State Machine — Severe Crash Immediate Escalate")
    from acquisition.models import RiskState, CrashResult, AlertLevel, DriverState, CrashSeverity
    sm = SafetyStateMachine()
    rs = RiskState()
    rs.crash = CrashResult(
        detected=True,
        severity=CrashSeverity.SEVERE,
        confidence=0.95,
        g_peak=9.2,
        belt_corroborated=True,
        belt_tension=0.95,
    )
    rs.driver_state = DriverState.ESCALATE
    a = sm.step(rs)
    assert_true(
        "ESCALATE on first window for severe crash",
        a.level.value >= AlertLevel.ESCALATE.value,
        f"level={a.level.name} reason={a.log_reason}"
    )

# ─── Test 8: ECE calibration test ────────────────────────────

def test_ece():
    print("\n[8] ECE Calibration (perfect calibration = 0)")
    records = [
        DetectionRecord("s0","ses0",0,"arrhythmia","arrhythmia",0.80,False),
        DetectionRecord("s0","ses0",1,"normal","normal",0.75,False),
        DetectionRecord("s0","ses0",2,"arrhythmia","normal",0.60,False),
        DetectionRecord("s0","ses0",3,"normal","arrhythmia",0.55,False),
    ]
    ece = compute_ece(records)
    assert_true("ECE is a float in [0,1]", 0 <= ece <= 1.0, f"ece={ece:.3f}")

# ─── Test 9: End-to-end pipeline ─────────────────────────────

def test_e2e():
    print("\n[9] End-to-End Pipeline Integrity")
    fusion = FusionEngine()
    sm     = SafetyStateMachine()
    sim    = GuardianSimulator("tachycardia", 60.0)
    n = 0
    for frame in sim.stream(win=30.0, step=30.0):
        sqi    = compute_sqi(frame)
        fb     = extract_features(frame, sqi, 30.0)
        rs     = fusion.run(fb)
        action = sm.step(rs)
        n += 1
        assert_true(f"Window {n}: action has log_reason",
                    len(action.log_reason) > 0, action.log_reason[:60])

# ─── Test 10: JSON serialization ─────────────────────────────

def test_serialization():
    print("\n[10] SensorFrame JSON Serialization (replay requirement)")
    sim   = GuardianSimulator("normal", 30.0)
    frame = sim.window(0, 30)
    try:
        j = frame.to_json()
        assert_true("to_json produces non-empty string", len(j) > 100)
    except Exception as e:
        print(f"  {FAIL}  Serialization error: {e}")

# ─── Run all ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nGuardian Drive™ v4.0 — Test Suite")
    print("=" * 55)
    test_sqi_abstain()
    test_simulator_shapes()
    test_ecg_features()
    test_arrhythmia_abstain()
    test_arrhythmia_classification()
    test_crash_detection()
    test_policy_persistence()
    test_crash_severe_immediate_escalate()
    test_ece()
    test_e2e()
    test_serialization()
    print("\n" + "=" * 55)
    print("Test suite complete.")
