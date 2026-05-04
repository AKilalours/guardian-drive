"""
tools/replay_scenario.py
Guardian Drive -- Fault Injection Replay Harness

Tests safety state machine behavior under sensor failures.
Each scenario defines inputs + expected state transitions.

Usage:
    python tools/replay_scenario.py scenarios/ecg_dropout.json

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""
import json, sys, numpy as np
from pathlib import Path

def sqi_numpy(window, thresholds=(0.5,0.05,0.1,0.1)):
    q=[min(1.0,float(np.std(window[c]))/thresholds[c])
       for c in range(4)]
    return {"total":min(1.0,0.5*q[0]+0.3*q[1]+0.2*q[2])}

def state_machine(r, thresh):
    if r < thresh:           return "NOMINAL"
    elif r < thresh+0.20:    return "ADVISORY"
    elif r < thresh+0.40:    return "CAUTION"
    elif r < thresh+0.60:    return "PULLOVER"
    else:                    return "ESCALATE"

def run_scenario(scenario_path: str) -> dict:
    scenario = json.loads(Path(scenario_path).read_text())
    print(f"\nScenario: {scenario['name']}")
    print(f"Description: {scenario.get('description','')}")

    results = []
    passed  = True

    for step in scenario["steps"]:
        t         = step["t"]
        fault     = step.get("fault", None)
        tcn_prob  = step.get("tcn_prob", 0.5)
        imu_g     = step.get("imu_g", 0.1)
        hrv_rmssd = step.get("hrv_rmssd", 45.0)
        expected  = step.get("expected_state")

        # Generate window with fault injection
        window = np.random.randn(4, 4200).astype(np.float32)
        if fault == "ecg_dropout":
            window[0] = 0.0  # flat ECG
        elif fault == "low_sqi":
            window *= 0.001  # near-zero signal
        elif fault == "camera_occlusion":
            tcn_prob = 0.5   # uncertain
        elif fault == "microsleep":
            tcn_prob = 0.92  # high low-arousal prob
            window[0] *= 0.3 # degraded ECG

        sqi = sqi_numpy(window)

        if sqi["total"] < 0.30:
            state = "ABSTAIN"
        else:
            r_phys  = sqi["total"] * tcn_prob
            r_imu   = min(1.0, imu_g/3.0)
            r_neuro = max(0.0, 1.0-min(hrv_rmssd/50,1.0))
            r_ctx   = 0.1
            r_total = (0.40*r_phys + 0.20*r_imu +
                       0.10*r_ctx  + 0.30*r_neuro)
            state = state_machine(r_total, 0.35)

        ok = (state == expected) if expected else True
        if not ok:
            passed = False

        results.append({
            "t":        t,
            "fault":    fault,
            "sqi":      round(sqi["total"],3),
            "state":    state,
            "expected": expected,
            "passed":   ok,
        })
        status = "PASS" if ok else "FAIL"
        print(f"  t={t}s fault={fault or 'none':20s} "
              f"state={state:12s} expected={expected or 'any':12s} {status}")

    return {"scenario": scenario["name"],
            "passed": passed,
            "steps": results}

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)>1 else \
           "scenarios/ecg_dropout.json"
    result = run_scenario(path)
    print(f"\nResult: {'PASSED' if result['passed'] else 'FAILED'}")
