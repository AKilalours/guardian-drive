"""
Guardian Drive™ v4.1 — Evaluation Runner

Runs all 3 tasks across all scenarios, collects records,
computes LOSO metrics, prints a full evaluation report.

Usage:
  python -m evaluation.runner
  python -m evaluation.runner --out reports --no-save
"""

from __future__ import annotations

import argparse, json, time
from dataclasses import is_dataclass, asdict
from pathlib import Path

import numpy as np

from acquisition.simulator import GuardianSimulator, SCENARIO_PARAMS
from sqi.compute import compute_sqi
from features.extract import extract_features
from policy.fusion import FusionEngine
from evaluation.metrics import DetectionRecord, loso_evaluation
from acquisition.models import TaskLabel, ArrhythmiaClass

# 10 simulated subjects, each 4 minutes, to get realistic LOSO splits
SUBJECTS     = [f"sim_{i:02d}" for i in range(10)]
DURATION_SEC = 240.0
WINDOW_SEC   = 30.0
STEP_SEC     = 10.0


def _run_all_windows(fusion: FusionEngine, subject_id: str):
    """Run all scenarios for one subject. Returns list of (fb, risk_state)."""
    results = []
    for sc in SCENARIO_PARAMS:
        sim = GuardianSimulator(sc, DURATION_SEC, subject_id=subject_id,
                                inject_artifacts=(sc == "artifact"))
        for frame in sim.stream(win=WINDOW_SEC, step=STEP_SEC):
            sqi  = compute_sqi(frame)
            fb   = extract_features(frame, sqi, WINDOW_SEC)
            rs   = fusion.run(fb)
            results.append((fb, rs))
    return results


def _arrhythmia_records(pairs):
    records = []
    for fb, rs in pairs:
        # Task A slice: arrhythmia vs normal/artifact only
        if fb.label not in (TaskLabel.NORMAL, TaskLabel.ARRHYTHMIA, TaskLabel.ARTIFACT):
            continue

        true_pos = fb.label in (TaskLabel.ARRHYTHMIA,)
        true_label = "arrhythmia" if true_pos else "normal"

        if rs.arrhythmia:
            pred_label = (
                "arrhythmia"
                if (not rs.arrhythmia.abstained) and rs.arrhythmia.cls not in (ArrhythmiaClass.NORMAL, ArrhythmiaClass.NOISY)
                else "normal"
            )
            conf = float(getattr(rs.arrhythmia, "confidence", 0.0) or 0.0)
            records.append(DetectionRecord(
                subject_id=fb.subject_id,
                session_id=fb.session_id,
                window_start=fb.timestamp,
                true_label=true_label,
                pred_label=pred_label,
                confidence=conf,
                abstained=bool(getattr(rs.arrhythmia, "abstained", False)),
                score=(conf if pred_label == "arrhythmia" else 1.0 - conf),
            ))
    return records


def _drowsiness_records(pairs):
    records = []
    for fb, rs in pairs:
        # Task B slice: alert vs drowsy/fatigued (exclude arrhythmia/crash scenarios)
        if fb.label not in (TaskLabel.NORMAL, TaskLabel.DROWSY, TaskLabel.FATIGUED):
            continue

        true_pos = fb.label in (TaskLabel.DROWSY, TaskLabel.FATIGUED)
        true_label = "drowsy" if true_pos else "alert"

        if rs.drowsiness:
            pred_label = "drowsy" if rs.drowsiness.score > 0.50 else "alert"
            records.append(DetectionRecord(
                subject_id=fb.subject_id,
                session_id=fb.session_id,
                window_start=fb.timestamp,
                true_label=true_label,
                pred_label=pred_label,
                confidence=float(rs.drowsiness.confidence),
                abstained=bool(rs.drowsiness.abstained),
                score=float(np.clip(rs.drowsiness.score, 0.0, 1.0)),
            ))
    return records


def _crash_records(pairs):
    records = []
    for fb, rs in pairs:
        # Task C slice: crash vs no_crash only
        if fb.label not in (TaskLabel.NORMAL, TaskLabel.CRASH_MILD, TaskLabel.CRASH_SEVERE):
            continue

        true_label = "crash" if fb.label in (TaskLabel.CRASH_MILD, TaskLabel.CRASH_SEVERE) else "no_crash"

        if rs.crash:
            pred_label = "crash" if rs.crash.detected else "no_crash"
            records.append(DetectionRecord(
                subject_id=fb.subject_id,
                session_id=fb.session_id,
                window_start=fb.timestamp,
                true_label=true_label,
                pred_label=pred_label,
                confidence=float(rs.crash.confidence),
                abstained=False,
                score=float(np.clip(rs.crash.confidence, 0.0, 1.0)),
                latency_ms=float(getattr(rs.crash, "latency_ms", 0.0) if rs.crash.detected else 0.0),
            ))
    return records


def _jsonable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def run_evaluation(out_dir: str = "reports", save_json: bool = True) -> dict:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    print("\nGuardian Drive™ — Evaluation Runner")
    print("=" * 65)
    print(f"Subjects: {len(SUBJECTS)}  Duration: {DURATION_SEC}s  Window: {WINDOW_SEC}s  Step: {STEP_SEC}s")
    print(f"Windows per subject: ~{int((DURATION_SEC-WINDOW_SEC)/STEP_SEC + 1) * len(SCENARIO_PARAMS)}")
    print()

    fusion = FusionEngine()  # will auto-load trained models if artifacts are present
    all_a, all_b, all_c = [], [], []
    t0 = time.monotonic()

    for subj in SUBJECTS:
        pairs = _run_all_windows(fusion, subj)
        all_a.extend(_arrhythmia_records(pairs))
        all_b.extend(_drowsiness_records(pairs))
        all_c.extend(_crash_records(pairs))
        print(f"  Subject {subj}: {len(pairs)} windows processed")

    elapsed = time.monotonic() - t0
    print(f"\nAll windows processed in {elapsed:.1f}s\n")
    print("=" * 65)

    rep_a = loso_evaluation(all_a, "arrhythmia", "Task A — Arrhythmia Screening", WINDOW_SEC)
    rep_b = loso_evaluation(all_b, "drowsy",     "Task B — Drowsiness Screening", WINDOW_SEC)
    rep_c = loso_evaluation(all_c, "crash",      "Task C — Crash Detection",       WINDOW_SEC)

    for rep in (rep_a, rep_b, rep_c):
        print(f"\n{'─'*65}")
        print(rep.summary())

    print(f"\n{'─'*65}")
    print("ACCEPTANCE CRITERIA CHECK:")
    checks = [
        ("Task A Sensitivity >= 0.70", rep_a.sensitivity >= 0.70),
        ("Task A FAR/hr <= 5.00",       rep_a.far_per_hour <= 5.00),
        ("Task A ECE <= 0.15",          rep_a.ece <= 0.15),
        ("Task B Sensitivity >= 0.65",  rep_b.sensitivity >= 0.65),
        ("Task B FAR/hr <= 3.00",       rep_b.far_per_hour <= 3.00),
        ("Task C Sensitivity >= 0.90",  rep_c.sensitivity >= 0.90),
        ("Task C FAR/hr <= 1.00",       rep_c.far_per_hour <= 1.00),
    ]
    for name, passed in checks:
        print(f"  {'✅' if passed else '❌'}  {name}")

    out = {"task_a": rep_a, "task_b": rep_b, "task_c": rep_c}

    if save_json:
        (outp / "evaluation_report.json").write_text(
            json.dumps({k: _jsonable(v) for k, v in out.items()}, indent=2),
            encoding="utf-8"
        )
        print(f"\nSaved: {outp/'evaluation_report.json'}")

    return out


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports", help="Output directory for JSON report")
    ap.add_argument("--no-save", action="store_true", help="Do not write JSON report")
    args = ap.parse_args()
    run_evaluation(out_dir=args.out, save_json=not args.no_save)


if __name__ == "__main__":
    _cli()
