from __future__ import annotations

"""
Train Task C (Crash) model on *simulated* data.

Usage:
  python -m models.train_task_c --out artifacts/task_c_model.joblib --report reports/task_c_train.json
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

from acquisition.simulator import GuardianSimulator, SCENARIO_PARAMS
from sqi.compute import compute_sqi
from features.extract import extract_features
from acquisition.models import TaskLabel

from .model_utils import ensure_parent


def _get(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default


def _vec(fb):
    imu = fb.imu
    g = float(_get(imu, ["crash_g_peak", "g_peak", "g"], np.nan))
    j = float(_get(imu, ["crash_jerk", "jerk"], np.nan))
    m = float(_get(imu, ["motion_rms", "motion"], np.nan))
    return [g, j, m]


def build_dataset(subjects, duration_sec, window_sec, step_sec):
    X, y, groups = [], [], []
    for subj in subjects:
        for sc in SCENARIO_PARAMS:
            sim = GuardianSimulator(sc, duration_sec, subject_id=subj, inject_artifacts=(sc == "artifact"))
            for frame in sim.stream(win=window_sec, step=step_sec):
                sqi = compute_sqi(frame)
                fb = extract_features(frame, sqi, window_sec)
                if fb.label not in (TaskLabel.NORMAL, TaskLabel.CRASH_MILD, TaskLabel.CRASH_SEVERE):
                    continue
                X.append(_vec(fb))
                y.append(1 if fb.label in (TaskLabel.CRASH_MILD, TaskLabel.CRASH_SEVERE) else 0)
                groups.append(subj)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(groups)


def evaluate_loso(X, y, groups, clf):
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    y_true, y_prob, y_pred = [], [], []
    for tr, te in gkf.split(X, y, groups):
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        y_true.extend(y[te].tolist())
        y_prob.extend(prob.tolist())
        y_pred.extend(pred.tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)

    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "auc": auc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "n": int(len(y_true)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/task_c_model.joblib")
    ap.add_argument("--report", default="reports/task_c_train.json")
    ap.add_argument("--subjects", type=int, default=12)
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--step", type=float, default=10.0)
    args = ap.parse_args()

    subjects = [f"sim_{i:02d}" for i in range(args.subjects)]
    t0 = time.time()
    X, y, groups = build_dataset(subjects, args.duration, args.window, args.step)

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    metrics = evaluate_loso(X, y, groups, clf)

    # Fit final model
    clf.fit(X, y)

    outp = Path(args.out)
    ensure_parent(outp)
    joblib.dump(clf, outp)

    rep = {
        "task": "task_c",
        "note": "Trained on SIMULATED data only",
        "features": ["crash_g_peak", "crash_jerk", "motion_rms"],
        "dataset": {"subjects": len(subjects), "rows": int(len(y)), "pos": int(y.sum()), "neg": int((1-y).sum())},
        "loso_metrics": metrics,
        "saved_model": str(outp),
        "elapsed_sec": float(time.time() - t0),
    }

    rep_path = Path(args.report)
    ensure_parent(rep_path)
    rep_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"Saved model: {outp}")
    print(f"Saved report: {rep_path}")
    print(json.dumps(rep["loso_metrics"], indent=2))


if __name__ == "__main__":
    main()
