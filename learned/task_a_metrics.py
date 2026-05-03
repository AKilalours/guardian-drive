"""
learned/task_a_metrics.py
Guardian Drive -- Task A Arrhythmia Screening Metrics

Runs patient-level evaluation on PTB-XL / PTBDB dataset.
Produces per-class AUROC, sensitivity, specificity, confusion matrix.

Dataset: datasets/ptbdb/ (available locally on Mac)
Classes: Normal vs Abnormal (binary) -- ptbdb format

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""

import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def load_ptbdb(data_dir: str = "datasets/ptbdb") -> tuple:
    """
    Load PTBDB dataset.
    Format: CSV files with ECG signals + labels
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"PTBDB not found at {data_dir}")

    print(f"Loading PTBDB from {data_path}")
    all_files = list(data_path.rglob("*.csv"))
    print(f"Found {len(all_files)} CSV files")

    X_list, y_list, patient_ids = [], [], []

    for i, f in enumerate(all_files[:500]):  # limit for speed
        try:
            import pandas as pd
            df = pd.read_csv(f, header=None)
            if df.shape[1] < 188:
                continue
            # Last column is label (1=normal, 0=abnormal in ptbdb format)
            signal = df.iloc[:, :-1].values
            label  = int(df.iloc[0, -1])
            # Use first 180 samples as features
            for row in signal[:5]:
                if len(row) >= 180:
                    X_list.append(row[:180])
                    y_list.append(label)
                    patient_ids.append(i)  # same patient = same id
        except Exception:
            continue

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)
    g = np.array(patient_ids, dtype=int)
    print(f"Loaded: {len(X)} windows | "
          f"normal={y.sum()} abnormal={(y==0).sum()}")
    return X, y, g


def extract_hrv_features(X: np.ndarray) -> np.ndarray:
    """Extract HRV-style features from ECG windows."""
    features = []
    for window in X:
        # Basic statistical features
        feat = [
            window.mean(),
            window.std(),
            np.percentile(window, 25),
            np.percentile(window, 75),
            np.max(window) - np.min(window),
            np.abs(np.diff(window)).mean(),   # mean absolute diff
            np.abs(np.diff(window)).std(),    # std of diff
            np.sqrt(np.mean(np.diff(window)**2)),  # RMSSD proxy
            np.mean(window > 0),              # positive fraction
            np.corrcoef(window[:-1], window[1:])[0,1],  # lag-1 autocorr
        ]
        # FFT features (10 bins)
        fft = np.abs(np.fft.rfft(window))[:10]
        fft = fft / (fft.sum() + 1e-6)
        feat.extend(fft.tolist())
        features.append(feat)
    return np.array(features, dtype=np.float32)


def run_patient_level_cv(X: np.ndarray, y: np.ndarray,
                           groups: np.ndarray,
                           n_splits: int = 5) -> dict:
    """
    Patient-level stratified cross-validation.
    No patient leaks between train and test.
    """
    print(f"\nRunning {n_splits}-fold patient-level CV...")
    print("Extracting features...")
    X_feat = extract_hrv_features(X)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                               random_state=42)
    all_preds, all_trues = [], []
    fold_aucs = []

    for fold, (tr_idx, te_idx) in enumerate(
            cv.split(X_feat, y, groups)):
        X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
        y_tr, y_te = y[tr_idx],      y[te_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]

        if len(np.unique(y_te)) == 2:
            auc = roc_auc_score(y_te, probs)
            fold_aucs.append(auc)
            all_preds.extend(probs)
            all_trues.extend(y_te)
            print(f"  Fold {fold+1}: AUC={auc:.4f} "
                  f"(test={len(y_te)}, "
                  f"normal={y_te.sum()}, "
                  f"abnormal={(y_te==0).sum()})")

    # Aggregate metrics
    all_preds  = np.array(all_preds)
    all_trues  = np.array(all_trues)
    preds_bin  = (all_preds > 0.5).astype(int)

    overall_auc = roc_auc_score(all_trues, all_preds)
    cm          = confusion_matrix(all_trues, preds_bin)
    tn,fp,fn,tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    ppv         = tp / (tp + fp + 1e-6)
    npv         = tn / (tn + fn + 1e-6)

    result = {
        "task":          "Task A Arrhythmia Screening",
        "dataset":       "PTBDB (binary: normal vs abnormal)",
        "evaluation":    "Patient-level stratified k-fold CV",
        "n_splits":      n_splits,
        "n_samples":     len(all_trues),
        "fold_aucs":     [round(a, 4) for a in fold_aucs],
        "mean_auc":      round(float(np.mean(fold_aucs)), 4),
        "std_auc":       round(float(np.std(fold_aucs)), 4),
        "overall_auc":   round(overall_auc, 4),
        "sensitivity":   round(sensitivity, 4),
        "specificity":   round(specificity, 4),
        "ppv":           round(ppv, 4),
        "npv":           round(npv, 4),
        "confusion_matrix": cm.tolist(),
        "model":         "RandomForest (200 trees, balanced)",
        "features":      "20 HRV-style features per window",
        "note":          "Binary normal/abnormal. Not multi-class AUROC. "
                         "Not validated for clinical use.",
        "authors":       "Akilan Manivannan & Akila Lourdes Miriyala Francis"
    }
    return result


if __name__ == "__main__":
    print("Guardian Drive -- Task A Metrics")
    print("=" * 50)

    try:
        X, y, groups = load_ptbdb("datasets/ptbdb")
        result = run_patient_level_cv(X, y, groups, n_splits=5)

        print(f"\nTask A Results:")
        print(f"  Mean AUC:    {result['mean_auc']} +/- {result['std_auc']}")
        print(f"  Overall AUC: {result['overall_auc']}")
        print(f"  Sensitivity: {result['sensitivity']}")
        print(f"  Specificity: {result['specificity']}")
        print(f"  PPV:         {result['ppv']}")
        print(f"  NPV:         {result['npv']}")

        Path("learned/results").mkdir(exist_ok=True)
        Path("learned/results/task_a_metrics.json").write_text(
            json.dumps(result, indent=2))
        print("\nSaved: learned/results/task_a_metrics.json")
        print(json.dumps(result, indent=2))

    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Run from guardian-drive/ directory")
        print("PTBDB should be at datasets/ptbdb/")
