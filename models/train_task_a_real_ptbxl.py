import argparse
import ast
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import wfdb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Strong rhythm-abnormality positives only.
POS_CODES = {
    "AFIB",   # atrial fibrillation
    "AFLT",   # atrial flutter
    "SVTAC",  # supraventricular tachycardia
    "PSVT",   # paroxysmal supraventricular tachycardia
    "SVARR",  # supraventricular arrhythmia
    "PAC",    # atrial premature complex
    "PVC",    # ventricular premature complex
    "BIGU",   # bigeminal pattern
    "TRIGU",  # trigeminal pattern
}

# Clean negatives only.
NEG_CODES = {
    "NORM",
    "SR",
}


def parse_scp_codes(s: str) -> set[str]:
    if not isinstance(s, str) or not s.strip():
        return set()
    raw = ast.literal_eval(s)
    out = set()
    for k, v in raw.items():
        try:
            if float(v) > 0:
                out.add(str(k))
        except Exception:
            pass
    return out


def assign_label(codes: set[str]):
    if codes & POS_CODES:
        return 1
    if codes and codes.issubset(NEG_CODES):
        return 0
    return None


def resample_linear(x: np.ndarray, out_len: int) -> np.ndarray:
    if len(x) == out_len:
        return x.astype(np.float32, copy=False)
    xp = np.linspace(0.0, 1.0, num=len(x), dtype=np.float32)
    fp = x.astype(np.float32, copy=False)
    xnew = np.linspace(0.0, 1.0, num=out_len, dtype=np.float32)
    return np.interp(xnew, xp, fp).astype(np.float32)


def extract_features(record_base: Path) -> np.ndarray:
    sig, fields = wfdb.rdsamp(str(record_base))

    if sig.ndim != 2 or sig.shape[1] == 0:
        raise ValueError(f"Unexpected signal shape for {record_base}: {sig.shape}")

    # Prefer lead II for rhythm information.
    lead_idx = 1 if sig.shape[1] > 1 else 0
    x = sig[:, lead_idx].astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-record normalization.
    x = x - np.mean(x)
    std = np.std(x)
    if std < 1e-6:
        std = 1.0
    x = x / std

    # PTB-XL low-rate records are 10 s @ 100 Hz => ~1000 samples.
    # Force a fixed compact representation.
    x_250 = resample_linear(x, 250)

    # Frequency features.
    fft_mag = np.abs(np.fft.rfft(x_250))[:128].astype(np.float32)

    # Basic shape/change statistics.
    dx = np.diff(x_250)
    stats = np.array(
        [
            float(np.mean(x_250)),
            float(np.std(x_250)),
            float(np.min(x_250)),
            float(np.max(x_250)),
            float(np.percentile(x_250, 5)),
            float(np.percentile(x_250, 95)),
            float(np.mean(np.abs(dx))),
            float(np.std(dx)),
            float(np.max(np.abs(dx))) if len(dx) else 0.0,
        ],
        dtype=np.float32,
    )

    return np.concatenate([x_250, fft_mag, stats], axis=0)


def pick_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.20, 0.80, 61):
        pred = (probs >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="PTB-XL root containing ptbxl_database.csv")
    ap.add_argument("--out", default="artifacts/task_a_model_real_ptbxl.joblib")
    ap.add_argument("--report", default="reports/task_a_train_real_ptbxl.json")
    args = ap.parse_args()

    t0 = time.time()

    root = Path(args.root)
    meta_csv = root / "ptbxl_database.csv"
    scp_csv = root / "scp_statements.csv"

    if not meta_csv.exists():
        raise SystemExit(f"Missing metadata file: {meta_csv}")
    if not scp_csv.exists():
        raise SystemExit(f"Missing SCP file: {scp_csv}")

    df = pd.read_csv(meta_csv)
    if "scp_codes" not in df.columns or "filename_lr" not in df.columns or "strat_fold" not in df.columns:
        raise SystemExit("PTB-XL metadata missing required columns: scp_codes, filename_lr, strat_fold")

    df["codes"] = df["scp_codes"].apply(parse_scp_codes)
    df["label"] = df["codes"].apply(assign_label)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    # Use only low-rate records for speed and reproducibility.
    df = df[df["filename_lr"].notna()].copy()

    X_list = []
    y_list = []
    fold_list = []
    ecg_ids = []
    skipped = 0

    print(f"Candidate records after label filtering: {len(df)}")

    for i, row in enumerate(df.itertuples(index=False), start=1):
        rel = Path(row.filename_lr)
        record_base = root / rel
        try:
            feat = extract_features(record_base)
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"SKIP {record_base}: {e}")
            continue

        X_list.append(feat)
        y_list.append(int(row.label))
        fold_list.append(int(row.strat_fold))
        ecg_ids.append(int(row.ecg_id))

        if i % 500 == 0:
            print(f"Loaded {i}/{len(df)}")

    if not X_list:
        raise SystemExit("No usable PTB-XL records were loaded.")

    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    folds = np.asarray(fold_list, dtype=np.int64)

    train_mask = folds <= 8
    val_mask = folds == 9
    test_mask = folds == 10

    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise SystemExit("Bad split after filtering; train/val/test is empty.")

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    best_auc = -1.0
    best_c = None
    best_model = None

    for c in [0.1, 0.3, 1.0, 3.0]:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c,
                        max_iter=4000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        print(f"C={c:.3f}  val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_c = c
            best_model = model

    val_probs = best_model.predict_proba(X_val)[:, 1]
    best_thr = pick_threshold(y_val, val_probs)
    print(f"Best C={best_c}  Best val AUC={best_auc:.4f}  Best threshold={best_thr:.3f}")

    # Final fit on train+val, test on fold 10 only.
    trainval_mask = folds <= 9
    X_trainval, y_trainval = X[trainval_mask], y[trainval_mask]

    final_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=best_c,
                    max_iter=4000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    final_model.fit(X_trainval, y_trainval)

    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= best_thr).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average="binary", zero_division=0
    )
    report = {
        "task": "task_a",
        "note": "REAL PTB-XL rhythm-based Task A baseline. Positive = strong rhythm abnormality codes only; negative = clean normal/sinus only.",
        "dataset": {
            "source": "PTB-XL",
            "root": str(root),
            "loaded_records": int(len(X)),
            "skipped_records": int(skipped),
            "train_records": int(train_mask.sum()),
            "val_records": int(val_mask.sum()),
            "test_records": int(test_mask.sum()),
            "train_pos": int(y_train.sum()),
            "train_neg": int((1 - y_train).sum()),
            "val_pos": int(y_val.sum()),
            "val_neg": int((1 - y_val).sum()),
            "test_pos": int(y_test.sum()),
            "test_neg": int((1 - y_test).sum()),
        },
        "labels": {
            "positive_codes": sorted(POS_CODES),
            "negative_codes": sorted(NEG_CODES),
        },
        "features": {
            "lead": "II",
            "time_points": 250,
            "fft_bins": 128,
            "extra_stats": 9,
            "total_dim": int(X.shape[1]),
        },
        "selection": {
            "best_C": float(best_c),
            "best_val_auc": float(best_auc),
            "best_threshold": float(best_thr),
        },
        "test_metrics": {
            "auc": float(roc_auc_score(y_test, test_probs)),
            "pr_auc": float(average_precision_score(y_test, test_probs)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
            "n": int(len(y_test)),
        },
        "saved_model": args.out,
        "elapsed_sec": time.time() - t0,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path = Path(args.report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": final_model,
            "threshold": best_thr,
            "task": "task_a",
            "source": "PTB-XL",
            "positive_codes": sorted(POS_CODES),
            "negative_codes": sorted(NEG_CODES),
            "feature_spec": {
                "lead": "II",
                "time_points": 250,
                "fft_bins": 128,
                "extra_stats": 9,
            },
        },
        out_path,
    )

    rep_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\nSaved model:", out_path)
    print("Saved report:", rep_path)
    print(json.dumps(report["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
