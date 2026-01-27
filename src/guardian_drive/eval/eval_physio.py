from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from guardian_drive.data.windows import load_processed, preprocess_wesad_subject, save_processed
from guardian_drive.models.physio_cnn import PhysioCNN1D
from guardian_drive.utils.config import ensure_dir, load_yaml
from guardian_drive.utils.seed import set_seed

console = Console()


def _as_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(s) for s in x]
    return [str(x)]


def _channel_names(cfg: dict) -> list[str]:
    use_chest = bool(cfg["preprocess"]["use_chest"])
    use_wrist = bool(cfg["preprocess"]["use_wrist"])
    names: list[str] = []
    if use_chest:
        names += list(cfg["preprocess"]["signals"]["chest"])
    if use_wrist:
        names += list(cfg["preprocess"]["signals"].get("wrist", []))
    return names


def _cache_path(cfg: dict, subject: str) -> Path:
    sr = int(cfg["preprocess"]["resample_hz"])
    win = float(cfg["preprocess"]["window_sec"])
    stride = float(cfg["preprocess"]["stride_sec"])
    names = "_".join(_channel_names(cfg))
    root = Path("data/processed/wesad_physio")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{subject}_sr{sr}_win{win:g}_stride{stride:g}_{names}.npz"


def _map_binary_labels(
    y_raw: np.ndarray, stress_label: int, nonstress_labels: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isin(y_raw, [stress_label] + list(nonstress_labels))
    y = y_raw[keep]
    y = (y == stress_label).astype(np.int64)
    return y, keep


def _load_or_build_subject(cfg: dict, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = _cache_path(cfg, subject)
    names = _channel_names(cfg)

    if cache.exists():
        z = load_processed(cache)
        X = z["X"].astype(np.float32)
        y = z["y"].astype(np.int64)
        sqi = z["sqi"].astype(np.float32)
    else:
        ps = preprocess_wesad_subject(
            wesad_root=cfg["data"]["wesad_root"],
            subject=subject,
            use_chest=cfg["preprocess"]["use_chest"],
            use_wrist=cfg["preprocess"]["use_wrist"],
            chest_signals=cfg["preprocess"]["signals"]["chest"],
            wrist_signals=cfg["preprocess"]["signals"].get("wrist", []),
            resample_hz=cfg["preprocess"]["resample_hz"],
            window_sec=cfg["preprocess"]["window_sec"],
            stride_sec=cfg["preprocess"]["stride_sec"],
            label_strategy=cfg["preprocess"]["label_strategy"],
            sqi_rules=cfg["quality"]["rules"],
        )
        save_processed(ps, cache, channel_names=names, sr_hz=int(cfg["preprocess"]["resample_hz"]))
        X, y, sqi = ps.X, ps.y, ps.sqi

    stress_label = int(cfg["data"]["task"]["stress_label"])
    nonstress_labels = list(cfg["data"]["task"]["nonstress_labels"])
    y_mapped, keep = _map_binary_labels(y, stress_label, nonstress_labels)

    X = X[keep]
    sqi = sqi[keep]
    return X, y_mapped, sqi


def _normalize_batch(
    X: np.ndarray, norm_mode: str, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    nm = str(norm_mode).lower()
    if X.size == 0:
        return X.astype(np.float32)

    if nm == "global":
        std_safe = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        return ((X - mean[None, :, None]) / std_safe[None, :, None]).astype(np.float32)

    if nm == "per_window":
        m = X.mean(axis=2, keepdims=True)
        s = X.std(axis=2, keepdims=True) + 1e-6
        return ((X - m) / s).astype(np.float32)

    if nm in ("none", "off", "false"):
        return X.astype(np.float32)

    raise ValueError(f"Unknown normalize_mode: {norm_mode}")


def _infer_logits(
    model: torch.nn.Module, X: np.ndarray, device: str, batch_size: int = 512
) -> np.ndarray:
    if X.size == 0:
        return np.zeros((0,), dtype=np.float32)
    model.eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device=device, dtype=torch.float32)
            logits = model(xb).squeeze(-1)
            outs.append(logits.detach().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def _best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    idx = int(np.nanargmax(f1))
    if idx >= len(thr):
        return 0.5
    return float(thr[idx])


def _threshold_precision_at_least(
    y_true: np.ndarray, y_prob: np.ndarray, p_min: float = 0.90
) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.99
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    candidates: list[tuple[float, float]] = []
    for i in range(len(thr)):
        if float(prec[i]) >= p_min:
            candidates.append((float(rec[i]), float(thr[i])))
    if not candidates:
        return 0.99
    candidates.sort(reverse=True)
    return float(candidates[0][1])


def _ece_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    if y_true.size == 0:
        return float("nan"), []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        m = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(m):
            continue
        acc = float(np.mean(y_true[m] == (y_prob[m] >= 0.5)))
        conf = float(np.mean(y_prob[m]))
        frac = float(np.mean(m))
        ece += frac * abs(acc - conf)
        rows.append({"bin_lo": lo, "bin_hi": hi, "frac": frac, "acc": acc, "conf": conf})
    return float(ece), rows


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict[str, float]:
    if y_true.size == 0:
        return {
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "brier": float("nan"),
            "ece_15": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "threshold": float(thr),
            "tp": 0.0,
            "fp": 0.0,
            "tn": 0.0,
            "fn": 0.0,
        }

    y_hat = (y_prob >= float(thr)).astype(np.int64)
    tp = int(np.sum((y_hat == 1) & (y_true == 1)))
    fp = int(np.sum((y_hat == 1) & (y_true == 0)))
    tn = int(np.sum((y_hat == 0) & (y_true == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true == 1)))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    pr_auc = (
        float(average_precision_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(_ece_bins(y_true, y_prob, n_bins=15)[0])

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "ece_15": ece,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(thr),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def _oracle_best_f1(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    if y_true.size == 0:
        return {"oracle_best_f1": float("nan"), "oracle_thr": float("nan")}
    qs = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, 201))).astype(np.float32)
    thrs = np.unique(np.concatenate([[0.0], qs, [1.0]])).astype(np.float32)
    best_f1 = -1.0
    best_thr = 0.5
    for t in thrs:
        f1 = _metrics(y_true, y_prob, float(t))["f1"]
        if np.isfinite(f1) and f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(t)
    return {"oracle_best_f1": best_f1, "oracle_thr": best_thr}


def _fit_logit_calibrator(val_logits: np.ndarray, val_y: np.ndarray) -> tuple[float, float]:
    # Logistic regression on a single feature (logit): p_cal = sigmoid(a*logit + b)
    if val_logits.size == 0 or len(np.unique(val_y)) < 2:
        return 1.0, 0.0
    X = val_logits.reshape(-1, 1).astype(np.float64)
    y = val_y.astype(int)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(X, y)
    a = float(lr.coef_[0, 0])
    b = float(lr.intercept_[0])
    return a, b


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["run"]["seed"]))
    out_dir = ensure_dir(cfg["run"]["out_dir"])

    device = str(cfg["run"]["device"]).lower()
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]CUDA requested but not available. Falling back to CPU.[/yellow]")
        device = "cpu"

    ckpt_path = out_dir / "model_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}. Run training first.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    scaler = ckpt.get("scaler", {})
    norm_mode = str(
        scaler.get("normalize_mode", cfg["preprocess"].get("normalize_mode", "global"))
    ).lower()
    mean = scaler.get("mean", None)
    std = scaler.get("std", None)
    if mean is None or std is None:
        raise ValueError("Checkpoint scaler missing mean/std.")
    mean = mean.numpy() if hasattr(mean, "numpy") else np.asarray(mean, dtype=np.float32)
    std = std.numpy() if hasattr(std, "numpy") else np.asarray(std, dtype=np.float32)

    thr_f1_raw = float(ckpt["val_best"]["thr_f1"])
    thr_p90_raw = float(ckpt["val_best"]["thr_p90"])
    val_subjects = _as_list(
        ckpt["val_best"].get("val_subjects", cfg["data"]["split"].get("val_subject"))
    )

    model = PhysioCNN1D(
        in_channels=int(cfg["model"]["in_channels"]),
        hidden_channels=list(cfg["model"]["hidden_channels"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        dropout=float(cfg["model"]["dropout"]),
        num_classes=1,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # Load TEST
    test_s = str(cfg["data"]["split"]["test_subject"])
    X, y, sqi = _load_or_build_subject(cfg, test_s)

    sqi_thr = float(cfg["quality"]["sqi_threshold"])
    acc = sqi >= sqi_thr
    coverage = float(np.mean(acc)) if acc.size else 0.0
    abstain = 1.0 - coverage

    total_n = int(y.size)
    pos_n = int(np.sum(y == 1))
    neg_n = int(np.sum(y == 0))
    base_pos_rate = float(pos_n / total_n) if total_n > 0 else float("nan")

    Xn = _normalize_batch(X, norm_mode, mean, std)
    test_logits_all = _infer_logits(model, Xn, device=device)
    test_p_all = _sigmoid(test_logits_all)

    y_acc = y[acc]
    p_raw = test_p_all[acc]
    logits_raw = test_logits_all[acc]

    # Save debug artifact
    np.savez_compressed(
        out_dir / "accepted_scores.npz",
        y=y_acc.astype(np.int64),
        logits_raw=logits_raw.astype(np.float32),
        p_raw=p_raw.astype(np.float32),
    )

    # RAW metrics using raw val thresholds
    m_raw_f1 = _metrics(y_acc, p_raw, thr_f1_raw)
    m_raw_p90 = _metrics(y_acc, p_raw, thr_p90_raw)
    ece_raw, bins_raw = _ece_bins(y_acc, p_raw, n_bins=15)

    # Fit calibrator on VAL (accepted windows only)
    val_logits_list: list[np.ndarray] = []
    val_y_list: list[np.ndarray] = []

    for vs in val_subjects:
        Xv, yv, sqiv = _load_or_build_subject(cfg, str(vs))
        accv = sqiv >= sqi_thr
        if not np.any(accv):
            continue
        Xvn = _normalize_batch(Xv, norm_mode, mean, std)
        lv_all = _infer_logits(model, Xvn, device=device)
        val_logits_list.append(lv_all[accv])
        val_y_list.append(yv[accv])

    if val_logits_list and val_y_list:
        val_logits = np.concatenate(val_logits_list).astype(np.float32)
        val_y = np.concatenate(val_y_list).astype(np.int64)
    else:
        val_logits = np.zeros((0,), dtype=np.float32)
        val_y = np.zeros((0,), dtype=np.int64)

    a, b = _fit_logit_calibrator(val_logits, val_y)

    # Calibrated probabilities on test accepted
    p_cal = _sigmoid(a * logits_raw + b)

    # Thresholds computed on calibrated VAL (not on test)
    p_val_cal = (
        _sigmoid(a * val_logits + b) if val_logits.size else np.zeros((0,), dtype=np.float32)
    )
    thr_f1_cal = _best_threshold_f1(val_y, p_val_cal) if val_logits.size else 0.5
    thr_p90_cal = (
        _threshold_precision_at_least(val_y, p_val_cal, p_min=0.90) if val_logits.size else 0.99
    )

    val_rate_cal_f1 = float(np.mean(p_val_cal >= thr_f1_cal)) if p_val_cal.size else float("nan")

    # Cal metrics (apply calibrated-val thresholds)
    m_cal_f1 = _metrics(y_acc, p_cal, thr_f1_cal)
    m_cal_p90 = _metrics(y_acc, p_cal, thr_p90_cal)
    ece_cal, bins_cal = _ece_bins(y_acc, p_cal, n_bins=15)

    # Adapt threshold (rate-match on test, using target rate from calibrated VAL)
    thr_f1_cal_adapt = float("nan")
    m_cal_f1_adapt: dict[str, float] = {}
    if np.isfinite(val_rate_cal_f1) and p_cal.size:
        q = float(np.clip(1.0 - val_rate_cal_f1, 0.0, 1.0))
        thr_f1_cal_adapt = float(np.quantile(p_cal, q))
        m_cal_f1_adapt = _metrics(y_acc, p_cal, thr_f1_cal_adapt)

    oracle_raw = _oracle_best_f1(y_acc, p_raw)
    oracle_cal = _oracle_best_f1(y_acc, p_cal)

    # Save reliability bins (raw + cal)
    with (out_dir / "reliability_bins_raw.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["bin_lo", "bin_hi", "frac", "acc", "conf"])
        w.writeheader()
        for r in bins_raw:
            w.writerow(r)

    with (out_dir / "reliability_bins.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["bin_lo", "bin_hi", "frac", "acc", "conf"])
        w.writeheader()
        for r in bins_cal:
            w.writerow(r)

    report = {
        "split": {"test_subject": test_s, "val_subjects": val_subjects},
        "coverage_accepted": coverage,
        "abstain_rate": abstain,
        "counts": {
            "total_n": total_n,
            "pos_n": pos_n,
            "neg_n": neg_n,
            "base_pos_rate": base_pos_rate,
            "accepted_n": int(np.sum(acc)),
            "rejected_n": int(np.sum(~acc)),
        },
        "thresholds": {
            "raw": {"thr_f1": thr_f1_raw, "thr_p90": thr_p90_raw},
            "cal": {"thr_f1": thr_f1_cal, "thr_p90": thr_p90_cal, "val_rate_f1": val_rate_cal_f1},
            "cal_adapt": {"thr_f1_adapt": thr_f1_cal_adapt},
        },
        "accepted_metrics": {
            "raw": {
                "f1_opt": m_raw_f1,
                "p>=0.90": m_raw_p90,
                "ece_15": float(ece_raw),
                "pr_auc": m_raw_f1["pr_auc"],
            },
            "cal": {
                "f1_opt": m_cal_f1,
                "p>=0.90": m_cal_p90,
                "ece_15": float(ece_cal),
                "pr_auc": m_cal_f1["pr_auc"],
            },
            "cal_adapt": {"f1_opt_adapt": m_cal_f1_adapt},
        },
        "diagnostics": {
            "norm_mode": norm_mode,
            "a": float(a),
            "b": float(b),
            "pred_pos_rate_raw@thr_f1_raw": float(np.mean(p_raw >= thr_f1_raw))
            if p_raw.size
            else float("nan"),
            "pred_pos_rate_cal@thr_f1_cal": float(np.mean(p_cal >= thr_f1_cal))
            if p_cal.size
            else float("nan"),
            "pred_pos_rate_cal@thr_f1_adapt": float(np.mean(p_cal >= thr_f1_cal_adapt))
            if (p_cal.size and np.isfinite(thr_f1_cal_adapt))
            else float("nan"),
            "oracle_raw": oracle_raw,
            "oracle_cal": oracle_cal,
        },
        "artifacts": {
            "accepted_scores_npz": str(out_dir / "accepted_scores.npz"),
            "reliability_bins_raw_csv": str(out_dir / "reliability_bins_raw.csv"),
            "reliability_bins_cal_csv": str(out_dir / "reliability_bins.csv"),
        },
        "notes": [
            "PR-AUC is threshold-free and is the primary cross-subject metric.",
            "Calibration is fit on validation subjects only (accepted windows).",
            "cal_adapt uses unlabeled test-score quantile matching to preserve the calibrated validation alert rate (deployment-style).",
            "Oracle best-F1 is diagnostic only (NOT a reportable metric).",
        ],
    }
    (out_dir / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Console tables
    tbl = Table(title=f"TEST (accepted only) — subject={test_s}")
    tbl.add_column("coverage")
    tbl.add_column("PR-AUC(raw)")
    tbl.add_column("F1(raw@ckpt_thr)")
    tbl.add_column("PR-AUC(cal)")
    tbl.add_column("F1(cal@val_thr)")
    tbl.add_column("F1(cal@adapt)")
    tbl.add_row(
        f"{coverage:.3f}",
        f"{report['accepted_metrics']['raw']['pr_auc']:.4f}",
        f"{m_raw_f1['f1']:.4f}",
        f"{report['accepted_metrics']['cal']['pr_auc']:.4f}",
        f"{m_cal_f1['f1']:.4f}",
        f"{m_cal_f1_adapt.get('f1', float('nan')):.4f}" if m_cal_f1_adapt else "nan",
    )
    console.print(tbl)

    tbl2 = Table(title="Diagnostics")
    tbl2.add_column("norm_mode")
    tbl2.add_column("a")
    tbl2.add_column("b")
    tbl2.add_column("thr_f1_raw")
    tbl2.add_column("thr_f1_cal")
    tbl2.add_column("thr_f1_adapt")
    tbl2.add_column("pred_raw")
    tbl2.add_column("pred_cal@val")
    tbl2.add_column("pred_cal@adapt")
    tbl2.add_row(
        str(norm_mode),
        f"{a:.4f}",
        f"{b:.4f}",
        f"{thr_f1_raw:.6f}",
        f"{thr_f1_cal:.6f}",
        f"{thr_f1_cal_adapt:.6f}" if np.isfinite(thr_f1_cal_adapt) else "nan",
        f"{report['diagnostics']['pred_pos_rate_raw@thr_f1_raw']:.4f}",
        f"{report['diagnostics']['pred_pos_rate_cal@thr_f1_cal']:.4f}",
        f"{report['diagnostics']['pred_pos_rate_cal@thr_f1_adapt']:.4f}",
    )
    console.print(tbl2)

    console.print(f"Wrote: {out_dir / 'test_report.json'}")
    console.print(f"Wrote: {out_dir / 'accepted_scores.npz'}")
    console.print(f"Wrote: {out_dir / 'reliability_bins.csv'}")
    console.print(f"Wrote: {out_dir / 'reliability_bins_raw.csv'}")


if __name__ == "__main__":
    main()
