from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset

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


@dataclass
class Bundle:
    X: np.ndarray  # (N,C,T)
    y: np.ndarray  # (N,)
    sqi: np.ndarray  # (N,)
    subj: np.ndarray  # (N,)


def _load_or_build_subject(cfg: dict, subject: str) -> tuple[Bundle, list[str]]:
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
    subj = np.array([subject] * X.shape[0], dtype=object)
    return Bundle(X=X, y=y_mapped, sqi=sqi, subj=subj), names


def _concat(bundles: list[Bundle]) -> Bundle:
    if not bundles:
        return Bundle(
            X=np.zeros((0, 0, 0), dtype=np.float32),
            y=np.zeros((0,), dtype=np.int64),
            sqi=np.zeros((0,), dtype=np.float32),
            subj=np.zeros((0,), dtype=object),
        )
    X = np.concatenate([b.X for b in bundles], axis=0)
    y = np.concatenate([b.y for b in bundles], axis=0)
    sqi = np.concatenate([b.sqi for b in bundles], axis=0)
    subj = np.concatenate([b.subj for b in bundles], axis=0)
    return Bundle(X=X, y=y, sqi=sqi, subj=subj)


def _fit_global_scaler(X: np.ndarray) -> dict[str, np.ndarray]:
    if X.size == 0:
        raise RuntimeError("Cannot fit scaler on empty X.")
    mean = X.mean(axis=(0, 2), dtype=np.float64)  # (C,)
    std = X.std(axis=(0, 2), dtype=np.float64) + 1e-6
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def _apply_global_scaler(x_ct: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x_ct - mean[:, None]) / std[:, None]).astype(np.float32)


def _apply_per_window_norm(x_ct: np.ndarray) -> np.ndarray:
    m = x_ct.mean(axis=1, keepdims=True)
    s = x_ct.std(axis=1, keepdims=True) + 1e-6
    return ((x_ct - m) / s).astype(np.float32)


class PhysioDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        normalize_mode: str,
        mean: np.ndarray | None,
        std: np.ndarray | None,
        augment: bool,
        noise_std: float,
        channel_dropout_prob: float,
    ) -> None:
        self.X = X
        self.y = y
        self.normalize_mode = str(normalize_mode).lower()
        self.mean = mean
        self.std = std
        self.augment = bool(augment)
        self.noise_std = float(noise_std)
        self.channel_dropout_prob = float(channel_dropout_prob)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]  # (C,T)

        if self.normalize_mode == "per_window":
            x = _apply_per_window_norm(x)
        elif self.normalize_mode == "global":
            if self.mean is None or self.std is None:
                raise RuntimeError("global normalization requested but mean/std are missing.")
            x = _apply_global_scaler(x, self.mean, self.std)
        elif self.normalize_mode in ("none", "off", "false"):
            x = x.astype(np.float32)
        else:
            raise ValueError(f"Unknown preprocess.normalize_mode: {self.normalize_mode}")

        if self.augment:
            if self.noise_std > 0:
                x = x + np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
            if self.channel_dropout_prob > 0:
                drop = np.random.rand(x.shape[0]) < self.channel_dropout_prob
                x[drop, :] = 0.0

        y = np.array(self.y[idx], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def _infer_probs(
    model: torch.nn.Module, dl: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device=device, dtype=torch.float32)
            logits = model(xb).squeeze(-1)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            ps.append(prob)
            ys.append(yb.detach().cpu().numpy())
    y = np.concatenate(ys).astype(np.int64)
    p = np.concatenate(ps).astype(np.float32)
    return y, p


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

    # Subjects
    subjects = [str(s) for s in cfg["data"]["subjects"]]
    subjects = [s for s in subjects if s != "S12"]  # common missing

    test_s = str(cfg["data"]["split"]["test_subject"])
    val_subjects = _as_list(cfg["data"]["split"].get("val_subject"))
    val_subjects = [s for s in val_subjects if s and s != "S12"]
    val_subjects = list(dict.fromkeys(val_subjects))

    if test_s not in subjects:
        raise ValueError(f"test_subject={test_s} not in data.subjects")
    if not val_subjects:
        raise ValueError("data.split.val_subject is empty (needs 1+ subjects).")
    for vs in val_subjects:
        if vs not in subjects:
            raise ValueError(f"val_subject {vs} not in data.subjects")
        if vs == test_s:
            raise ValueError(f"val_subject {vs} cannot equal test_subject {test_s}")

    exclude = set(val_subjects + [test_s])
    train_subjects = [s for s in subjects if s not in exclude]
    console.print(f"Split: train={train_subjects} val={val_subjects} test={[test_s]}")

    # Load
    bundles_train: list[Bundle] = []
    bundles_val: list[Bundle] = []
    names_ref: list[str] | None = None

    for s in train_subjects:
        b, names = _load_or_build_subject(cfg, s)
        names_ref = names_ref or names
        if names != names_ref:
            raise ValueError(f"Channel mismatch: {s} has {names}, expected {names_ref}")
        bundles_train.append(b)

    for s in val_subjects:
        b, names = _load_or_build_subject(cfg, s)
        names_ref = names_ref or names
        if names != names_ref:
            raise ValueError(f"Channel mismatch: {s} has {names}, expected {names_ref}")
        bundles_val.append(b)

    train = _concat(bundles_train)
    val = _concat(bundles_val)

    # SQI gating
    sqi_thr = float(cfg["quality"]["sqi_threshold"])
    train_acc = train.sqi >= sqi_thr
    val_acc = val.sqi >= sqi_thr

    if not np.any(train_acc):
        raise RuntimeError("No training windows pass SQI threshold.")
    if not np.any(val_acc):
        raise RuntimeError("No validation windows pass SQI threshold.")

    train_X = train.X[train_acc]
    train_y = train.y[train_acc]
    val_X = val.X[val_acc]
    val_y = val.y[val_acc]

    # Normalize
    normalize_mode = str(cfg["preprocess"].get("normalize_mode", "global")).lower()
    if normalize_mode not in ("global", "per_window", "none", "off", "false"):
        raise ValueError(f"Unsupported preprocess.normalize_mode={normalize_mode}")

    if normalize_mode == "global":
        scaler = _fit_global_scaler(train_X)
        mean, std = scaler["mean"], scaler["std"]
    else:
        mean = np.zeros((train_X.shape[1],), dtype=np.float32)
        std = np.ones((train_X.shape[1],), dtype=np.float32)
        scaler = {"mean": mean, "std": std}

    # Save scaler artifact (debuggable)
    (out_dir / "scaler.json").write_text(
        json.dumps(
            {
                "normalize_mode": normalize_mode,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "channel_names": names_ref,
                "sqi_threshold": sqi_thr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Class imbalance
    pos = float(np.sum(train_y == 1))
    neg = float(np.sum(train_y == 0))
    pos_weight_val = float(neg / (pos + 1e-12))
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)

    # Model
    model = PhysioCNN1D(
        in_channels=int(cfg["model"]["in_channels"]),
        hidden_channels=list(cfg["model"]["hidden_channels"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        dropout=float(cfg["model"]["dropout"]),
        num_classes=1,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    rob = cfg.get("robustness", {})
    augment = bool(rob.get("enable", True))
    noise_std = float(rob.get("noise_std", 0.0))
    ch_drop = float(rob.get("channel_dropout_prob", 0.0))

    ds_tr = PhysioDataset(
        X=train_X,
        y=train_y,
        normalize_mode=normalize_mode,
        mean=mean,
        std=std,
        augment=augment,
        noise_std=noise_std,
        channel_dropout_prob=ch_drop,
    )
    dl_tr = DataLoader(
        ds_tr, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=0
    )

    ds_val = PhysioDataset(
        X=val_X,
        y=val_y,
        normalize_mode=normalize_mode,
        mean=mean,
        std=std,
        augment=False,
        noise_std=0.0,
        channel_dropout_prob=0.0,
    )
    dl_val = DataLoader(ds_val, batch_size=512, shuffle=False, num_workers=0)

    # Train loop
    best_pr = -1.0
    best_path = out_dir / "model_best.pt"
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))
    epochs = int(cfg["train"]["epochs"])

    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        for xb, yb in dl_tr:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(-1)
            loss = loss_fn(logits, yb)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            running += float(loss.item()) * int(xb.size(0))
            n_seen += int(xb.size(0))

        train_loss = running / max(n_seen, 1)

        # Val predictions
        yv, pv = _infer_probs(model, dl_val, device=device)

        thr_f1 = _best_threshold_f1(yv, pv)
        thr_p90 = _threshold_precision_at_least(yv, pv, p_min=0.90)

        val_pr_auc = (
            float(average_precision_score(yv, pv)) if len(np.unique(yv)) > 1 else float("nan")
        )
        val_roc_auc = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else float("nan")

        # Predicted positive rates on val (used later by eval for rate-matching if desired)
        val_rate_f1 = float(np.mean(pv >= thr_f1)) if pv.size else float("nan")
        val_rate_p90 = float(np.mean(pv >= thr_p90)) if pv.size else float("nan")

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_pr_auc": val_pr_auc,
            "val_roc_auc": val_roc_auc,
            "thr_f1": float(thr_f1),
            "thr_p90": float(thr_p90),
            "val_rate_f1": val_rate_f1,
            "val_rate_p90": val_rate_p90,
            "coverage_train": float(np.mean(train_acc)),
            "coverage_val": float(np.mean(val_acc)),
            "normalize_mode": normalize_mode,
            "pos_weight": pos_weight_val,
            "val_subjects": val_subjects,
            "test_subject": test_s,
        }
        history.append(record)
        (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        tbl = Table(title=f"Epoch {epoch}")
        tbl.add_column("train_loss")
        tbl.add_column("val_pr_auc")
        tbl.add_column("thr_f1")
        tbl.add_row(f"{train_loss:.4f}", f"{val_pr_auc:.4f}", f"{thr_f1:.6f}")
        console.print(tbl)

        if np.isfinite(val_pr_auc) and val_pr_auc > best_pr:
            best_pr = float(val_pr_auc)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "scaler": {
                        "normalize_mode": normalize_mode,
                        "mean": mean,
                        "std": std,
                        "channel_names": names_ref,
                    },
                    "val_best": {
                        "pr_auc": best_pr,
                        "roc_auc": float(val_roc_auc),
                        "thr_f1": float(thr_f1),
                        "thr_p90": float(thr_p90),
                        "val_rate_f1": float(val_rate_f1),
                        "val_rate_p90": float(val_rate_p90),
                        "pos_weight": pos_weight_val,
                        "val_subjects": val_subjects,
                        "sqi_threshold": sqi_thr,
                    },
                },
                best_path,
            )
            console.print(
                f"[green]Saved best checkpoint -> {best_path} (val PR-AUC={best_pr:.4f})[/green]"
            )

    if best_path.exists():
        chk = torch.load(best_path, map_location="cpu", weights_only=False)
        console.print(f"Best checkpoint val PR-AUC={chk['val_best']['pr_auc']:.4f}")
        console.print(
            "Now run: python -m guardian_drive.eval.eval_physio --config <your_fold_config>"
        )
    else:
        console.print("[red]No checkpoint written. Check training logs.[/red]")


if __name__ == "__main__":
    main()
