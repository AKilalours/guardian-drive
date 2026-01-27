from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from guardian_drive.eval.eval_physio import _infer, _load_or_build_subject, _metrics, _normalize
from guardian_drive.models.physio_cnn import PhysioCNN1D
from guardian_drive.utils.config import ensure_dir, load_yaml
from guardian_drive.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["run"]["seed"]))
    out_dir = ensure_dir(cfg["run"]["out_dir"])

    device = cfg["run"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ckpt_path = out_dir / "model_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    scaler = ckpt["scaler"]
    mean = (
        scaler["mean"].numpy() if hasattr(scaler["mean"], "numpy") else np.asarray(scaler["mean"])
    )
    std = scaler["std"].numpy() if hasattr(scaler["std"], "numpy") else np.asarray(scaler["std"])
    thr = float(ckpt["val_best"]["thr_f1"])

    model = PhysioCNN1D(
        in_channels=int(cfg["model"]["in_channels"]),
        hidden_channels=list(cfg["model"]["hidden_channels"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        dropout=float(cfg["model"]["dropout"]),
        num_classes=1,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    test_s = cfg["data"]["split"]["test_subject"]
    X, y, sqi, _ = _load_or_build_subject(cfg, test_s)
    Xn = _normalize(X, mean, std)
    p = _infer(model, Xn, device=device)

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []
    for t in thresholds:
        acc = sqi >= t
        cov = float(np.mean(acc))
        if np.sum(acc) == 0:
            rows.append(
                {
                    "sqi_thr": t,
                    "coverage": cov,
                    "pr_auc": None,
                    "f1": None,
                    "precision": None,
                    "recall": None,
                }
            )
            continue
        m = _metrics(y[acc], p[acc], thr)
        rows.append(
            {
                "sqi_thr": t,
                "coverage": cov,
                "pr_auc": m["pr_auc"],
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
            }
        )

    out = {
        "test_subject": test_s,
        "rows": rows,
        "note": "Threshold uses val thr_f1; sweep varies SQI only.",
    }
    (out_dir / "sqi_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
