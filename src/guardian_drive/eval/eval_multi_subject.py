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
    ap.add_argument("--subjects", nargs="*", default=None, help="Override subjects to evaluate")
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

    sqi_thr = float(cfg["quality"]["sqi_threshold"])

    # Evaluate across subjects (exclude val subject by default)
    all_subjects = list(cfg["data"]["subjects"])
    val_s = cfg["data"]["split"]["val_subject"]
    test_s_default = cfg["data"]["split"]["test_subject"]

    subjects = args.subjects if args.subjects else [s for s in all_subjects if s != val_s]
    if test_s_default not in subjects:
        subjects.append(test_s_default)

    rows = []
    for s in subjects:
        X, y, sqi, _ = _load_or_build_subject(cfg, s)
        acc = sqi >= sqi_thr
        cov = float(np.mean(acc))
        if np.sum(acc) == 0:
            rows.append({"subject": s, "coverage": cov})
            continue
        Xn = _normalize(X, mean, std)
        p = _infer(model, Xn, device=device)
        m = _metrics(y[acc], p[acc], thr)

        rows.append(
            {
                "subject": s,
                "coverage": cov,
                "pos_rate": float(np.mean(y == 1)),
                "pr_auc": m["pr_auc"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            }
        )

    # Aggregate (ignore missing)
    numeric = ["coverage", "pr_auc", "precision", "recall", "f1", "pos_rate"]
    agg = {}
    for k in numeric:
        vals = [r[k] for r in rows if k in r and r[k] is not None]
        if not vals:
            continue
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    out = {"sqi_threshold": sqi_thr, "rows": rows, "aggregate": agg}
    (out_dir / "multi_subject_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
