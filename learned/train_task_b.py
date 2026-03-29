"""Train Task B learned model (TinyTCN).

This script is designed to be replaced with real data.

Run (synthetic demo):
  pip install -r requirements-ml.txt
  python -m learned.train_task_b --synthetic

Run (real):
  python -m learned.train_task_b --dataset /path/to/your_dataset

Outputs:
- artifacts/task_b_tcn.pt
- reports/task_b_train.json

"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except Exception as e:  # pragma: no cover
    raise SystemExit("PyTorch not installed. Install with: pip install -r requirements-ml.txt")

from .models_tcn import TinyTCN
from .datasets import SyntheticTaskBDataset


def train_synthetic(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = SyntheticTaskBDataset()
    X, y = ds.generate()
    n = len(y)
    idx = np.arange(n)
    np.random.shuffle(idx)
    tr, va = idx[: int(0.8*n)], idx[int(0.8*n):]

    Xtr, ytr = torch.tensor(X[tr]), torch.tensor(y[tr])
    Xva, yva = torch.tensor(X[va]), torch.tensor(y[va])

    model = TinyTCN(n_features=X.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    dl_tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    dl_va = DataLoader(TensorDataset(Xva, yva), batch_size=256)

    best = 0.0
    for epoch in range(10):
        model.train()
        for xb, yb in dl_tr:
            p = model(xb)
            loss = bce(p, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            ps = []
            ys = []
            for xb, yb in dl_va:
                ps.append(model(xb).cpu().numpy())
                ys.append(yb.cpu().numpy())
        p = np.concatenate(ps)
        y0 = np.concatenate(ys)
        acc = float(((p > 0.5) == (y0 > 0.5)).mean())
        if acc > best:
            best = acc
            torch.save(model.state_dict(), out_dir / "task_b_tcn.pt")
        print(f"epoch={epoch} val_acc={acc:.3f} best={best:.3f}")

    (out_dir / "task_b_train.json").write_text(json.dumps({"val_acc": best, "note": "synthetic demo only"}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default="artifacts")
    args = p.parse_args()

    out_dir = Path(args.out)
    if args.synthetic:
        train_synthetic(out_dir)
    else:
        raise SystemExit("Real dataset training not wired yet. Use --synthetic as a demo.")


if __name__ == "__main__":
    main()
