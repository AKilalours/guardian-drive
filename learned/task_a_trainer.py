"""
learned/task_a_trainer.py
Guardian Drive — Task A: Arrhythmia Risk Screening
Adapted from: physionetchallenges/python-classifier-2021 (BSD)

Loads YOUR existing data at:  data/raw/ptbdb/1.0.0/

Run:
    pip install wfdb                            # one-time
    python learned/task_a_trainer.py            # train
    python learned/task_a_trainer.py --eval     # eval only

Outputs:
    learned/models/task_a_cnn.pt
    learned/results/task_a_eval.json
"""
from __future__ import annotations
import argparse, glob, json, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

FS         = 1000
WIN_S      = 10
WIN_LEN    = FS * WIN_S
STEP_LEN   = WIN_LEN // 2
BATCH      = 32
EPOCHS     = 30
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ABNORMAL  = {"infarction", "st/t", "conduction", "hypertrophy"}
_NORMAL    = {"normal"}


# ── dataset ───────────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    def __init__(self, segs, labels):
        self.X = [torch.FloatTensor(s).unsqueeze(0) for s in segs]
        self.y = torch.LongTensor(labels)
    def __len__(self):         return len(self.y)
    def __getitem__(self, i):  return self.X[i], self.y[i]


def _load(data_dir: str):
    segs, labels = [], []
    try:
        import wfdb
        files = sorted(glob.glob(os.path.join(data_dir, "**", "*.hea"), recursive=True))
        print(f"[Task A] Found {len(files)} .hea files")
        for hea in files:
            rec = hea.replace(".hea", "")
            try:
                r = wfdb.rdrecord(rec)
                c = " ".join(r.comments or []).lower()
                if   any(k in c for k in _ABNORMAL): lab = 1
                elif any(k in c for k in _NORMAL):   lab = 0
                else: continue
                sig = r.p_signal[:, 0].astype(np.float32)
                sig = (sig - sig.mean()) / (sig.std() + 1e-6)
                for s in range(0, len(sig) - WIN_LEN + 1, STEP_LEN):
                    segs.append(sig[s:s + WIN_LEN]); labels.append(lab)
            except Exception: continue
        if segs:
            print(f"[Task A] {len(segs)} windows  normal={labels.count(0)}  abnormal={labels.count(1)}")
            return segs, labels
    except ImportError:
        print("[Task A] wfdb not found — run: pip install wfdb")

    # .npy fallback
    for npy in sorted(glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)):
        arr = np.load(npy, allow_pickle=True).flatten().astype(np.float32)
        lab = 1 if "abnormal" in npy.lower() else 0
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        for s in range(0, len(arr) - WIN_LEN + 1, STEP_LEN):
            segs.append(arr[s:s + WIN_LEN]); labels.append(lab)

    if not segs:
        raise FileNotFoundError(
            f"No PTBDB data found in {data_dir}.\n"
            "Install wfdb:  pip install wfdb\n"
            "Or place .npy files with 'normal'/'abnormal' in the filename."
        )
    print(f"[Task A] {len(segs)} windows (npy)  normal={labels.count(0)}  abnormal={labels.count(1)}")
    return segs, labels


# ── model ─────────────────────────────────────────────────────────────────────
class ArrhythmiaNet(nn.Module):
    """Lightweight 1-D CNN — ~180 K params — runs real-time on CPU."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 32,  25, stride=2, padding=12), nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 15, stride=2, padding=7),  nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128,  9, stride=2, padding=4),  nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(32),
        )
        self.clf = nn.Sequential(
            nn.Flatten(), nn.Linear(128*32, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 2)
        )
    def forward(self, x): return self.clf(self.enc(x))


# ── train / eval helpers ──────────────────────────────────────────────────────
def _train_epoch(model, loader, opt, crit):
    model.train(); loss_sum = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); loss = crit(model(X), y); loss.backward(); opt.step()
        loss_sum += loss.item()
    return loss_sum / len(loader)


def _eval(model, loader) -> dict:
    model.eval(); probs, labs = [], []
    with torch.no_grad():
        for X, y in loader:
            p = torch.softmax(model(X.to(DEVICE)), 1)[:, 1].cpu().numpy()
            probs.extend(p); labs.extend(y.numpy())
    probs, labs = np.array(probs), np.array(labs)
    auc = float(roc_auc_score(labs, probs)) if len(set(labs)) > 1 else 0.0
    best = {"sensitivity": 0.0, "far_per_hr": 999.0, "threshold": 0.5}
    hours = len(labs) * WIN_S / 3600
    for t in np.arange(0.3, 0.95, 0.01):
        pred = (probs >= t).astype(int)
        tp = ((pred==1)&(labs==1)).sum(); fn = ((pred==0)&(labs==1)).sum(); fp = ((pred==1)&(labs==0)).sum()
        sens = tp/(tp+fn+1e-6); far = fp/(hours+1e-6)
        if far <= 2.0 and sens > best["sensitivity"]:
            best = {"sensitivity": round(float(sens),4), "far_per_hr": round(float(far),4), "threshold": round(float(t),3)}
    return {"auc": round(auc,4), **best}


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    out = Path("learned/models");   out.mkdir(parents=True, exist_ok=True)
    res = Path("learned/results");  res.mkdir(parents=True, exist_ok=True)
    mp  = out / "task_a_cnn.pt"

    print(f"\n{'='*55}\nTask A — Arrhythmia Screening  |  device={DEVICE}\n{'='*55}")
    segs, labels = _load(args.data_dir)
    segs, labels = np.array(segs, np.float32), np.array(labels, np.int64)

    if args.eval:
        model = ArrhythmiaNet().to(DEVICE)
        model.load_state_dict(torch.load(mp, map_location=DEVICE))
        r = _eval(model, DataLoader(ECGDataset(list(segs), list(labels)), BATCH))
        print(f"Eval: {r}")
        json.dump(r, open(res/"task_a_eval.json","w"), indent=2)
        return

    idx = np.random.permutation(len(labels)); split = int(0.8*len(idx))
    ti, vi = idx[:split], idx[split:]
    tl = DataLoader(ECGDataset(list(segs[ti]), list(labels[ti])), BATCH, shuffle=True,  num_workers=2)
    vl = DataLoader(ECGDataset(list(segs[vi]), list(labels[vi])), BATCH, shuffle=False, num_workers=2)

    model = ArrhythmiaNet().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit  = nn.CrossEntropyLoss()
    best_auc = 0.0

    print(f"Training {EPOCHS} epochs on {len(ti)} windows …\n")
    for ep in range(1, EPOCHS+1):
        t0   = time.time()
        loss = _train_epoch(model, tl, opt, crit); sched.step()
        r    = _eval(model, vl)
        print(f"  ep {ep:02d}/{EPOCHS}  loss={loss:.4f}  AUC={r['auc']:.4f}  "
              f"sens={r['sensitivity']:.4f}  FAR/hr={r['far_per_hr']:.2f}  ({time.time()-t0:.1f}s)")
        if r["auc"] > best_auc:
            best_auc = r["auc"]
            torch.save(model.state_dict(), mp)
            json.dump(r, open(res/"task_a_eval.json","w"), indent=2)
            print(f"  ✓ saved  (AUC={best_auc:.4f})")

    print(f"\nDone — best AUC={best_auc:.4f}  →  {mp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw/ptbdb/1.0.0")
    ap.add_argument("--eval", action="store_true")
    main(ap.parse_args())
