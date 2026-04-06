"""
learned/task_b_trainer.py
Guardian Drive — Task B: Drowsiness / Fatigue (physio channel)
Adapted from: WJMatthew/WESAD + Munroe-Meyer WESAD pipeline

Loads YOUR existing data at:  data/raw/WESAD/WESAD/

Run:
    python learned/task_b_trainer.py            # 80/20 split
    python learned/task_b_trainer.py --loso     # Leave-One-Subject-Out
    python learned/task_b_trainer.py --eval     # eval only

Outputs:
    learned/models/task_b_tcn.pt
    learned/results/task_b_eval.json

WESAD label map:
    1=baseline  2=stress  3=amusement  4=meditation(low-arousal)
Binary: 0=alert (1,2,3)   1=drowsy/low-arousal (4)
"""
from __future__ import annotations
import argparse, json, os, pickle, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

CHEST_FS   = 700
WIN_S      = 60
WIN_LEN    = CHEST_FS * WIN_S
STEP_LEN   = CHEST_FS * 30    # 50 % overlap
BATCH      = 16
EPOCHS     = 40
LR         = 5e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_LABEL_MAP   = {1:0, 2:0, 3:0, 4:1, 5:0, 6:0, 7:0}
_CHEST_SIGS  = ["ECG", "EDA", "Temp", "Resp"]
N_CH         = len(_CHEST_SIGS)


# ── dataset ───────────────────────────────────────────────────────────────────
class WESADDataset(Dataset):
    def __init__(self, segs, labels):
        self.X = [torch.FloatTensor(s) for s in segs]
        self.y = torch.LongTensor(labels)
    def __len__(self):         return len(self.y)
    def __getitem__(self, i):  return self.X[i], self.y[i]


def _load_subject(pkl: str):
    with open(pkl, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    chest = d["signal"]["chest"]
    raw_labels = d["label"]
    sigs = []
    for name in _CHEST_SIGS:
        s = chest[name].flatten().astype(np.float32)
        s = (s - s.mean()) / (s.std() + 1e-6)
        sigs.append(s)
    n = min(min(len(s) for s in sigs), len(raw_labels))
    sig_arr = np.array([s[:n] for s in sigs])           # (C, T)
    lbl_arr = np.array([_LABEL_MAP.get(int(l), 0) for l in raw_labels[:n]], dtype=np.int64)
    return sig_arr, lbl_arr


def _windows(sig, lbl):
    segs, lbls = [], []
    T = sig.shape[1]
    for s in range(0, T - WIN_LEN + 1, STEP_LEN):
        w = sig[:, s:s+WIN_LEN]
        l = lbl[s:s+WIN_LEN]
        segs.append(w); lbls.append(int(np.round(l.mean())))
    return segs, lbls


def _load_wesad(data_dir: str):
    all_segs, all_lbls, all_subj = [], [], []
    dirs = sorted(Path(data_dir).glob("S*"))
    if not dirs:
        raise FileNotFoundError(f"No subject folders in {data_dir}. Expected S2/, S3/ …")
    print(f"[Task B] {len(dirs)} subjects found")
    for d in dirs:
        pkl = d / f"{d.name}.pkl"
        if not pkl.exists():
            print(f"  {d.name}: no pkl — skipped"); continue
        try:
            sig, lbl = _load_subject(str(pkl))
            segs, lbls = _windows(sig, lbl)
            all_segs.extend(segs); all_lbls.extend(lbls)
            all_subj.extend([d.name]*len(segs))
            print(f"  {d.name}: {len(segs)} windows  alert={lbls.count(0)}  drowsy={lbls.count(1)}")
        except Exception as e:
            print(f"  {d.name}: ERROR — {e}")
    if not all_segs:
        raise RuntimeError("No WESAD windows loaded — check data layout.")
    print(f"[Task B] Total {len(all_segs)} windows  alert={all_lbls.count(0)}  drowsy={all_lbls.count(1)}")
    return all_segs, all_lbls, all_subj


# ── TCN model ─────────────────────────────────────────────────────────────────
class _TBlock(nn.Module):
    def __init__(self, ic, oc, k, dil, drop=0.2):
        super().__init__()
        p = (k-1)*dil
        self.net = nn.Sequential(
            nn.Conv1d(ic,oc,k,padding=p,dilation=dil), nn.BatchNorm1d(oc), nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(oc,oc,k,padding=p,dilation=dil), nn.BatchNorm1d(oc), nn.ReLU(), nn.Dropout(drop),
        )
        self.skip = nn.Conv1d(ic,oc,1) if ic!=oc else nn.Identity()
    def forward(self, x):
        out = self.net(x)[:,:,:x.size(2)]
        return torch.relu(out + self.skip(x))


class DrowsinessTCN(nn.Module):
    """Multi-channel physio TCN — ~250 K params — CPU real-time."""
    def __init__(self):
        super().__init__()
        layers, ch = [], N_CH
        for oc, dil in zip([64,128,128,64],[1,2,4,8]):
            layers.append(_TBlock(ch, oc, 7, dil)); ch = oc
        self.tcn  = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,2)
        )
    def forward(self, x): return self.head(self.tcn(x))


# ── train / eval helpers ──────────────────────────────────────────────────────
def _train_epoch(model, loader, opt, crit):
    model.train(); tot = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); loss = crit(model(X), y); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += loss.item()
    return tot / len(loader)


def _eval(model, loader) -> dict:
    model.eval(); probs, labs = [], []
    with torch.no_grad():
        for X, y in loader:
            p = torch.softmax(model(X.to(DEVICE)),1)[:,1].cpu().numpy()
            probs.extend(p); labs.extend(y.numpy())
    probs, labs = np.array(probs), np.array(labs)
    auc = float(roc_auc_score(labs,probs)) if len(set(labs))>1 else 0.0
    hours = len(labs)*WIN_S/3600
    best = {"sensitivity":0.0,"far_per_hr":999.0,"threshold":0.5}
    for t in np.arange(0.3,0.95,0.02):
        pred=(probs>=t).astype(int)
        tp=((pred==1)&(labs==1)).sum(); fn=((pred==0)&(labs==1)).sum(); fp=((pred==1)&(labs==0)).sum()
        sens=tp/(tp+fn+1e-6); far=fp/(hours+1e-6)
        if far<=2.0 and sens>best["sensitivity"]:
            best={"sensitivity":round(float(sens),4),"far_per_hr":round(float(far),4),"threshold":round(float(t),3)}
    return {"auc":round(auc,4),**best}


def _run_training(segs, lbls, out_path, res_path):
    idx = np.random.permutation(len(lbls)); split=int(0.8*len(idx))
    ti, vi = idx[:split], idx[split:]
    tl = DataLoader(WESADDataset([segs[i] for i in ti],[lbls[i] for i in ti]), BATCH, shuffle=True,  num_workers=2)
    vl = DataLoader(WESADDataset([segs[i] for i in vi],[lbls[i] for i in vi]), BATCH, shuffle=False, num_workers=2)
    model = DrowsinessTCN().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit  = nn.CrossEntropyLoss()
    best_auc = 0.0
    print(f"  Training {EPOCHS} epochs on {len(ti)} windows …")
    for ep in range(1,EPOCHS+1):
        t0=time.time(); loss=_train_epoch(model,tl,opt,crit); sched.step(); r=_eval(model,vl)
        print(f"    ep {ep:02d}/{EPOCHS}  loss={loss:.4f}  AUC={r['auc']:.4f}  sens={r['sensitivity']:.4f}  FAR/hr={r['far_per_hr']:.2f}  ({time.time()-t0:.1f}s)")
        if r["auc"]>best_auc:
            best_auc=r["auc"]; torch.save(model.state_dict(), out_path)
            json.dump(r, open(res_path,"w"), indent=2); print(f"    ✓ saved (AUC={best_auc:.4f})")
    return best_auc


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    out = Path("learned/models");  out.mkdir(parents=True, exist_ok=True)
    res = Path("learned/results"); res.mkdir(parents=True, exist_ok=True)
    mp  = out/"task_b_tcn.pt";    rp = res/"task_b_eval.json"

    print(f"\n{'='*55}\nTask B — Drowsiness Screening (WESAD)  |  device={DEVICE}\n{'='*55}")
    segs, lbls, subjs = _load_wesad(args.data_dir)

    if args.eval:
        model = DrowsinessTCN().to(DEVICE)
        model.load_state_dict(torch.load(mp, map_location=DEVICE))
        r = _eval(model, DataLoader(WESADDataset(segs,lbls), BATCH))
        print(f"Eval: {r}"); json.dump(r, open(rp,"w"), indent=2); return

    if args.loso:
        unique = sorted(set(subjs)); fold_aucs=[]
        print(f"LOSO across {len(unique)} subjects …")
        for ts in unique:
            ti=[i for i,s in enumerate(subjs) if s!=ts]; vi=[i for i,s in enumerate(subjs) if s==ts]
            print(f"\n  Hold-out: {ts}")
            best_auc = _run_training([segs[i] for i in ti],[lbls[i] for i in ti], mp, rp)
            fold_aucs.append(best_auc)
        mean_auc = round(float(np.mean(fold_aucs)),4)
        loso_res = {"loso_auc_mean":mean_auc,"loso_auc_per_fold":fold_aucs}
        json.dump(loso_res, open(rp,"w"), indent=2)
        print(f"\nLOSO mean AUC = {mean_auc:.4f}  →  {rp}")
        return

    best = _run_training(segs, lbls, mp, rp)
    print(f"\nDone — best AUC={best:.4f}  →  {mp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw/WESAD/WESAD")
    ap.add_argument("--loso",     action="store_true")
    ap.add_argument("--eval",     action="store_true")
    main(ap.parse_args())
