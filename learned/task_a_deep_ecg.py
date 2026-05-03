"""
learned/task_a_deep_ecg.py
Task A Deep ECG -- 1D CNN + Attention for arrhythmia screening

Replaces RandomForest with deep learning:
- 1D CNN feature extraction
- Self-attention for temporal dependencies
- Patient-level evaluation on PTBDB

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

class ECGAttention(nn.Module):
    """Self-attention over temporal ECG features."""
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        # x: [B, T, D]
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)

class DeepECGClassifier(nn.Module):
    """
    1D CNN + Self-Attention for ECG classification.
    Architecture mirrors Tesla-style temporal feature extraction.
    """
    def __init__(self, input_len=1000, d=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, d, 3, padding=1), nn.BatchNorm1d(d), nn.ReLU(),
            nn.MaxPool1d(2))
        self.attn  = ECGAttention(d=d, heads=4)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.head  = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x):
        # x: [B, 1, T]
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.transpose(1, 2)   # [B, T', D]
        x = self.attn(x)
        x = self.pool(x.transpose(1,2)).squeeze(-1)
        return self.head(x).squeeze(-1)

def load_ptbdb_wfdb(data_root="datasets/ptbdb/1.0.0",
                     n_samples=1000):
    import wfdb
    data_path = Path(data_root)
    patients  = sorted([d for d in data_path.iterdir() if d.is_dir()])
    HEALTHY   = {f"patient{str(i).zfill(3)}" for i in range(1,53)}
    X_list,y_list,groups=[],[],[]
    for pat_id,pdir in enumerate(patients):
        label = 1 if pdir.name in HEALTHY else 0
        for hea in pdir.glob("*.hea"):
            try:
                rec=wfdb.rdrecord(str(hea).replace(".hea",""),
                                  channels=[0,1,2,3,4,5,6,7,8,9,10,11])
                sig=rec.p_signal
                if sig is None or sig.shape[0]<n_samples: continue
                ecg=sig[:n_samples,0].astype(np.float32)
                if np.isnan(ecg).any(): continue
                ecg=(ecg-ecg.mean())/(ecg.std()+1e-6)
                X_list.append(ecg); y_list.append(label)
                groups.append(pat_id)
            except Exception: continue
    return (np.array(X_list,dtype=np.float32),
            np.array(y_list,dtype=int),
            np.array(groups,dtype=int))

if __name__ == "__main__":
    print("Task A -- Deep ECG 1D CNN + Attention")
    print("=" * 50)

    X,y,g = load_ptbdb_wfdb()
    print(f"Loaded: {len(X)} records | normal={y.sum()} abnormal={(y==0).sum()}")

    device = "mps" if (hasattr(torch.backends,"mps") and
                        torch.backends.mps.is_available()) else "cpu"
    print(f"Device: {device}")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs=[]
    all_preds,all_trues=[],[]

    for fold,(tr,te) in enumerate(cv.split(X,y,g)):
        X_tr,X_te=X[tr],X[te]
        y_tr,y_te=y[tr],y[te]
        if len(np.unique(y_te))<2: continue

        # Normalize
        mu=X_tr.mean(); std=X_tr.std()+1e-6
        X_tr_n=(X_tr-mu)/std; X_te_n=(X_te-mu)/std

        dl=DataLoader(
            TensorDataset(torch.FloatTensor(X_tr_n[:,None,:]),
                          torch.FloatTensor(y_tr)),
            batch_size=32,shuffle=True,drop_last=True)

        model=DeepECGClassifier(input_len=1000).to(device)
        pw=torch.tensor([(y_tr==0).sum()/max(y_tr.sum(),1)],
                         dtype=torch.float32).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20)
        cr=nn.BCEWithLogitsLoss(pos_weight=pw)

        for ep in range(20):
            model.train()
            for xb,yb in dl:
                xb,yb=xb.to(device),yb.to(device)
                opt.zero_grad()
                cr(model(xb),yb).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
            sch.step()

        model.eval()
        te_dl=DataLoader(
            TensorDataset(torch.FloatTensor(X_te_n[:,None,:])),
            batch_size=64)
        preds=[]
        with torch.no_grad():
            for (xb,) in te_dl:
                p=torch.sigmoid(model(xb.to(device))).cpu().numpy()
                preds.extend(p)
        auc=roc_auc_score(y_te,preds)
        fold_aucs.append(auc)
        all_preds.extend(preds); all_trues.extend(y_te)
        print(f"  Fold {fold+1}: AUC={auc:.4f}")

    all_preds=np.array(all_preds); all_trues=np.array(all_trues)
    overall=roc_auc_score(all_trues,all_preds)
    cm=confusion_matrix(all_trues,(all_preds>0.5).astype(int))
    tn,fp,fn,tp=cm.ravel()

    result={
        "task":"Task A Deep ECG",
        "model":"1D CNN + Self-Attention",
        "dataset":"PTBDB 290 patients",
        "evaluation":"Patient-level 5-fold CV",
        "fold_aucs":[round(a,4) for a in fold_aucs],
        "mean_auc":round(float(np.mean(fold_aucs)),4),
        "std_auc":round(float(np.std(fold_aucs)),4),
        "overall_auc":round(overall,4),
        "sensitivity":round(tp/(tp+fn+1e-6),4),
        "specificity":round(tn/(tn+fp+1e-6),4),
        "ppv":round(tp/(tp+fp+1e-6),4),
        "npv":round(tn/(tn+fn+1e-6),4),
        "vs_rf_mean_auc":0.6378,
        "improvement":round(float(np.mean(fold_aucs))-0.6378,4),
        "note":"Not validated for clinical use.",
        "authors":"Akilan Manivannan & Akila Lourdes Miriyala Francis"
    }
    Path("learned/results").mkdir(exist_ok=True)
    Path("learned/results/task_a_deep_ecg.json").write_text(
        json.dumps(result,indent=2))
    print(f"\nDeep ECG AUC: {result['mean_auc']} vs RF: 0.6378")
    print(f"Improvement:  {result['improvement']:+.4f}")
    print(json.dumps(result,indent=2))
