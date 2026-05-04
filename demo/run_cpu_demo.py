"""
demo/run_cpu_demo.py
Guardian Drive -- CPU demo (no GPU required)
Runs full pipeline on synthetic data to verify installation.

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""
import numpy as np
import torch
import torch.nn as nn
import json, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Guardian Drive -- CPU Demo")
print("=" * 50)

# 1. Synthetic window
np.random.seed(42)
window = np.random.randn(4, 4200).astype(np.float32)
print(f"Input window: {window.shape} (4 channels x 4200 samples)")

# 2. SQI
from tests.test_sqi import sqi_numpy
sqi = sqi_numpy(window)
print(f"SQI: {sqi['total']:.3f} -> {'PREDICT' if sqi['total']>0.3 else 'ABSTAIN'}")

# 3. HRV
from tests.test_hrv_features import hrv_numpy
# Simulate RR intervals from ECG
ecg = window[0]
threshold = ecg.mean() + 0.5*ecg.std()
peaks = np.where((ecg[1:-1]>ecg[:-2]) &
                 (ecg[1:-1]>ecg[2:]) &
                 (ecg[1:-1]>threshold))[0]
if len(peaks) > 3:
    rr = np.diff(peaks)/700*1000
    hrv = hrv_numpy(rr)
    print(f"HRV: RMSSD={hrv['rmssd']:.1f}ms SDNN={hrv['sdnn']:.1f}ms")
else:
    print("HRV: insufficient peaks (synthetic noise signal)")

# 4. TCN inference
class TCNBlock(nn.Module):
    def __init__(self,i,o,d=1):
        super().__init__()
        self.conv=nn.Conv1d(i,o,3,padding=(3-1)*d,dilation=d)
        self.bn=nn.BatchNorm1d(o); self.relu=nn.ReLU()
        self.res=nn.Conv1d(i,o,1) if i!=o else nn.Identity()
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))[:,:,:x.size(2)]+self.res(x)

class DrowsinessTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1=TCNBlock(4,32,1);self.b2=TCNBlock(32,64,2)
        self.b3=TCNBlock(64,64,4);self.b4=TCNBlock(64,64,8)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),
                                nn.Dropout(0.1),nn.Linear(32,1))
    def forward(self,x):
        x=self.b4(self.b3(self.b2(self.b1(x))))
        return self.head(self.pool(x).squeeze(-1)).squeeze(-1)

model = DrowsinessTCN().eval()
for wpath in ["learned/models/task_b_tcn_cuda.pt",
              "learned/models/task_b_tcn_ddp.pt"]:
    if Path(wpath).exists():
        s = torch.load(wpath, map_location="cpu", weights_only=True)
        if isinstance(s,dict) and "model" in s: s=s["model"]
        model.load_state_dict(s, strict=False)
        print(f"Model: loaded {wpath}")
        break
else:
    print("Model: using random weights (no checkpoint found)")

mu  = window.mean(axis=1, keepdims=True)
std = window.std(axis=1,  keepdims=True)+1e-6
x   = torch.FloatTensor((window-mu)/std).unsqueeze(0)

times = []
for _ in range(100):
    t0 = time.perf_counter()
    with torch.no_grad():
        logit = model(x).item()
    times.append((time.perf_counter()-t0)*1000)
prob = float(torch.sigmoid(torch.tensor(logit)))
print(f"TCN: prob={prob:.4f} "
      f"median={sorted(times)[50]:.2f}ms")

# 5. Fusion + state machine
from tests.test_state_machine import state_machine
r = 0.40*sqi["total"]*prob + 0.20*0.1 + 0.10*0.3 + 0.30*0.0
thresh = 0.35
state = state_machine(r, thresh)
print(f"Fusion: r_total={r:.4f} thresh={thresh}")
print(f"State:  {state}")
print("\nCPU demo complete -- installation verified")
