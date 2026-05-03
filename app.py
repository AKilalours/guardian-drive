"""
Guardian Drive -- Permanent Live Demo
Hosted on HuggingFace Spaces

All features from the project in one Gradio interface:
- Task B: Low-arousal state classification (WESAD TCN AUC 0.9738)
- Task A: ECG arrhythmia screening (PTB-XL)
- Waypoint predictor: causal transformer on nuScenes
- BEV visualization: real nuScenes annotations
- Benchmark: TorchScript latency results
- System overview

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import json
import time
import os
from pathlib import Path

# ── Model definition ──────────────────────────────────────────────
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        pad = (3-1)*dilation
        self.conv = nn.Conv1d(in_ch, out_ch, 3, padding=pad, dilation=dilation)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out[:,:,:x.size(2)] + self.res(x)

class DrowsinessTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1   = TCNBlock(4,  32, dilation=1)
        self.b2   = TCNBlock(32, 64, dilation=2)
        self.b3   = TCNBlock(64, 64, dilation=4)
        self.b4   = TCNBlock(64, 64, dilation=8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32,1))
    def forward(self, x):
        x = self.b4(self.b3(self.b2(self.b1(x))))
        return self.head(self.pool(x).squeeze(-1))

class CausalSelfAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h; self.d = d
        self.qkv  = nn.Linear(d, 3*d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).split(C,dim=2)
        q=q.view(B,T,self.h,-1).transpose(1,2)
        k=k.view(B,T,self.h,-1).transpose(1,2)
        v=v.view(B,T,self.h,-1).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * (C//self.h)**-0.5
        mask= torch.triu(torch.ones(T,T,device=x.device),1).bool()
        att = att.masked_fill(mask,-1e9)
        att = torch.softmax(att,-1)
        y   = (att @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.proj(y)

class WaypointTransformer(nn.Module):
    def __init__(self, d=64, h=4, n_layers=3, seq_in=10, seq_out=5):
        super().__init__()
        self.embed = nn.Linear(4, d)
        self.blocks= nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d),
                'att': CausalSelfAttention(d,h),
                'ln2': nn.LayerNorm(d),
                'ffn': nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
            }) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2*seq_out)
        self.seq_out = seq_out
    def forward(self, x):
        x = self.embed(x)
        for b in self.blocks:
            x = x + b['att'](b['ln1'](x))
            x = x + b['ffn'](b['ln2'](x))
        x = self.ln_f(x)
        return self.head(x[:,-1,:]).view(-1, self.seq_out, 2)

# ── Load models ───────────────────────────────────────────────────
def load_tcn():
    m = DrowsinessTCN()
    for wpath in ["learned/models/task_b_tcn_cuda.pt",
                  "learned/models/task_b_tcn.pt",
                  "task_b_tcn_cuda.pt"]:
        if Path(wpath).exists():
            state = torch.load(wpath, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            m.load_state_dict(state, strict=False)
            print(f"TCN loaded: {wpath}")
            break
    m.eval()
    return m

def load_waypoint():
    m = WaypointTransformer()
    for wpath in ["learned/models/waypoint_transformer.pt",
                  "waypoint_transformer.pt"]:
        if Path(wpath).exists():
            state = torch.load(wpath, map_location="cpu")
            m.load_state_dict(state, strict=False)
            print(f"Waypoint loaded: {wpath}")
            break
    m.eval()
    return m

TCN_MODEL = load_tcn()
WPT_MODEL = load_waypoint()

# ── WESAD simulator ───────────────────────────────────────────────
def simulate_wesad(condition: str, noise: float = 0.05):
    """Simulate 4-channel physiological window for a given condition."""
    np.random.seed(42)
    t = np.linspace(0, 60, 4200)

    if condition == "Alert (Baseline)":
        ecg  = 0.8*np.sin(2*np.pi*1.2*t) + noise*np.random.randn(4200)
        eda  = 0.3 + 0.05*np.sin(2*np.pi*0.05*t) + noise*np.random.randn(4200)
        temp = 36.5 + 0.1*np.sin(2*np.pi*0.01*t) + 0.01*np.random.randn(4200)
        resp = 0.6*np.sin(2*np.pi*0.25*t) + noise*np.random.randn(4200)
    elif condition == "Low Arousal (Proxy Drowsy)":
        ecg  = 0.5*np.sin(2*np.pi*0.9*t) + 0.15*np.random.randn(4200)
        eda  = 0.1 + 0.02*np.sin(2*np.pi*0.02*t) + 0.05*np.random.randn(4200)
        temp = 36.2 + 0.05*np.sin(2*np.pi*0.005*t) + 0.01*np.random.randn(4200)
        resp = 0.3*np.sin(2*np.pi*0.18*t) + 0.1*np.random.randn(4200)
    else:  # Stress
        ecg  = 1.2*np.sin(2*np.pi*1.8*t) + 0.2*np.random.randn(4200)
        eda  = 0.8 + 0.2*np.sin(2*np.pi*0.1*t) + 0.1*np.random.randn(4200)
        temp = 36.8 + 0.2*np.sin(2*np.pi*0.02*t) + 0.02*np.random.randn(4200)
        resp = 0.9*np.sin(2*np.pi*0.35*t) + 0.15*np.random.randn(4200)

    return np.stack([ecg, eda, temp, resp])  # [4, 4200]

# ── Tab 1: Task B demo ────────────────────────────────────────────
def run_taskb(condition, noise_level, custom_ecg, custom_eda):
    """Run WESAD TCN inference on simulated or custom physiological input."""
    t0 = time.perf_counter()

    if custom_ecg and custom_eda:
        try:
            ecg_vals = np.array([float(x) for x in custom_ecg.split(",")[:4200]])
            eda_vals = np.array([float(x) for x in custom_eda.split(",")[:4200]])
            n = min(len(ecg_vals), len(eda_vals), 4200)
            if n < 100:
                return "Need at least 100 values", "", ""
            ecg_vals = np.pad(ecg_vals[:n], (0, 4200-n))
            eda_vals = np.pad(eda_vals[:n], (0, 4200-n))
            temp = np.full(4200, 36.5)
            resp = np.sin(2*np.pi*0.25*np.linspace(0,60,4200))
            window = np.stack([ecg_vals, eda_vals, temp, resp])
        except Exception as e:
            return f"Parse error: {e}", "", ""
    else:
        window = simulate_wesad(condition, noise=float(noise_level))

    # Normalize per-channel
    mu  = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True) + 1e-6
    window = (window - mu) / std

    x = torch.FloatTensor(window).unsqueeze(0)  # [1, 4, 4200]
    with torch.no_grad():
        logit = TCN_MODEL(x).item()
    prob  = torch.sigmoid(torch.tensor(logit)).item()
    latency_ms = (time.perf_counter()-t0)*1000

    if prob > 0.65:
        state = "LOW-AROUSAL (Proxy Drowsy Risk)"
        emoji = "CAUTION"
    elif prob > 0.35:
        state = "BORDERLINE"
        emoji = "ADVISORY"
    else:
        state = "ALERT"
        emoji = "NOMINAL"

    result = f"""
### Result: {emoji} -- {state}

| Metric | Value |
|--------|-------|
| Low-arousal probability | **{prob:.4f}** |
| Raw logit | {logit:.4f} |
| Inference latency | {latency_ms:.2f} ms |
| Model | WESAD TCN (AUC 0.9738, window-level split) |
| Input | {condition} simulation |

**Disclaimer**: This classifies WESAD low-arousal physiological states
as a proxy for drowsiness risk. Not a medical device. Not validated
for clinical use. AUC 0.9738 uses window-level split with known
leakage risk -- subject-independent evaluation not yet run.
"""
    channels = f"""
### Input Channels (4200 samples each)

| Channel | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| ECG | {window[0].mean():.3f} | {window[0].std():.3f} | {window[0].min():.3f} | {window[0].max():.3f} |
| EDA | {window[1].mean():.3f} | {window[1].std():.3f} | {window[1].min():.3f} | {window[1].max():.3f} |
| Temp | {window[2].mean():.3f} | {window[2].std():.3f} | {window[2].min():.3f} | {window[2].max():.3f} |
| Resp | {window[3].mean():.3f} | {window[3].std():.3f} | {window[3].min():.3f} | {window[3].max():.3f} |
"""
    return result, channels

# ── Tab 2: Waypoint predictor ─────────────────────────────────────
def run_waypoint(scenario):
    """Run causal transformer waypoint predictor on a scenario."""
    t0 = time.perf_counter()

    scenarios = {
        "Straight highway": [(i*5.0, 0.0, 0.0) for i in range(10)],
        "Left turn": [(i*3.0, -i*0.8, -0.1*i) for i in range(10)],
        "Right turn": [(i*3.0,  i*0.8,  0.1*i) for i in range(10)],
        "Approaching stop": [(i*2.0*(1-i/20), 0.0, 0.0) for i in range(10)],
        "Lane change left": [(i*4.0, -i*0.3 if i>5 else 0, 0) for i in range(10)],
    }

    poses = scenarios.get(scenario, scenarios["Straight highway"])
    states = []
    for x,y,theta in poses:
        states.extend([x/50.0, y/10.0, np.cos(theta), np.sin(theta)])

    inp = torch.FloatTensor(states).view(1, 10, 4)
    with torch.no_grad():
        pred = WPT_MODEL(inp).squeeze(0).numpy()

    latency_ms = (time.perf_counter()-t0)*1000

    rows = "\n".join([
        f"| t+{k+1} | {pred[k,0]*50:.2f} m | {pred[k,1]*10:.2f} m |"
        for k in range(5)])

    result = f"""
### Waypoint Prediction -- {scenario}

**Input**: 10 past ego states (x, y, cos theta, sin theta)
**Output**: 5 future waypoints (dx, dy offsets)

| Timestep | Delta-X | Delta-Y |
|----------|---------|---------|
{rows}

| Metric | Value |
|--------|-------|
| Inference latency | {latency_ms:.2f} ms |
| Model parameters | 151,626 |
| Architecture | 3-layer causal self-attention (GPT-2 style) |
| Training | 264 windows from nuScenes mini (31,206 pose records) |
| ADE on mini split | 7.70 m |

**Note**: This is a toy baseline on nuScenes mini (10 scenes).
Not a competitive AV trajectory predictor.
"""
    return result

# ── Tab 3: Benchmark results ──────────────────────────────────────
def show_benchmark():
    bench_path = "learned/results/libtorch_benchmark.json"
    if Path(bench_path).exists():
        data = json.loads(Path(bench_path).read_text())
        cpu  = data.get("cpu", {})
        mps  = data.get("mps", {})

        cpu_rows = "\n".join([
            f"| {k} | {v['median_ms']} ms | {v['p95_ms']} ms | {v['tput']} seq/s |"
            for k,v in cpu.items()])
        mps_rows = "\n".join([
            f"| {k} | {v['median_ms']} ms | {v['p95_ms']} ms | {v['tput']} seq/s |"
            for k,v in mps.items()]) if mps else "Not measured"

        result = f"""
### LibTorch / TorchScript Benchmark Results
**Model**: WESAD TCN (AUC 0.9738) | **Params**: {data.get('params',36161):,} | **FP32**: {data.get('fp32_mb',0.14):.2f} MB

#### CPU (Apple M4)
| Batch | Median | p95 | Throughput |
|-------|--------|-----|-----------|
{cpu_rows}

#### MPS (Apple M4 GPU)
| Batch | Median | p95 | Throughput |
|-------|--------|-----|-----------|
{mps_rows}

#### Key Results
- **MPS batch=1**: 0.49ms median, 2043 seq/s
- **MPS batch=8**: 3.37ms median, 2373 seq/s
- **CPU batch=1**: 1.91ms median, 524 seq/s
- TorchScript (.pt): 0.18 MB -- same format used by LibTorch C++ runtime
- ONNX (.onnx): 0.3 MB -- cross-platform, NVIDIA CUDA via ORT
- CoreML (.mlpackage): 0.2 MB -- Apple Neural Engine
"""
    else:
        result = """
### Benchmark Results (from paper)

| Config | Median | p95 | Throughput |
|--------|--------|-----|-----------|
| CPU bs=1 (Apple M4) | 1.91 ms | 2.73 ms | 524 seq/s |
| CPU bs=8 | 22.19 ms | 29.18 ms | 361 seq/s |
| MPS bs=1 (Apple M4 GPU) | 0.49 ms | 0.58 ms | 2044 seq/s |
| MPS bs=8 | 3.37 ms | 4.22 ms | 2373 seq/s |

TorchScript: 0.18 MB | ONNX: 0.3 MB | CoreML: 0.2 MB
"""
    return result

# ── Tab 4: System overview ────────────────────────────────────────
OVERVIEW = """
## Guardian Drive -- System Overview

**Built by Akilan Manivannan & Akila Lourdes Miriyala Francis**
Long Island University, Brooklyn, NY

GitHub: [AkilanManivannanak/guardian-drive](https://github.com/AkilanManivannanak/guardian-drive)
| [AKilalours/guardian-drive](https://github.com/AKilalours/guardian-drive)

---

### What this demo shows

| Tab | Module | Status |
|-----|--------|--------|
| Task B Inference | WESAD TCN low-arousal classifier | AUC 0.9738 (window-level split) |
| Waypoint Predictor | Causal transformer on nuScenes | ADE 7.70m (mini, 264 windows) |
| Benchmark | TorchScript / MPS latency | MPS 0.49ms @bs=1 |
| Overview | System documentation | This tab |

### Datasets Used
| Dataset | Purpose | Size |
|---------|---------|------|
| WESAD | Task B training | 15 subjects, 28,930 windows |
| PTB-XL | Task A arrhythmia (placeholder) | 21,837 clinical ECG |
| nuScenes mini | BEV, VO, SLAM, waypoint | 10 scenes, 18,538 annotations |

### Models
| Model | File | Size | Result |
|-------|------|------|--------|
| WESAD TCN | task_b_tcn_cuda.pt | 0.32 MB | AUC 0.9738 |
| CoreML export | guardian_drive_tcn.mlpackage | 0.2 MB | Apple ANE |
| ONNX export | wesad_tcn.onnx | 0.3 MB | Cross-platform |
| TorchScript | wesad_tcn_scripted.pt | 0.18 MB | LibTorch C++ |
| Waypoint TCN | waypoint_transformer.pt | -- | ADE 7.70m |

### Scientific Limitations
1. **Task B AUC 0.9738** uses window-level 80/20 split with known leakage.
   Subject-independent LOSO evaluation not yet run.
2. **WESAD** classifies low-arousal physiological states, not validated
   driver drowsiness.
3. **Task A** (arrhythmia) has no diagnostic performance metrics.
   Architectural placeholder only.
4. **Not a medical device**. Not clinically validated.
5. **Emergency workflow** is a prototype simulation -- not real dispatch.

### Pipeline Latency (Apple M4)
- Sequential: ~60ms median | ~87ms p95
- Parallel (Tasks A+B): ~45ms median | ~70ms p95
- Emergency response (prototype): <2s end-to-end
"""

# ── Build Gradio app ──────────────────────────────────────────────
with gr.Blocks(
    title="Guardian Drive -- Live Demo",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"),
    css="""
    .header-text { text-align: center; }
    .warning-box { background: #fff3cd; padding: 10px; border-radius: 8px;
                   border-left: 4px solid #ffc107; margin: 8px 0; }
    """
) as demo:

    gr.Markdown("""
# Guardian Drive -- Live Demo
### Multimodal AI Driver Monitoring Research Prototype
**Akilan Manivannan & Akila Lourdes Miriyala Francis** | Long Island University

> **Not a medical device. Not clinically validated. Research prototype only.**

[![GitHub](https://img.shields.io/badge/GitHub-guardian--drive-181717?logo=github)](https://github.com/AkilanManivannanak/guardian-drive)
    """)

    with gr.Tabs():

        # ── Tab 1: Task B ─────────────────────────────────────────
        with gr.TabItem("Task B -- Low-Arousal Classifier"):
            gr.Markdown("""
### WESAD TCN Inference (AUC 0.9738, window-level split)
Classify a 60-second physiological window as alert vs low-arousal.
**Note**: AUC 0.9738 uses window-level split with known data leakage.
Subject-independent evaluation is required future work.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    condition = gr.Dropdown(
                        choices=["Alert (Baseline)",
                                 "Low Arousal (Proxy Drowsy)",
                                 "Stress"],
                        value="Alert (Baseline)",
                        label="Physiological Condition (simulated)")
                    noise = gr.Slider(0.0, 0.3, value=0.05, step=0.01,
                                      label="Noise Level")
                    gr.Markdown("**Or paste custom CSV values (comma-separated):**")
                    custom_ecg = gr.Textbox(label="Custom ECG values (optional)",
                                            placeholder="0.1, -0.2, 0.8, ...",
                                            lines=2)
                    custom_eda = gr.Textbox(label="Custom EDA values (optional)",
                                            placeholder="0.3, 0.31, 0.29, ...",
                                            lines=2)
                    btn_taskb = gr.Button("Run Inference", variant="primary")

                with gr.Column(scale=2):
                    out_result   = gr.Markdown()
                    out_channels = gr.Markdown()

            btn_taskb.click(
                run_taskb,
                inputs=[condition, noise, custom_ecg, custom_eda],
                outputs=[out_result, out_channels])

        # ── Tab 2: Waypoint ───────────────────────────────────────
        with gr.TabItem("Waypoint Predictor"):
            gr.Markdown("""
### Causal Transformer Waypoint Predictor (nuScenes mini)
Predict 5 future waypoints from 10 past ego states.
**ADE 7.70m on nuScenes mini (264 training windows). Toy baseline only.**
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    scenario = gr.Dropdown(
                        choices=["Straight highway", "Left turn",
                                 "Right turn", "Approaching stop",
                                 "Lane change left"],
                        value="Straight highway",
                        label="Driving Scenario")
                    btn_wp = gr.Button("Predict Waypoints", variant="primary")
                with gr.Column(scale=2):
                    out_wp = gr.Markdown()
            btn_wp.click(run_waypoint, inputs=[scenario], outputs=[out_wp])

        # ── Tab 3: Benchmark ──────────────────────────────────────
        with gr.TabItem("LibTorch Benchmark"):
            gr.Markdown("### TorchScript / MPS / CPU Latency Benchmark")
            btn_bench = gr.Button("Show Benchmark Results", variant="primary")
            out_bench = gr.Markdown()
            btn_bench.click(show_benchmark, inputs=[], outputs=[out_bench])

        # ── Tab 4: Overview ───────────────────────────────────────
        with gr.TabItem("System Overview"):
            gr.Markdown(OVERVIEW)

if __name__ == "__main__":
    demo.launch(share=True)
