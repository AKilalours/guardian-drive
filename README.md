
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:03070e,50:0a1220,100:00d4ff&height=220&section=header&text=Guardian%20Drive%E2%84%A2%20%F0%9F%9B%A1%EF%B8%8F&fontSize=58&fontColor=ffffff&fontAlignY=38&desc=Multimodal%20AI%20Driver%20Safety%20System&descAlignY=58&descSize=22&animation=fadeIn" width="100%"/>

<br/>

### Built by **Akila Lourdes Miriyala Francis** & **Akilan Manivannan**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Task%20B%20AUC-0.9514-00C853?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/nuScenes-18538%20Annotations-0088FF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PTB--XL-Real%20Clinical%20ECG-FF6B35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/GPT--2-Waypoint%20Transformer-9B59B6?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FastAPI-WebSocket-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/MediaPipe-FaceMesh%20468pt-FF6D00?style=for-the-badge"/>
</p>

<p align="center">
  <a href="https://github.com/AKilalours/guardian-drive">
    <img src="https://img.shields.io/badge/GitHub-Source%20Code-181717?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
  &nbsp;
  <a href="https://github.com/AKilalours/guardian-drive/blob/main/reports/training_summary.md">
    <img src="https://img.shields.io/badge/Training%20Results-AUC%200.9514-00C853?style=for-the-badge"/>
  </a>
</p>

<br/>

> **Production-architecture multimodal AI safety system** fusing real-time ECG cardiac screening, physiological drowsiness detection (WESAD TCN AUC 0.9514), camera-based driver monitoring, nuScenes BEV perception, and a GPT-2 style waypoint transformer — with automated hospital routing, Discord dispatch, voice alerts, and real GPS integration.

> **Every model trained on real clinical data. Every claim verified. No inflated metrics.**

<br/>

</div>

---

## 👥 Team Contributions

| Phase | Akila Lourdes Miriyala Francis | Akilan Manivannan |
|-------|-------------------------------|-------------------|
| **Data Pipeline** | WESAD physiological loader, PTB-XL ECG parser, SQI computation | nuScenes BEV annotation loader, ego pose matching, IMU replay |
| **Model Training** | WESAD TCN trainer (AUC 0.9514), PTB-XL arrhythmia screener | GPT-2 waypoint transformer, drowsiness CNN, Task C crash model |
| **Real-Time Server** | FastAPI WebSocket server, pipeline loop, voice alerts | nuScenes BEV streaming, scenario simulator, CARLA bridge |
| **Dashboard** | ECG waveform, risk ring, task panels, sleepiness classifier | BEV radar renderer, vehicle telemetry, map panel, GPS routing |
| **Integrations** | Discord webhook, GPS geolocation, OSM hospital routing | MediaPipe FaceMesh EAR/PERCLOS/yawn, OpenStreetMap POI |
| **MLOps** | Model versioning, SQI gating, claim guardrail, evaluation | Experiment tracking, training curves, results JSON, CHANGELOG |

---

## 🎯 System Performance

| Component | Metric | Result | Status |
|-----------|--------|--------|--------|
| Task B — WESAD TCN | AUC (held-out test) | **0.9514** | ✅ |
| Task A — PTB-XL | Real clinical training | RandomForest + CNN | ✅ |
| Waypoint Transformer | ADE on nuScenes mini | **7.70m** | ✅ |
| nuScenes BEV | Real 3D annotations | **18,538** | ✅ |
| nuScenes ego poses | Real driving data | **31,206** | ✅ |
| SQI gating | Abstain when noisy | Real-time | ✅ |
| Pipeline latency | WebSocket cycle | **~50ms** | ✅ |
| Camera FaceMesh | Landmark points | **468** | ✅ |
| Voice alert | Firing latency | **< 500ms** | ✅ |
| GPS accuracy | IP geolocation | Brooklyn ±3km | ✅ |

---

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SENSOR LAYER                                │
│        ECG · EDA · IMU · Camera · GPS (IP/CoreLocation)             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              SIGNAL QUALITY INDEX (SQI)                             │
│   ECG quality · EDA contact · Resp quality · Motion level           │
│            Abstain if quality insufficient                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FEATURE EXTRACTION                              │
│   HR/HRV/RMSSD/SDNN · R-peak · FFT · Resp rate                      │
│   EDA SCL/SCR · IMU posture · EAR/PERCLOS/MAR                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FUSION ENGINE                                  │
│                                                                     │
│  Task A: Arrhythmia          Task B: Drowsiness                     │
│  PTB-XL RandomForest         WESAD TCN AUC 0.9514                   │
│  1D CNN ECG features         Camera EAR/PERCLOS fusion              │
│                                                                     │
│  Task C: Crash Detection     Task D: Neuro Risk Proxy               │
│  IMU g-force threshold       HRV + EDA heuristic                    │
│                                                                     │
│  Waypoint Transformer (GPT-2 causal attention)                      │
│  nuScenes ego poses → next 5 waypoints (ADE 7.70m)                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  SAFETY STATE MACHINE                               │
│                                                                     │
│  NOMINAL → ADVISORY → CAUTION → PULLOVER → ESCALATE                 │
│                                                                     │
│  NOMINAL:   Monitor only                                            │
│  ADVISORY:  Voice · Nearest cafe (OSM) · Map panel                  │
│  CAUTION:   Voice · Nearest motel (OSM) · Map panel                 │
│  PULLOVER:  Voice · Pull over NOW · Rest stop routing               │
│  ESCALATE:  Hospital routing · Discord auto-fire · 911              │
│             Autopilot banner · Voice emergency alert                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│               REAL-TIME DASHBOARD (WebSocket)                       │
│  ECG waveform · nuScenes BEV radar · MediaPipe camera               │
│  Risk ring · Vehicle telemetry · GPS + inline Google Maps           │
│  POI banner · Emergency panel · Sleepiness vs Drowsiness            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Models & Training

### Task B — Drowsiness Detection (WESAD TCN)

> **Akila Lourdes Miriyala Francis** — TCN architecture & training pipeline
> **Akilan Manivannan** — Feature engineering & evaluation framework

| Property | Value |
|----------|-------|
| Dataset | WESAD — Wearable Stress and Affect Detection |
| Subjects | 15 (S2–S17, real physiological recordings) |
| Windows | 2,874 (60s sliding, 50% overlap) |
| Class balance | Alert: 2,480 / Drowsy: 394 |
| Model | Temporal Convolutional Network (PyTorch) |
| Channels | 4 — ECG, EDA, Temperature, Respiration |
| **Best AUC** | **0.9514** (epoch 11/15, 80/20 split) |

Training curve:

| Epoch | Loss | AUC |
|-------|------|-----|
| 1 | 0.3867 | 0.8330 |
| 5 | 0.2917 | 0.9212 |
| 8 | 0.2749 | 0.9382 |
| **11** | **0.2295** | **0.9514** ← best |
| 15 | 0.2227 | 0.9482 |

### Task A — Arrhythmia Screening (PTB-XL)

> **Akila Lourdes Miriyala Francis** — Data loading, feature extraction
> **Akilan Manivannan** — Model training, evaluation pipeline

| Property | Value |
|----------|-------|
| Dataset | PTB-XL PhysioNet — real clinical ECG |
| Model | RandomForest + 1D CNN (FFT + HRV features) |
| Classes | Normal / AFib / Tachycardia / Bradycardia / ST-T |

### Waypoint Transformer — GPT-2 Architecture

> **Akilan Manivannan** — Transformer architecture
> **Akila Lourdes Miriyala Francis** — nuScenes data pipeline, training

| Property | Value |
|----------|-------|
| Architecture | Causal self-attention (GPT-2 style) |
| Dataset | nuScenes mini — 31,206 real ego poses |
| Task | Predict next 5 waypoints from 10 past states |
| Layers | 3 blocks · 4 heads · d=64 |
| Parameters | 151,626 |
| **ADE** | **7.70m** on held-out nuScenes scenes |

### nuScenes BEV Perception

> **Akilan Manivannan** — BEV loader, object transformation
> **Akila Lourdes Miriyala Francis** — Dashboard rendering, radar display

| Property | Value |
|----------|-------|
| Dataset | nuScenes mini v1.0 |
| Scenes | 10 real driving (Singapore + Boston) |
| Annotations | **18,538 real 3D bounding boxes** |
| Ego poses | 31,206 real vehicle positions |
| Classes | car · truck · bus · pedestrian · motorcycle · bicycle · cone · barrier |

---

## 🚨 Escalation Flow

```
Cardiac event / crash / sustained drowsiness → ESCALATE

  1. Voice fires immediately
     "Emergency. Medical event detected. Routing to nearest hospital."

  2. OpenStreetMap query (real-time, no API key)
     Nearest hospital within 3km — name, distance, ETA

  3. Google Maps panel appears in dashboard
     Real route from current GPS to hospital, dismissible

  4. Discord webhook fires automatically
     Rich embed: GPS + task scores + hospital + maps link

  5. Emergency panel activates
     Call 911 button · ER navigation · Copy dispatch script

  6. Autopilot banner
     "AUTOPILOT ENGAGED — Routing to emergency care"
```

---

## ✅ Honest Claims

| Component | Status | Evidence |
|-----------|--------|---------|
| Task B AUC 0.9514 | ✅ Real WESAD training | `learned/results/task_b_eval.json` |
| Task A PTB-XL | ✅ Real clinical data | `artifacts/task_a_model_real_ptbxl.joblib` |
| nuScenes BEV | ✅ Real 3D annotations | `acquisition/nuscenes_bev.py` |
| Waypoint transformer | ✅ Real PyTorch training | `learned/models/waypoint_transformer.pt` |
| SQI gating | ✅ Abstains when noisy | `sqi/compute.py` |
| GPS routing | ✅ Real OSM Overpass API | `integrations/navigation_osm.py` |
| Discord dispatch | ✅ Real webhook auto-fire | `integrations/discord_webhook.py` |
| Voice alerts | ✅ macOS say command | server/app.py |
| Medical grade | ❌ Not clinically validated | `policy/fusion.py` claim_guardrail |

---

## 🚀 Quick Start

```bash
git clone https://github.com/AKilalours/guardian-drive.git
cd guardian-drive
cp .env.example .env
pip install -r requirements.txt

GD_LIVE_ONLY=0 GD_ENABLE_WEBCAM=1 GD_WINDOW_SEC=8 GD_STEP_SEC=2 \
GD_TASK_B_MLP_WEIGHTS=learned/models/task_b_tcn.pt GD_VOICE=1 \
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000

open http://127.0.0.1:8000

python integrations/gps_mac.py --watch
```

### Demo Scenarios

| Scenario | Demonstrates |
|----------|-------------|
| Normal | Baseline monitoring |
| Drowsy | Fatigue detection + cafe POI + voice |
| AFib | Arrhythmia detection + ECG anomaly |
| Tachycardia | High HR cardiac alert |
| Bradycardia | Low HR cardiac alert |
| Crash Mild | IMU impact detection |
| **Crash Severe** | Full escalation — hospital + Discord + 911 |

### Retrain Models

```bash
python learned/task_b_trainer.py --data_dir datasets/WESAD/WESAD
python learned/task_a_trainer.py --data_dir datasets/ptbdb/1.0.0
python learned/waypoint_transformer.py --train
```

---

## 📁 Project Structure

```
guardian-drive/
├── acquisition/
│   ├── nuscenes_bev.py          Real nuScenes 3D annotation loader
│   ├── seat_ecg_node.py         Seat ECG hardware node
│   └── simulator.py             Scenario simulator
├── artifacts/
│   ├── task_a_model_real_ptbxl.joblib
│   └── drowsiness_cnn_subjectsplit.pt
├── datasets/
│   ├── WESAD/                   15 subjects physiological data
│   ├── ptbdb/                   PTB-XL clinical ECG records
│   └── nuscenes/v1.0-mini/      18,538 real 3D annotations
├── features/                    ECG/EDA/IMU/camera feature extraction
├── integrations/
│   ├── voice_alerts.py          macOS voice by alert level
│   ├── gps_mac.py               Real GPS CoreLocation/IP
│   ├── vision_webcam.py         MediaPipe EAR/PERCLOS/yawn
│   ├── navigation_osm.py        OpenStreetMap hospital routing
│   └── discord_webhook.py       Auto-fire on ESCALATE
├── learned/
│   ├── task_b_trainer.py        WESAD TCN trainer
│   ├── task_a_trainer.py        PTB-XL CNN trainer
│   ├── waypoint_transformer.py  GPT-2 trajectory predictor
│   ├── models/
│   │   ├── task_b_tcn.pt        AUC 0.9514
│   │   └── waypoint_transformer.pt  ADE 7.70m
│   └── results/
│       ├── task_b_eval.json
│       └── waypoint_transformer_eval.json
├── policy/
│   ├── fusion.py                Multi-task fusion engine
│   └── state_machine.py         NOMINAL to ESCALATE
├── server/
│   ├── app.py                   FastAPI WebSocket v4.3
│   └── static/
│       └── GuardianDrive_Dashboard.html
├── sqi/                         Signal quality index
├── sim/carla_bridge.py          CARLA autopilot bridge
├── LICENSE
├── MLOPS.md
├── ARCHITECTURE.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── SAFETY.md
├── .env.example
└── requirements.txt
```

---

## 🔧 Tech Stack

| Layer | Technology | Who |
|-------|-----------|-----|
| Deep Learning | PyTorch 2.x — TCN, CNN, GPT-2 | Both |
| ML | Scikit-learn — RandomForest | Akila |
| Computer Vision | MediaPipe TFLite — FaceMesh | Akilan |
| Signal Processing | SciPy — ECG, HRV, R-peak | Akila |
| AV Perception | nuScenes — real 3D BEV | Akilan |
| Backend | FastAPI + WebSocket | Akila |
| Navigation | OpenStreetMap Overpass API | Akilan |
| Alerts | macOS say + Discord webhook | Both |
| GPS | IP geolocation + CoreLocation | Akila |

---

## 📚 References

| Reference | Application |
|-----------|-------------|
| Tesla FSD Neural Planner | GPT-2 waypoint transformer |
| openpilot (comma.ai) | Driver monitoring architecture |
| TPVFormer | BEV occupancy representation |
| WESAD Dataset (Schaeck et al.) | Task B training |
| PTB-XL (Wagner et al.) | Task A training |
| nuScenes (Caesar et al.) | BEV + waypoint training |
| MediaPipe FaceMesh | 468-point EAR/PERCLOS/yawn |

---

## ⚠️ Safety Disclaimer

Guardian Drive is a research and portfolio project. It is **not a medical device** and has **not been clinically validated**. Do not use to make real medical or safety decisions.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00d4ff,50:0a1220,100:03070e&height=140&section=footer" width="100%"/>

<br/>

**Akila Lourdes Miriyala Francis** · **Akilan Manivannan**

*Guardian Drive™ · PyTorch · nuScenes · PTB-XL · WESAD · FastAPI · MediaPipe · GPT-2*

**© 2026 — All Rights Reserved**

