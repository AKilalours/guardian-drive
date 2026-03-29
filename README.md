# Guardian Drive™ v7.0
## Real-Data Driver Safety Pipeline (Recruiter-Credible)

> **A safety-critical, multi-sensor physiological monitoring pipeline
> that screens for driver incapacitation risk from arrhythmia, drowsiness,
> and crash events — with SQI-gated abstain behavior, persistent state machine
> escalation, and reproducible LOSO evaluation.**

---

## Scope (Honest)

This system is an **incapacitation risk screening tool**, not a medical diagnostic device.

| ✅ Defensible claim | ❌ Not claimed |
|---|---|
| Single-lead ECG abnormal rhythm screening with abstain behavior | STEMI diagnosis |
| Drowsiness/fatigue risk scoring from multi-channel physiology | Blood alcohol measurement |
| Crash event detection and severity classification via IMU | Autopilot routing |
| Emergency escalation policy + dispatch packets | Direct PSAP/NG911 integration |
| **Human-in-the-loop 911** (one-tap dial + auto dispatch script) | Auto-dialing 911 from code |
| **Real hospital lookup** via OpenStreetMap Overpass (GPS → nearest ER) | Autonomous cath lab routing |
| **CARLA simulation**: route-to-hospital on ESCALATE | OEM Autopilot/vehicle-control takeover |
| FAR/hour-reported evaluation with LOSO splits | FDA-cleared diagnostics |


---

## What you get in v5.0 (what makes it “AI Engineer grade”)

This release adds **production-style plumbing** so the project looks like a real safety/robotics stack:

- **Integration interfaces** (`integrations/`): GPS, navigation advisory, telephony (console + optional Twilio), vehicle-control *stubs*.
- **Replayable event logs** (`runs/*.jsonl`) for audits, debugging, and offline evaluation.
- **Streamlit UI** (`ui/streamlit_app.py`) for live simulation + replay visualization.
- **Optional learned model hooks** (`learned/`): tiny PyTorch models (MLP/TCN) designed for Raspberry Pi 5.

### Hard constraints (reality)
- **Autopilot / steering / braking is OEM-only.** This repo provides *interfaces* + simulation stubs — not hacks.
- **911/PSAP integration is not shipped.** We provide *human-in-the-loop* dialing + a generated dispatch script. Real PSAP integration requires a certified eCall/NG911 provider.

---

## Real-World Demo Integrations (what is actually real)

### Free emergency contact: Discord webhook (real)
Set a Discord webhook and Guardian Drive will post a dispatch packet when **ESCALATE** triggers.

```bash
export GD_DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

### Human-in-the-loop 911 (realistic + safe)
The dashboard includes:
- **CALL 911** button (`tel:911`)
- **Copy Dispatch Script** button (auto-generated details for you to read)

### Nearby hospital routing (real)
Enable **GPS** in the dashboard. The server uses **OpenStreetMap Overpass** to find the nearest hospital/ER and provides a route link.

### CARLA “autopilot to hospital” (simulation)
This repo can be paired with a CARLA bridge that starts route-to-hospital when **ESCALATE** triggers.
This must be labeled **SIMULATION** in any recruiter/demo context.


## What's new in v4.1 (Engineering upgrades)

- **Crash policy fixed:** severe crash escalates immediately (no 5s hold) + optional belt corroboration.
- **Respiration features fixed:** FFT-based respiration rate estimator (avoids peak double-counting under noise).
- **Metrics fixed:** ROC-AUC / PR-AUC now computed from probability scores (not raw confidences).
- **Task-specific evaluation slices:** drowsiness metrics are computed on alert-vs-drowsy scenarios (not mixed with arrhythmia/crash), matching how real datasets are scoped.

---

---

## Three Flagship Tasks

### Task A — Arrhythmia Risk Screening
Detects AFib, Tachycardia, Bradycardia from single-lead ECG.

**Acceptance criteria:**
- Sensitivity ≥ 0.80 at FAR ≤ 2/hour ← *target Phase 2*
- Time-to-detection ≤ 30s (one window)
- ECE calibration < 0.10 ← *temperature scaling implemented, tuning needed*

**Current baseline (rule-based, simulated data):**
- Sensitivity: 1.00 | FAR/hr: 0.00 | ECE: 0.30 (abstain: 30%)
- ECE gap: overconfident on abstained windows — temperature T needs fit on held-out val set

### Task B — Drowsiness / Fatigue Screening
Multi-channel fusion: ECG HRV + EDA + Respiration + IMU posture.

**Acceptance criteria:**
- AUC ≥ 0.75 on LOSO split ← *target Phase 2*
- Robust under motion, contact loss, temperature changes
- Abstain when SQI < threshold ✅ implemented

**Current baseline:**
- Sensitivity: 0.136 | Specificity: 0.99 | FAR/hr: 1.09
- Gap: drowsy/fatigued scenarios overlap heavily with stressed/normal in physio features.
  Requires trained classifier (WESAD-style dataset) — not fixable by rules alone.

> **Important reality check:** WESAD is a *stress / affect* dataset (wrist + chest sensors).  
> It does **not** contain "driver drowsiness / sleep" labels, and it won't magically solve drowsiness detection.  
> If you want **real** drowsiness, you need **driver drowsiness datasets** (video/eye closure) and/or a **webcam pipeline**.

### Real Webcam Drowsiness (Vision) — NEW
If your goal is “use my webcam and detect drowsiness / sleep”, use the `vision/` module:

```bash
pip install -r requirements-vision.txt

# Heuristic (no training): EAR + PERCLOS + yawn on your live webcam
python -m vision.run_webcam_drowsy --camera 0

# (Optional) Build a REAL labeled dataset with your webcam (manual labeling)
python -m vision.capture_dataset --camera 0

# (Optional) Train a lightweight CNN on that dataset
python -m vision.train_cnn --data data/vision --out runs/vision_model.pt

# (Optional) Run the trained CNN live
python -m vision.run_webcam_cnn --ckpt runs/vision_model.pt
```

**Mac/Conda note:** If you hit `AttributeError: module 'mediapipe' has no attribute 'solutions'`, you're on MediaPipe 0.10.30+. This repo uses the legacy `mp.solutions.*` API, so install the pinned versions in `requirements-vision.txt` (and ideally use a clean Python 3.11 env).

**What’s real here:** it runs on your camera feed, in real time.  
**What’s still required:** if you want a *trained* model, you need labels (your own or public drowsiness datasets).


### Task C — Crash Detection + Severity
IMU-based crash detection with severity classification (mild/severe).

**Acceptance criteria:**
- Latency ≤ 1.5s after onset ✅ p50=120ms p95=120ms
- Near-zero missed severe crashes ← *sensitivity gap in windowed evaluation*
- FAR ≤ 1/hour ✅ 0.00/hr

**Gap:** Crash events fall in few windows per session. Window-level sensitivity
is low because most windows post-crash are labeled normal. Fix: event-level
detection metric (not window-level) in Phase 2.

---

## Architecture

```
SMART SEATBELT (9 channels)
ECG 250Hz · EDA 16Hz · Resp 50Hz · IMU 100Hz · Temp · Alcohol · Belt

    ↓ raw SensorFrame (typed dataclass, JSON-serializable)

SQI STATE ESTIMATOR
  ecg_quality · eda_contact · resp_quality · motion_level · belt_worn
  → abstain: bool  (refuse decision when < 2 channels usable)

    ↓ SQIState (per-channel probabilities)

FEATURE EXTRACTION (quality-gated)
  ECG: HR, HRV-RMSSD, HRV-SDNN, RR-irregularity, P-wave fraction, QRS width, ST
  EDA: SCL mean/slope, SCR rate/amplitude
  Resp: rate, irregularity, shallow flag
  IMU: accel RMS, jerk peak, posture score, crash flag + severity

    ↓ FeatureBundle (typed, per-channel, channel absent = field is None)

TASK MODELS
  Task A: ArrhythmiaScreener → ArrhythmiaResult (cls, confidence, abstained, reason)
  Task B: DrowsinessScreener → DrowsinessResult (score, channel contributions, abstained)
  Task C: CrashDetector      → CrashResult (detected, severity, latency_ms, g_peak)

    ↓ FusionEngine → RiskState

SAFETY-CRITICAL POLICY STATE MACHINE
  Persistence: CAUTION requires 15s hold, PULLOVER 20s, ESCALATE 5s
  Corroboration: PULLOVER+ requires ≥2 channels agreeing
  Cooldown: no repeat alert within 90s
  Abstain propagation: INACTIVE when SQI abstains
  Every action: log_reason + corroborated_by (fully traceable)

    ↓ PolicyAction (level, message, escalate_911_sim, log_reason, persistence_sec)
```

---

## Evaluation Methodology

This project reports metrics that matter for safety-critical systems.

### Primary metric: FAR/hour
False alarm rate per driving hour. Not accuracy. Not F1.
A system that fires every 10 minutes is useless regardless of accuracy.

### LOSO (Leave-One-Subject-Out) splits
No identity leakage. Each subject tested on model trained on all others.
(Rule-based baseline has no training, so LOSO measures per-subject performance.)

### Calibration (ECE)
Expected Calibration Error < 0.10 means confidence scores mean something.
Temperature scaling implemented. T needs fitting on real held-out validation data.

### Domain mismatch — stated explicitly
Public ECG datasets (MIT-BIH, PTB-XL) are 12-lead clinical recordings.
This system uses single-lead seatbelt-style ECG — a domain mismatch.
All evaluations state this. Models trained on clinical data require domain adaptation.

### Robustness tests (implemented, run via --eval)
- Motion artifact injection (ArtifactMode.motion_burst)
- Contact loss simulation (ArtifactMode.contact_loss)
- Abstain rate under each artifact mode

---

## Project Structure

```
guardian_drive_v4/
├── acquisition/
│   ├── models.py        # Typed dataclasses: SensorFrame, SQIState, FeatureBundle, etc.
│   └── simulator.py     # 10-scenario signal generator with artifact injection
├── sqi/
│   └── compute.py       # Per-channel SQI estimation
├── features/
│   └── extract.py       # Quality-gated feature extraction (ECG, EDA, Resp, IMU)
├── models/
│   ├── task_a.py        # Task A: Arrhythmia screener (rule-based baseline)
│   ├── task_b.py        # Task B: Drowsiness screener (multi-channel fusion)
│   └── task_c.py        # Task C: Crash detector + severity
├── policy/
│   ├── fusion.py        # Multi-task fusion → RiskState
│   └── state_machine.py # Safety-critical escalation state machine
├── evaluation/
│   ├── metrics.py       # FAR/hour, LOSO, ECE, PR-AUC, latency
│   └── runner.py        # Full evaluation across 10 subjects × 10 scenarios
├── tests/
│   └── test_pipeline.py # Unit + integration tests
├── main.py              # Live terminal demo
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install numpy scipy scikit-learn

# Live terminal demo (single scenario)
python main.py --scenario tachycardia
python main.py --scenario afib
python main.py --scenario drowsy
python main.py --scenario crash_severe
python main.py --list                    # all scenarios

# Full evaluation report (LOSO, FAR/hour, ECE, latency)
python main.py --eval
# or directly:
python -m evaluation.runner

# Tests
python tests/test_pipeline.py

# Live HTML Dashboard (FastAPI + WebSocket)
pip install -r requirements-server.txt

# Option A: Simulator
GD_PORT=8001 python -m server.app

# Option B: Real dataset (WESAD) — point to your extracted WESAD root
# Example (Ak's path):
# export GD_WESAD_ROOT="/Users/akilalourdes/guardian-drive/data/WESAD"
# export GD_WESAD_SUBJECT="S2"
# export GD_DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
# GD_PORT=8001 python -m server.app

# Open in browser:
# http://127.0.0.1:8001
```

---

## Phase 2 Upgrade Path

| Component | Current | Phase 2 |
|---|---|---|
| Task A model | Rule-based | LogReg / XGBoost on MIT-BIH |
| Task B model | Rule-based | LSTM/TCN on WESAD dataset |
| Temperature scaling | T=1.5 fixed | Fit T on held-out validation |
| Evaluation data | 10 simulated subjects | Real seatbelt prototype dataset |
| Domain gap | Stated, not bridged | Transfer learning + fine-tune |
| Hardware | Pi 5 target | Latency benchmark + quantization |
| Observability | Terminal logs | Structured JSON + replay tool |

---

## Data Collection Protocol (Phase 2)

To bridge the domain gap, a real seatbelt-style dataset is required:

**Rig:** ECG AFE electrodes on shoulder strap, EDA on strap contact area,
resp strain sensor, rigid IMU, belt tension FSR, temperature.

**Protocol (safe, doable without clinical setting):**
- 20–40 sessions × 5–10 participants × 20–40 min each
- Segments: resting baseline → normal driving/simulator → mild cognitive stress → late-session fatigue proxy → intentional artifact (belt shift, electrode lift)
- Labels: self-reported stress (1–10) every N min, reaction-time app, belt worn state, artifact events timestamped

---

## Software Engineering Checklist

- [x] Typed dataclasses throughout (`SensorFrame`, `SQIState`, `FeatureBundle`, etc.)
- [x] Abstain behavior at every model — SQI-gated, not just thresholded
- [x] Policy as a state machine with persistence, cooldown, corroboration
- [x] Every action carries `log_reason` + `corroborated_by` (traceable)
- [x] LOSO evaluation + FAR/hour + ECE reported
- [x] Artifact injection (contact loss, motion burst) for robustness testing
- [x] JSON serialization for session replay
- [x] Unit and integration tests
- [ ] mypy + ruff (add in Phase 2)
- [ ] Hydra config management (Phase 2)
- [ ] MLflow/W&B tracking (Phase 2 when real training added)
- [ ] Pi 5 latency benchmark (Phase 2 hardware)


---

## UI (Live + Replay)

Install UI deps:
```bash
pip install -r requirements-ui.txt
```

Run:
```bash
streamlit run ui/streamlit_app.py
```

- **Live** tab runs the pipeline on simulated scenarios and writes `runs/latest.jsonl`.
- **Replay** tab loads any JSONL run and plots SQI + scores over time.


---

## Regulatory Framing

**Safe to claim:**
- "Incapacitation risk screening with abstain behavior"
- "Abnormal rhythm pattern flagging from single-lead seatbelt ECG"
- "Emergency escalation policy in simulation mode"

**Requires FDA/CE clearance:**
- Arrhythmia diagnosis
- STEMI detection
- Blood alcohol measurement

---

*Guardian Drive™ v5.0 · Safety-Critical ML Pipeline · Raspberry Pi 5 Target*



---

## Live HTML Dashboard + Real-Time API (Recommended Demo)

This version ships a **real-time FastAPI server** that streams the **actual pipeline outputs**
to the existing `GuardianDrive_Dashboard.html` (now live + voice + GPS).

### Run
```bash
pip install -r requirements.txt
pip install -r requirements-server.txt
python -m server.app
```

Open: `http://127.0.0.1:8000`

### Dashboard controls
- **Scenario buttons** drive the backend simulator (`/api/scenario`)
- **📍 GPS toggle** uses browser geolocation and posts to `/api/gps`
- **🎙 Voice toggle** uses the browser Web Speech API for alerts
- **📣 Notify Contact** calls `/api/notify_contact` (console by default; Twilio optional)
- **🗺️ Open Route** opens Google Maps to the nearest ER (advisory)

---

## Real Data (v0.1) — What makes this “not a toy”

You cannot “claim real” without **your own recordings**.

Start minimal (ECG-only is enough to prove the ingestion + replay path):

```bash
pip install -r requirements-integrations.txt
python tools/record_serial_ecg.py --port /dev/tty.usbserial-XXXX --baud 115200 --seconds 600 --out data/raw/ecg_session.csv

python tools/build_windows_from_ecg_csv.py --csv data/raw/ecg_session.csv --fs 250 --window 30 --step 5 --out runs/ecg_replay.jsonl
```

Then replay the JSONL through the pipeline (same UI/server).

See: `docs/real_world.md` for a portfolio-safe “real-world” plan.

---

## Non-negotiable honesty (recruiter-proof)

This repository is **integration-ready** and includes **demo-safe stubs**.

- ✅ You can claim: safety-critical pipeline, SQI+abstain, state machine, streaming UI, replay logs, emergency contact notification, GPS+ER advisory.
- ❌ Do NOT claim: real vehicle autopilot control or direct PSAP/911 integration.
  Those require OEM + certification + legal approvals. Claim simulation only.


### Run dashboard on real/replayed SensorFrame windows

```bash
export GD_REPLAY_JSONL=runs/ecg_replay.jsonl
python -m server.app
```