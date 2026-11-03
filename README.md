# Guardian Drive

**Multimodal driver-safety research platform:** wearable physiological sensing, physics-based
simulation and generative modeling of noisy time-series, and safety-escalation policies in CARLA.

Akila Lourdes Miriyala Francis & Akilan Manivannan · MS Artificial Intelligence, Long Island University (Brooklyn) · 2026

[![CI](https://github.com/AKilalours/guardian-drive/actions/workflows/ci.yml/badge.svg)](https://github.com/AKilalours/guardian-drive/actions/workflows/ci.yml)

---

## Headline results

Every row links to the file that produced it. Anything that cannot be regenerated from this
repository is not in the table (see [What changed in this README](#what-changed-in-this-readme)).

| result | value | source |
|---|---|---|
| **Physics sensor-model augmentation**, AUROC under *real* 0 dB sensor noise (held out) | **0.654 → 0.818** (beats real-noise augmentation, 0.807) | [`downstream_results.json`](outputs/sensing_gen/downstream_results.json) |
| Diffusion samples + sensor model, AUROC clean / 0 dB noise | **0.838 / 0.827** (3 seeds) | [`guidance_results.json`](outputs/sensing_gen/guidance_results.json) |
| Diffusion sampler bug diagnosed and fixed (no retraining) | saturated samples **40 % → 0.5 %**; real-vs-fake discriminator **0.96 → 0.63** | same |
| Train-on-synthetic-test-on-real (diffusion) vs train-on-real | **0.829** vs 0.801 AUROC | same |
| Differentiable ECG simulator calibrated to real beats | **3,907 beats, median R² 0.951** | [`calibration_results.json`](outputs/sensing_gen/calibration_results.json) |
| Electrode–skin multiphysics FV solver, analytic verification | error **2×10⁻⁴**; wet 21.5 kΩ vs dry 443 kΩ at 10 Hz | [`multiphysics_results.json`](outputs/sensing_gen/multiphysics_results.json) |
| Inverse PINN, real ECG + real noise at 0 dB | **7.7 dB** vs 6.3 dB band-pass (filtering wins at ≥ 6 dB) | [`pinn_results.json`](outputs/sensing_gen/pinn_results.json) |
| CARLA 0.9.15, 7 town scenarios | **320 → 2** collisions vs no intervention | [`carla_real_evaluation_results.json`](outputs/carla_real_evaluation_results.json) |
| Behavior cloning vs rule-based expert (CARLA) | reward 705.5 vs 711.4; safety accuracy 98.1 % vs 98.3 % | [`carla_agent/eval_results.json`](outputs/carla_agent/eval_results.json) |
| Cardiac screen (Task A), PTB-XL 12-lead, 800-record run | AUC **0.8545** | [`ptbxl_cardiac_results.json`](outputs/ptbxl_cardiac_results.json) |
| Tests | unit + Hypothesis property + sensing_gen suites in CI | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |

---

## 1. `sensing_gen/`: simulation + generative models for real-world sensing data

Wearable physiological signals are scarce, expensive to label and dominated by motion artifacts.
`sensing_gen` builds the tools to scale training data beyond what collection alone provides, and
to *measure* whether the extra data helps:

* **Differentiable physics simulator.** The McSharry ECG dynamical model in PyTorch, with an exact
  FFT-based exponential integrator, cardio-respiratory coupling, and calibration to thousands of
  real PTB-XL beats by gradient descent *through the simulator*.
* **Physics-based sensor model.** Electrode–skin interface in a moving vehicle: seat/torso
  mechanics → skin-stretch potential, impedance-modulated gain, impedance-mismatch mains pickup,
  EMG bursts, contact loss and ADC effects. Batched, used as on-the-fly augmentation.
* **Inverse PINN.** Recovers the clean ECG *and* the ODE parameters from noisy recordings, with
  autograd ODE residuals at stochastic collocation points. Segments are solved in parallel.
* **Conditional 1-D diffusion model.** A U-Net with FiLM conditioning on diagnosis and heart rate,
  DDPM training, DDIM sampling and classifier-free guidance, plus simulation pre-training for the
  low-data regime.
* **Multiphysics electrode model.** Contact mechanics coupled to an electro-quasi-static
  finite-volume solver for layered tissue, verified against an analytic solution. It predicts
  wet vs dry electrode impedance and motion sensitivity.
* **Evaluation.** Fréchet / MMD / spectral / heart-rate fidelity, discriminative score,
  train-on-synthetic-test-on-real, and downstream AUROC under *real* recorded sensor noise
  (MIT-BIH NSTDB, held-out half).

Full method, protocol, result tables, negative results and limitations:
**[sensing_gen/README.md](sensing_gen/README.md)** · technical report: **[docs/sensing_gen_report.pdf](docs/sensing_gen_report.pdf)**.

```bash
bash scripts/run_sensing_gen.sh        # download data -> tests -> smoke run -> full run
```

---

## 2. Real-time driver-monitoring pipeline

```
SensorFrame (ECG · EDA · respiration · IMU · seat belt · webcam)
   │
   ├─ sqi/          signal-quality index per window → abstain when contact / motion is bad
   ├─ features/     R-peak detection, HRV, respiration, EAR / PERCLOS
   ├─ models/       Task A arrhythmia screen · Task B drowsiness · Task C crash · Task D stroke (FAST) screen
   ├─ policy/       inverse-variance fusion → conservative safety state machine
   │                (NOMINAL → ADVISORY → CAUTION → PULL_OVER → ESCALATE, persistence + hysteresis)
   └─ integrations/ OSM / OSRM routing to hospitals and rest stops, Twilio / Discord notification, voice
```

```bash
python3 backend_server.py                                              # WebSocket backend
python main.py --scenario drowsy --gps mock --no-webcam --duration 300 # pipeline
open dashboard/guardian_dreamview_fresh.html                           # dashboard
```

Scenarios: `normal` · `afib` · `tachycardia` · `bradycardia` · `drowsy` · `fatigued` · `stressed` ·
`impaired` · `crash_mild` · `crash_severe` · `artifact`. The scenario generator
(`acquisition/simulator.py`) is a deterministic test fixture, not a physiology model. The
physiology model lives in `sensing_gen/physics/`.

Real-data hooks: `acquisition/sources/wesad_frames.py` streams WESAD chest recordings, and
`models/train_task_a_real_ptbxl.py` trains the PTB-XL cardiac model.

---

## 3. Safety-escalation policies in CARLA

`carla_agent/` contains a Gym-style environment, a safety-critical reward, and BC / PPO policies.

* **Reward** (`carla_agent/reward/reward_fn.py`): balances missed-alarm cost against false-alarm
  ("cry-wolf") cost, with comfort (jerk) and lane-keeping penalties. Its invariants are checked with
  Hypothesis property tests (`tests/property/test_safety_properties.py`).
* **CARLA 0.9.15, 7 town scenarios** (`outputs/carla_real_evaluation_results.json`): 320 collisions
  without intervention vs 2 with the Guardian escalation machine (six states, eight detectors,
  Kalman-filtered).
* **Policy learning** (`outputs/carla_agent/eval_results.json`, 5 episodes each):

| policy | reward | safety-decision accuracy | collisions / episode |
|---|---|---|---|
| rule-based expert | 711.4 ± 45.9 | 98.3 % | 0.4 |
| behavior cloning | 705.5 ± 56.0 | 98.1 % | 0.2 |
| PPO | 155.7 ± 382.3 | 60.3 % | 0.4 |

BC matches the expert. PPO is unstable on these episodes (very high reward variance), which is
documented rather than hidden.

---

## 4. Perception and systems components

| component | status | evidence |
|---|---|---|
| PointPillars (from scratch) | trained on nuScenes-mini; `torch.compile` (inductor, max-autotune) gives **1.51×** (3.69 → 2.44 ms p50, RTX 4090) | `bev_perception/pointpillars.py`, `outputs/tensorrt_deployment_results.json` |
| Vectorized HD-map extraction | 50 real Argoverse 2 val scenes: 362 lanes, 48 crosswalks, lane topology graph | `bev_perception/hd_map_vectorizer.py`, `outputs/hd_map_vectorization_results.json` |
| Argoverse 2 motion forecasting | constant-velocity **baseline only** (sanity check, not a learned model) | `outputs/argoverse2_benchmark.json` |
| C++17 runtime | lock-free SPSC ring buffer with unit tests | `cpp_runtime/` |
| Fleet telemetry | log ingestion + DuckDB rare-event mining | `fleet_telemetry/` |

---

## Testing

```bash
python -m pytest -q tests      # unit + Hypothesis property tests + sensing_gen tests
```

GitHub Actions runs the full suite on every push (`.github/workflows/ci.yml`, CPU PyTorch).

---

## Repository layout

```
sensing_gen/        simulation, sensor model, PINN, diffusion, fidelity + downstream evaluation
acquisition/        sensor frames, WESAD / dataset loaders, scenario fixture generator
sqi/ features/      signal quality and feature extraction
models/             Task A-D screeners and training scripts (PTB-XL, WESAD)
policy/             fusion, state machine, medical guardrails
carla_agent/        CARLA environment, reward, BC / PPO policy, evaluation
bev_perception/     PointPillars, HD-map vectorizer, nuScenes streaming
cpp_runtime/        C++17 SPSC runtime
fleet_telemetry/    log ingestion and rare-event mining
integrations/       routing, notification, GPS, voice
dashboard/          browser dashboard
outputs/            result JSON files referenced above
tests/              pytest suite (unit, property, sensing_gen)
```

---

## What changed in this README

The previous README described several components whose code is not in this repository, for
example `diffusion_model.py`, `world_model.py`, `llm_finetuning.py`, `video_generation.py`,
`mpc_planner.py`, Task G/H detectors, federated learning, and TensorRT/CUDA kernel benchmarks.
It also reported some metrics that disagree with the result files (e.g. cardiac AUC 0.961 vs
`best_auc: 0.8545` in `outputs/ptbxl_cardiac_results.json`). Those claims were removed. Results
from exploratory runs that are not reproducible from this repository (several `outputs/*.json`
files with saturated metrics) are kept for transparency but are not reported. The generative
modeling work now lives, with code, tests and reproducible numbers, in `sensing_gen/`.

## License

MIT, see [LICENSE](LICENSE).
