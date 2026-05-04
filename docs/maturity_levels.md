# Guardian Drive -- Implementation Maturity Levels

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis

## Terminology Used in This Project

| Label | Meaning |
|-------|---------|
| **Implemented & Benchmarked** | Code runs, results measured, numbers verified |
| **Implemented** | Code runs, not fully benchmarked |
| **Prototyped** | Code exists, synthetic/partial validation only |
| **Designed** | Architecture documented, not yet implemented |

---

## Honest Status of Every Component

### Implemented & Benchmarked (real measured results)

| Component | Evidence | Numbers |
|-----------|---------|---------|
| WESAD TCN training | task_b_cuda_eval.json | AUC 0.9738 window-level |
| LOSO evaluation | task_b_loso_improved.json | AUC 0.769 ± 0.131 |
| DDP 2x T4 training | task_b_ddp_eval.json | AUC 0.9488 NCCL |
| TensorRT FP32 inference | tensorrt_benchmark.json | 0.157ms 7.52x |
| TensorRT FP16 inference | tensorrt_benchmark.json | 0.183ms 6.45x |
| HRV CUDA kernel | hrv_cuda_benchmark.json | 61.7x, diff=0.0000 |
| SQI CUDA kernel | sqi_ear_cuda_benchmark.json | 73.4x |
| EAR CUDA kernel | sqi_ear_cuda_benchmark.json | 319x, diff=0.000001 |
| LibTorch C++ binary | libtorch_benchmark.json | 1.99ms batch=1 |
| Real monocular SLAM | slam_real.json | 1,316 pts, 99.7% tracking |
| Real SfM (COLMAP) | sfm_real.json | 4,641 pts, 26/30 images |
| Diffusion DDPM | diffusion_eval_real.json | ADE 3.30m real nuScenes |
| Task A ECG (RF) | task_a_metrics.json | AUC 0.638 PTBDB 290 patients |
| Task A ECG (CNN) | task_a_deep_ecg.json | AUC 0.645 |
| EEG band power | eeg_analysis.json | 60ch MNE, 69 epochs |
| Camera robustness | ear_robustness.json | 7/8 conditions PASS |
| Fault injection replay | scenarios/ | 3/3 scenarios PASS |

### Implemented (code runs, partial validation)

| Component | Status | Limitation |
|-----------|--------|-----------|
| GPT-4o LLM alerts | Live API calls | Post-hoc only, not in decision path |
| OSM hospital/cafe routing | Real Overpass API | Prototype only |
| Seat haptic | macOS simulation | Not real haptic hardware |
| Emergency SMS | Twilio configured | Prototype, not production |
| SSL SimCLR pretraining | Loss converged | Not evaluated downstream |
| RL Q-learning policy | Trained | Heuristic reward, not validated |
| NeuroKit2 ECG HRV | Runs on PTBDB | LF/HF NaN at 10s window |
| Fleet aggregator | Code runs | No real fleet data |

### Prototyped (code exists, synthetic data only)

| Component | Status | What is missing |
|-----------|--------|----------------|
| Visual-Inertial Odometry | SO3 preintegration implemented | Real IMU hardware + sync |
| Camera robustness suite | Synthetic landmark perturbation | Real camera footage |
| Monocular BEV (LSS-style) | Architecture implemented | Training data + validation |
| Camera-only depth | MonocularDepthHead module | Training without lidar labels |

### Designed (architecture documented, not implemented)

| Component | Status | What is missing |
|-----------|--------|----------------|
| Nsight kernel profiling | Commands documented | Bare-metal T4 + Nsight install |
| C++17 real-time runtime | Architecture + CMake documented | Full implementation |
| TensorRT INT8 | Calibration documented | Calibration dataset + testing |
| Full EKF/factor graph VIO | Described | IMU hardware + implementation |
| GitHub Actions CI | ci.yml written | GPU runner for CUDA tests |

---

## What "Nsight profiling" means in this project

The file `benchmarks/nsight/kernel_profiles.md` contains:
- Real measured latencies (from CUDA event timing, 500 runs)
- Kernel design analysis (estimated occupancy from thread/shared-mem config)
- Nsight Compute commands ready to run on bare-metal T4

It does NOT contain:
- Real `ncu` output (requires Nsight Compute installed on T4)
- Measured occupancy (requires Nsight)
- Measured bandwidth utilization (requires Nsight)

Correct claim: "Documented kernel profiling methodology with real timing measurements. Nsight Compute commands prepared for bare-metal T4 profiling."

## What "VIO" means in this project

The file `acquisition/visual_inertial_odometry.py` contains:
- SO3 exponential map IMU preintegration (Forster 2017 style)
- EMA scale correction from IMU/visual correspondence
- Full VIOSystem class running on synthetic IMU data

It does NOT contain:
- Real IMU measurements
- Time-synchronized camera + IMU stream
- Validated metric scale against ground truth
- IMU bias estimation

Correct claim: "Prototyped VIO with SO3 IMU preintegration and scale correction. Validated visual component (1,316 map points, 99.7% tracking). Real IMU hardware needed for metric scale validation."

## What "C++17 runtime" means in this project

The file `cpp_inference/RUNTIME_ARCHITECTURE.md` contains:
- Full architecture design with lock-free ring buffer
- TensorRT wrapper code sketch
- Safety FSM C++ implementation sketch
- CMake build system

What IS implemented in C++:
- LibTorch C++ inference binary (compiled, 1.99ms measured)
- CUDA kernels as compiled .so files (linkable from C++)

What is NOT implemented:
- Full streaming runtime loop
- Camera reader integration
- Physiological ring buffer
- Complete CMake build linking all components

Correct claim: "Designed C++17 real-time runtime architecture. Implemented LibTorch C++ inference (1.99ms) and compiled CUDA kernel .so files."

---

## Resume Language Guide

### Correct
- "Implemented and benchmarked 4 custom CUDA kernels (HRV 61.7x, SQI 73.4x, EAR 319x, verified vs NumPy)"
- "Deployed TCN via TensorRT FP32 (0.157ms, 7.52x vs PyTorch, measured on Tesla T4)"
- "Prototyped VIO with SO3 IMU preintegration; visual component verified (1,316 SLAM map points)"
- "Documented C++17 runtime architecture; LibTorch C++ inference implemented (1.99ms)"
- "Prepared Nsight Compute profiling methodology; kernel timing measured via CUDA events"

### Incorrect (do not use)
- "Built C++17 TensorRT real-time runtime" -- not fully implemented
- "Nsight-profiled CUDA kernels" -- ncu not run, only CUDA event timing
- "Implemented Visual-Inertial Odometry" -- no real IMU data
- "Full production camera robustness testing" -- synthetic proxy only
