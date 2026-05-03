# Guardian Drive -- Custom CUDA Kernels

**Built by Akilan Manivannan & Akila Lourdes Miriyala Francis**
Long Island University, Brooklyn NY

GitHub: https://github.com/AkilanManivannanak/guardian-drive
GitHub: https://github.com/AKilalours/guardian-drive

---

## Overview

Guardian Drive implements 4 hand-written CUDA kernels that accelerate
the physiological signal processing pipeline on NVIDIA GPU.
This mirrors the approach used in Tesla FSD where custom CUDA replaces
generic PyTorch and NumPy operations for latency-critical inference paths.

All kernels verified on Tesla T4 GPU, CUDA 12.8.
All speedups measured with 500-run median timing.

---

## Benchmark Results Summary

| Kernel | Task | Speedup | CUDA Median | Baseline Median | Device |
|--------|------|---------|-------------|-----------------|--------|
| hrv_features_cuda | HRV feature extraction | **61.7x** | 0.0644ms | NumPy 3.9691ms | Tesla T4 |
| sqi_cuda | Signal Quality Index | **73.4x** | 0.0458ms | Python 3.3620ms | Tesla T4 |
| ear_cuda | Eye Aspect Ratio | **319x** | 0.0307ms | NumPy 9.7977ms | Tesla T4 |
| bev_project_cuda | BEV projection | measured | -- | PyTorch einsum | Tesla T4 |
| TensorRT FP32 | TCN inference | **7.52x** | 0.157ms | PyTorch 1.181ms | Tesla T4 |
| TensorRT FP16 | TCN inference | **6.45x** | 0.183ms | PyTorch 1.181ms | Tesla T4 |

---

## Kernel 1: HRV Feature Extraction (hrv_features_cuda)

**File**: `integrations/cuda/hrv.cu`
**Replaces**: numpy HRV loop in `server/app.py` Layer 3 feature extraction

Computes 5 HRV features from raw RR intervals in a single GPU pass using
shared memory tree reduction for parallel summation across threads.

### Features computed in one kernel call
- RMSSD -- root mean square successive differences
- SDNN  -- standard deviation of NN intervals
- pNN50 -- percentage of successive differences greater than 50ms
- Mean RR interval
- Range RR interval

### Results (Tesla T4, CUDA 12.8)

| Config | CUDA | NumPy | Speedup |
|--------|------|-------|---------|
| B=64, N=300 RR intervals | 0.0644ms | 3.9691ms | **61.7x** |
| p95 | 0.1109ms | 4.4161ms | -- |
| Correctness | max_diff = 0.0000 ms vs NumPy | -- | -- |

### Why this matters
Tesla's physiological monitoring patents specifically reference real-time
HRV computation. A fused CUDA kernel computing all 5 HRV features in
one pass eliminates 4 separate memory round trips vs sequential NumPy.

---

## Kernel 2: SQI Computation (sqi_cuda)

**File**: `integrations/cuda/sqi.cu`
**Replaces**: Python variance loop in Layer 2 SQI gating (server/app.py)

Computes Signal Quality Index across all 4 physiological channels
(ECG, EDA, Temperature, Respiration) simultaneously using a 2D CUDA grid
(batch x channel) with shared memory reduction per channel.

### Results (Tesla T4, CUDA 12.8)

| Config | CUDA | Python | Speedup |
|--------|------|--------|---------|
| B=32, C=4, T=4200 | 0.0458ms | 3.3620ms | **73.4x** |

### Pipeline integration
SQI is computed at Layer 2 before any inference -- it gates all downstream
computation. Accelerating it to 0.046ms ensures zero latency overhead
on the SQI abstention decision.

---

## Kernel 3: EAR Computation (ear_cuda)

**File**: `integrations/cuda/ear.cu`
**Replaces**: numpy landmark distance computation in `vision_webcam.py`

Computes Eye Aspect Ratio from 6 MediaPipe FaceMesh landmarks per frame,
processing 1000 camera frames in parallel.
Each thread handles one frame independently -- embarrassingly parallel.

Formula computed per thread:
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

### Results (Tesla T4, CUDA 12.8)

| Config | CUDA | NumPy | Speedup |
|--------|------|-------|---------|
| B=1000 frames | 0.0307ms | 9.7977ms | **319x** |
| Correctness | max_diff = 0.000001 | -- | -- |

### Why 319x
EAR computation is embarrassingly parallel -- no dependencies between
frames. NumPy processes frames sequentially in Python.
The CUDA kernel processes all 1000 frames in a single kernel launch.

---

## Kernel 4: BEV Projection (bev_project_cuda)

**File**: `acquisition/cuda/bev.cu`
**Replaces**: torch.einsum in `acquisition/nuscenes_bev.py`

Projects 18,538 real nuScenes 3D world annotations into a 200x200
BEV occupancy grid. Each CUDA thread handles one annotation point.
Uses atomicAdd for thread-safe grid cell accumulation.

### Architecture
```
Input:  world_points [N, 3]  -- N = 18,538 nuScenes annotations
        rotation     [3, 3]  -- ego rotation matrix
        translation  [3]     -- ego translation vector
Output: bev_grid     [H, W]  -- 200x200 occupancy grid

Per thread:
  1. Translate: p_centered = p_world - t_ego
  2. Rotate:    p_ego = R^T @ p_centered (using R stored column-major)
  3. Map:       grid_x = (p_ego_x - x_min) / resolution
  4. Accumulate: atomicAdd(&grid[gy][gx], 1.0f)
```

FP16 variant (`bev_project_kernel_fp16`) reduces memory bandwidth
for large annotation batches.

---

## TensorRT Deployment

**Model**: WESAD TCN (AUC 0.9738, window-level split)
**Hardware**: Tesla T4
**TensorRT version**: 10.16.1.11
**CUDA**: 12.8

| Precision | Median | p95 | Throughput | vs PyTorch |
|-----------|--------|-----|-----------|------------|
| PyTorch FP32 | 1.181ms | -- | 847 seq/s | baseline |
| TensorRT FP32 | 0.157ms | 0.183ms | 6,357 seq/s | **7.52x** |
| TensorRT FP16 | 0.183ms | 0.210ms | 5,456 seq/s | **6.45x** |

TensorRT conversion path: PyTorch (.pt) -> ONNX (opset 11) -> TensorRT engine.
FP16 kernel fusion: Conv1d + BN + ReLU fused into single TensorRT layer.
Dynamic batch: min=1, opt=8, max=64.

---

## Full Pipeline Integration

Every kernel connects to an existing Guardian Drive module:

```
Raw sensors (ECG 700Hz, EDA 700Hz, Camera 30fps, IMU, GPS)
        |
        v
[CUDA] sqi_cuda -- Layer 2 SQI gating
       73.4x vs Python | 0.046ms for B=32 windows x 4 channels x 4200 samples
       Abstains if SQI < 0.30, preventing false escalation
        |
        v
[CUDA] hrv_features_cuda -- Layer 3 feature extraction
       61.7x vs NumPy | 0.064ms for B=64 windows x 300 RR intervals
       Feeds RMSSD, SDNN, pNN50 into Task B and Task D heuristic
[CUDA] ear_cuda -- Layer 3 camera features
       319x vs NumPy | 0.031ms for 1000 frames
       Feeds EAR -> PERCLOS -> impairment_classifier.py
        |
        v
[TensorRT FP32] TCN inference -- Layer 4a Task B
       7.52x vs PyTorch | 0.157ms | 6,357 seq/s
       Low-arousal classification (AUC 0.9738 window-level)
        |
        v
[CUDA] bev_project_cuda -- Layer 4c AV context
       18,538 nuScenes annotations projected to 200x200 BEV grid
       Object count feeds C_traffic in risk modulation Eq. 7
        |
        v
Fusion Engine -- all inputs arrive as GPU tensors, no CPU transfer
r_total = 0.40*SQI*TaskB + 0.20*TaskC + 0.10*TaskD
        |
        v
Safety State Machine -- NOMINAL / ADVISORY / CAUTION / PULLOVER / ESCALATE
        |
        v
[GPT-4o]  LLM alert explanation (live, ~500 tokens)
[OSM GIS] Hospital / cafe / motel routing by impairment type
[TTS]     macOS voice alert < 500ms
[Haptic]  Seat vibration wake-up sequence
[SMS]     Emergency contact notification via Twilio
```

---

## DDP Multi-GPU Training

**Framework**: PyTorch DistributedDataParallel
**Backend**: NCCL (NVIDIA Collective Communications Library)
**Hardware**: 2x Tesla T4 GPU

```
torchrun --nproc_per_node=2 ddp_train.py
```

| Config | Value |
|--------|-------|
| Per-GPU batch | 64 |
| Total batch (2 GPUs) | 128 |
| Backend | NCCL |
| Best AUC | 0.9488 |
| Epochs | 15 |

Key DDP components used:
- `DistributedSampler` -- each GPU sees non-overlapping data subset
- `DistributedDataParallel` -- gradient all-reduce via NCCL after each backward
- `sampler.set_epoch(ep)` -- ensures different shuffling per epoch
- `model.module.state_dict()` -- extract weights from DDP wrapper for saving

---

## Diffusion Trajectory Model

**Architecture**: Attention-based DDPM (Denoising Diffusion Probabilistic Model)
**Hardware**: Tesla T4
**Data**: Real nuScenes mini ego poses (31,206 poses, 10,397 trajectory windows)

| Metric | Value |
|--------|-------|
| Parameters | 217,218 |
| T (diffusion steps) | 1,000 |
| Sampling | DDIM-style 100 steps |
| Best training loss | 0.0019 |
| ADE | 3.30m |
| FDE | 3.36m |
| Data | Real nuScenes ego poses |

The diffusion model learns the distribution of future trajectories rather
than a single point prediction -- matching the approach used in
MotionDiffuser (Jiang et al., CVPR 2023).

---

## Self-Supervised Pretraining (SSL)

**Architecture**: SimCLR contrastive learning on WESAD physiological sequences
**File**: `learned/ssl_pretrain.py`

No labels used during pretraining. Two augmented views of the same
physiological window are pulled together in representation space;
different windows are pushed apart via NT-Xent loss.

Augmentations:
- Gaussian noise (std=0.05)
- Random amplitude scaling (0.8-1.2x per channel)
- Random time shift (up to 50 samples)
- Random channel dropout (p=0.1)

This mirrors Tesla's use of self-supervised learning on unlabeled dashcam
footage -- the same PyTorch encoder architecture is used for both
SSL pretraining and supervised Task B fine-tuning.

---

## LibTorch C++ Inference

**File**: `cpp_inference/guardian_inference.cpp`
**Build**: CMake + LibTorch arm64 macOS
**Format**: TorchScript (.pt) -- same format loaded by LibTorch C++ runtime

| Batch | Median | p95 | Throughput |
|-------|--------|-----|-----------|
| 1 | 1.99ms | 2.41ms | 500 seq/s |
| 8 | 12.59ms | 19.68ms | 636 seq/s |
| 32 | 249.55ms | 277.09ms | 128 seq/s |

Build command:
```bash
cmake .. -DCMAKE_PREFIX_PATH=~/Downloads/libtorch -DCMAKE_BUILD_TYPE=Release
make -j4
./guardian_inference wesad_tcn_scripted.pt
```

---

## Why This Matters for Tesla

Tesla FSD v12 replaced Python preprocessing with custom CUDA kernels
for real-time perception on HW4 silicon. Guardian Drive demonstrates
the same architectural pattern across the full driver monitoring stack:

1. **Hand-written kernels below PyTorch** -- bev_project, hrv_features,
   sqi, ear all bypass the PyTorch dispatch overhead entirely
2. **Fused operations** -- HRV computes 5 features in one kernel pass
   vs 5 separate NumPy calls
3. **Shared memory reduction** -- HRV and SQI kernels use on-chip
   shared memory for tree reduction, minimizing global memory traffic
4. **Parallel batch processing** -- EAR processes 1000 frames in one
   kernel launch (319x speedup)
5. **TensorRT quantization** -- FP32 and FP16 engines for edge deployment,
   matching Tesla HW3/HW4 INT8 inference pipeline
6. **NCCL multi-GPU** -- DDP training scales linearly across T4 GPUs
7. **End-to-end GPU pipeline** -- all CUDA outputs stay as GPU tensors
   through Fusion Engine, zero CPU transfer in the critical path

The EAR kernel (319x) and SQI kernel (73.4x) show the largest gains
because they replace sequential Python loops with fully parallel GPU
computation -- exactly the optimization pattern used in Tesla's
vision preprocessing stack.

---

## File Index

| File | Description |
|------|-------------|
| `acquisition/cuda/bev.cu` | BEV projection kernel |
| `integrations/cuda/hrv.cu` | HRV feature extraction kernel |
| `integrations/cuda/sqi.cu` | SQI computation kernel |
| `integrations/cuda/ear.cu` | EAR landmark distance kernel |
| `cpp_inference/guardian_inference.cpp` | LibTorch C++ inference program |
| `cpp_inference/CMakeLists.txt` | CMake build configuration |
| `convert_tensorrt.py` | TensorRT conversion pipeline |
| `learned/ssl_pretrain.py` | SimCLR SSL pretraining |
| `policy/rl_agent.py` | Q-learning safety policy |
| `learned/results/tensorrt_benchmark.json` | TensorRT verified results |
| `learned/results/hrv_cuda_benchmark.json` | HRV kernel verified results |
| `learned/results/sqi_ear_cuda_benchmark.json` | SQI+EAR verified results |
| `learned/results/task_b_ddp_eval.json` | DDP training verified results |
| `learned/results/diffusion_eval_real.json` | Diffusion verified results |

---

*Not a medical device. Not clinically validated.*
*All benchmark results measured on real hardware with real data.*
*AUC 0.9738 uses window-level split with known data leakage -- see paper.*
