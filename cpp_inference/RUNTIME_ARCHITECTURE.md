# Guardian Drive -- C++17 Real-Time Runtime Architecture

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis

---

## Current Status

| Component | Status | Evidence |
|-----------|--------|---------|
| LibTorch C++ inference | **DONE** | 1.99ms batch=1, Apple M4 |
| TensorRT FP32/FP16 engines | **DONE** | 0.157ms / 0.183ms, Tesla T4 |
| CUDA kernels (.so compiled) | **DONE** | HRV 61.7x, SQI 73.4x, EAR 319x |
| CUDA kernels callable from C++ | **DONE** | .so linkable via dlopen |
| Full C++17 streaming runtime | **ARCHITECTURE** | See below |

---

## Existing C++ Binary

```bash
cd cpp_inference
cmake .. \
    -DCMAKE_PREFIX_PATH=~/Downloads/libtorch \
    -DCMAKE_BUILD_TYPE=Release
make -j4
./guardian_inference wesad_tcn_scripted.pt

# Output:
# Batch=1:  median=1.99ms  p95=2.41ms  throughput=500 seq/s
# Batch=8:  median=12.59ms p95=19.68ms throughput=636 seq/s
# Batch=32: median=249.55ms           throughput=128 seq/s
```

---

## Target C++17 Runtime Architecture

```
guardian_runtime/
├── main.cpp                     # Main loop, signal handling, CLI
├── camera_reader.cpp            # V4L2 / OpenCV webcam frame capture
├── physiological_ring_buffer.cpp# Lock-free SPSC circular buffer
├── cuda_feature_engine.cu       # HRV + SQI + EAR CUDA kernels
├── tensorrt_inference.cpp       # TensorRT engine wrapper
├── safety_state_machine.cpp     # Deterministic 5-state Mealy FSM
├── telemetry_logger.cpp         # JSONL / protobuf append-only logging
├── cpu_fallback.cpp             # ONNX Runtime CPU path
└── CMakeLists.txt
```

---

## Target Real-Time Loop

```cpp
// main.cpp -- target production loop (C++17)
int main() {
    // Initialize
    CameraReader      cam(640, 480, 30);       // 30fps webcam
    RingBuffer<Frame> phys_buf(4200);          // 6s physiological window
    CUDAFeatureEngine cuda_feats;              // HRV/SQI/EAR kernels
    TRTInference      trt_model("tcn.engine"); // TensorRT FP32
    SafetyFSM         fsm;                     // 5-state Mealy machine
    TelemetryLogger   logger("data/telemetry.jsonl");

    while (running) {
        auto t0 = Clock::now();

        // Layer 2: SQI gating (CUDA 0.046ms)
        auto window = phys_buf.get_window();
        auto sqi    = cuda_feats.compute_sqi(window);

        if (sqi.total < 0.30) {
            fsm.abstain();            // never escalate on noise
            continue;
        }

        // Layer 3: Feature extraction (CUDA 0.064ms + 0.031ms)
        auto hrv = cuda_feats.compute_hrv(window.ecg_channel());
        auto ear = cuda_feats.compute_ear(cam.get_landmarks());

        // Layer 4: TCN inference (TensorRT 0.157ms)
        auto prob = trt_model.infer(window.normalized());

        // Layer 5: State machine (deterministic <0.01ms)
        auto state = fsm.update(sqi, hrv, ear, prob);

        // Layer 6: Outputs
        if (state != NOMINAL) alert_system.fire(state);

        // Latency logging
        auto dt_ms = duration_ms(Clock::now() - t0);
        logger.log({state, prob, sqi, dt_ms});
    }
}
```

---

## Target Latency Budget

| Stage | Target | Current (Python) | Current (C++/CUDA) |
|-------|--------|-----------------|-------------------|
| SQI CUDA | < 0.1ms | 0.046ms (Python->CUDA) | 0.046ms |
| HRV CUDA | < 0.1ms | 0.064ms | 0.064ms |
| EAR CUDA | < 0.1ms | 0.031ms | 0.031ms |
| TensorRT inference | < 0.2ms | 0.157ms | 0.157ms |
| State machine | < 0.01ms | ~1ms (Python) | <0.01ms (C++) |
| Ring buffer ops | < 0.01ms | ~5ms (Python copy) | <0.01ms (lock-free) |
| Telemetry logging | async | ~2ms (blocking) | async thread |
| **End-to-end** | **< 5ms** | **~45ms (Python)** | **~0.5ms (target)** |

---

## CPU Fallback Behavior

| Failure Condition | Behavior | Implementation |
|------------------|---------|----------------|
| CUDA unavailable | NumPy/ONNX CPU path | `cpu_fallback.cpp` |
| TensorRT engine missing | TorchScript CPU | `libtorch_fallback.cpp` |
| Camera dropout | Physiology-only degraded mode | `camera_reader.cpp` |
| ECG SQI < 0.30 | Abstain physiology | `safety_state_machine.cpp` |
| High latency (>25ms) | Skip non-critical outputs | Watchdog timer |
| Model exception | Hold previous safe state | Try-catch in FSM |
| IMU unavailable | Visual-only SLAM | `visual_inertial_odometry.cpp` |

---

## Lock-Free Ring Buffer Design

```cpp
// physiological_ring_buffer.cpp
template<typename T>
class SPSCRingBuffer {
    // Single-producer single-consumer
    // Producer: sensor acquisition thread
    // Consumer: inference thread
    // No mutex -- cache-line aligned atomic head/tail
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    std::vector<T> buffer_;
public:
    void push(T sample) noexcept;    // sensor thread
    bool pop(T& sample) noexcept;    // inference thread
    Span<T> get_window(size_t n);    // zero-copy window view
};
```

---

## TensorRT Engine Wrapper

```cpp
// tensorrt_inference.cpp
class TRTInference {
    nvinfer1::IRuntime*           runtime_;
    nvinfer1::ICudaEngine*        engine_;
    nvinfer1::IExecutionContext*  ctx_;
    void* d_input_;   // pinned device memory
    void* d_output_;  // pinned device memory
public:
    float infer(const float* window, int batch=1) {
        // Zero-copy: window already in pinned memory
        ctx_->setInputShape("physio", {batch, 4, 4200});
        ctx_->executeV2({d_input_, d_output_});
        float logit;
        cudaMemcpy(&logit, d_output_, sizeof(float),
                   cudaMemcpyDeviceToHost);
        return sigmoid(logit);
    }
};
```

---

## Safety State Machine (C++)

```cpp
// safety_state_machine.cpp -- deterministic, no heap allocation
enum class State { NOMINAL, ADVISORY, CAUTION, PULLOVER, ESCALATE, ABSTAIN };

class SafetyFSM {
    State     state_     = State::NOMINAL;
    int       hold_count_ = 0;       // hysteresis counter
    float     prev_risk_  = 0.0f;
    static constexpr int HYSTERESIS = 3;  // 3 cycles before escalation

public:
    State update(SQI sqi, HRV hrv, float tcn_prob, float av_thresh) {
        if (sqi.total < 0.30f) return abstain();

        float r = 0.40f * sqi.total * tcn_prob
                + 0.20f * imu_risk()
                + 0.10f * av_context_risk()
                + 0.30f * hrv_risk(hrv);

        State new_state = classify(r, av_thresh);

        // Hysteresis: require N consecutive cycles to escalate
        if (new_state > state_) {
            hold_count_++;
            if (hold_count_ < HYSTERESIS) return state_;
        } else {
            hold_count_ = 0;
        }

        state_ = new_state;
        return state_;
    }
private:
    State classify(float r, float thresh) const noexcept {
        if (r < thresh)        return State::NOMINAL;
        if (r < thresh+0.20f)  return State::ADVISORY;
        if (r < thresh+0.40f)  return State::CAUTION;
        if (r < thresh+0.60f)  return State::PULLOVER;
        return                        State::ESCALATE;
    }
    State abstain() {
        hold_count_ = 0;
        return State::ABSTAIN;
    }
};
```

---

## Why This Matters for Tesla

Tesla FSD v12 runs at ~36fps on HW4 silicon with < 10ms end-to-end.
Python overhead (~45ms in Guardian Drive prototype) is too slow.

The Guardian Drive prototype proves all algorithmic components:
- CUDA kernels: verified, < 0.1ms each
- TensorRT: verified, 0.157ms
- LibTorch C++: compiled, 1.99ms

Production path: move hot path to C++17 linking existing CUDA kernels.
The algorithmic correctness is proven; runtime wrapping is engineering work.

---

## Build System (CMake)

```cmake
# cpp_inference/CMakeLists.txt (extend for full runtime)
cmake_minimum_required(VERSION 3.18)
project(guardian_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
find_package(Torch REQUIRED)
find_package(CUDA  REQUIRED)
find_package(TensorRT REQUIRED)  # for full runtime

add_executable(guardian_runtime
    main.cpp
    camera_reader.cpp
    physiological_ring_buffer.cpp
    cuda_feature_engine.cu
    tensorrt_inference.cpp
    safety_state_machine.cpp
    telemetry_logger.cpp
)

target_link_libraries(guardian_runtime
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    nvinfer
    nvonnxparser
    opencv_core
    opencv_videoio
)
```
