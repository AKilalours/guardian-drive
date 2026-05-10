/*
 * cpp_runtime/src/guardian_runtime.cpp
 * Guardian Drive C++17 Real-Time Inference Runtime
 *
 * Production runtime for Raspberry Pi / Jetson deployment.
 * Architecture:
 *   Thread 1 (acquisition): reads 9 sensors → SPSCRingBuffer
 *   Thread 2 (inference):   reads buffer → TensorRT inference → FSM → output
 *   Thread 3 (output):      reads FSM state → GPIO haptic → WebSocket → discord
 *
 * Compile:
 *   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
 *
 * Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
 * LIU Brooklyn — MS Artificial Intelligence
 */

#include "spsc_ring_buffer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <thread>

namespace guardian {

// ─────────────────────────────────────────────
// Sensor frame — fixed-size for zero-copy ring buffer
// ─────────────────────────────────────────────

struct SensorFrame {
    // Timestamp
    uint64_t timestamp_us{0};

    // Vision (from camera + MediaPipe)
    float ear{0.28f};
    float perclos{0.08f};
    uint8_t yawn_count{0};
    float facial_asymmetry{0.02f};

    // ECG / physiology
    float hrv_rmssd{45.0f};
    uint8_t ecg_hr{72};
    float spo2{98.0f};
    float gsr_us{3.0f};

    // IMU
    float accel_x{0.0f};
    float accel_y{0.0f};
    float accel_z{9.81f};
    float g_peak{0.0f};
    float jerk_peak{0.0f};

    // Vehicle
    float steering_delta{0.0f};
    float cabin_temp_c{22.0f};
    float speed_kph{0.0f};

    // GPS
    double lat{40.6892};
    double lon{-74.0445};

    // Fault flags
    bool ecg_dropout{false};
    bool gps_loss{false};
    bool camera_occluded{false};

    // Drive time
    uint32_t drive_seconds{0};
};

static_assert(std::is_trivially_copyable_v<SensorFrame>,
              "SensorFrame must be trivially copyable");

// ─────────────────────────────────────────────
// Alert state machine
// ─────────────────────────────────────────────

enum class AlertLevel : uint8_t {
    NOMINAL  = 0,
    ADVISORY = 1,
    CAUTION  = 2,
    PULLOVER = 3,
    ESCALATE = 4,
};

const char* alert_label(AlertLevel lvl) {
    static const char* labels[] = {
        "NOMINAL", "ADVISORY", "CAUTION", "PULLOVER", "ESCALATE"
    };
    return labels[static_cast<uint8_t>(lvl)];
}

/**
 * Deterministic 5-state safety FSM with hysteresis.
 * Prevents single-frame false escalation (requires N consecutive frames).
 */
class SafetyFSM {
    static constexpr int HYSTERESIS_UP   = 3;  // frames to escalate
    static constexpr int HYSTERESIS_DOWN = 8;  // frames to de-escalate (slower)

    AlertLevel _state{AlertLevel::NOMINAL};
    AlertLevel _pending{AlertLevel::NOMINAL};
    int _consecutive{0};

public:
    AlertLevel step(AlertLevel requested) {
        if (requested == _pending) {
            ++_consecutive;
        } else {
            _pending = requested;
            _consecutive = 1;
        }

        int threshold = (requested > _state) ? HYSTERESIS_UP : HYSTERESIS_DOWN;
        if (_consecutive >= threshold) {
            _state = requested;
            _consecutive = 0;
        }
        return _state;
    }

    [[nodiscard]] AlertLevel state() const { return _state; }
};

// ─────────────────────────────────────────────
// Impairment classifier (C++17 mirror of Python version)
// ─────────────────────────────────────────────

AlertLevel classify_frame(const SensorFrame& f) {
    // Crash override (g-peak)
    if (f.g_peak >= 2.0f) return AlertLevel::ESCALATE;

    // Stroke / hypoxia
    if (f.spo2 < 92.0f && f.hrv_rmssd < 15.0f) return AlertLevel::ESCALATE;

    // Microsleep
    if (f.ear < 0.15f || f.perclos > 0.80f) return AlertLevel::ESCALATE;

    // Cardiac
    if (f.ecg_hr > 120 || (f.ecg_hr > 0 && f.ecg_hr < 45)) return AlertLevel::PULLOVER;

    // Sleepy
    if (f.perclos > 0.25f && f.yawn_count >= 3) return AlertLevel::CAUTION;

    // Fatigued
    if (f.hrv_rmssd < 20.0f && !f.ecg_dropout) return AlertLevel::ADVISORY;
    if (f.drive_seconds > 90 * 60) return AlertLevel::ADVISORY;

    // Drowsy
    if (f.perclos > 0.15f) return AlertLevel::ADVISORY;

    return AlertLevel::NOMINAL;
}

// ─────────────────────────────────────────────
// Haptic output (GPIO PWM)
// ─────────────────────────────────────────────

struct HapticCommand {
    uint8_t  gpio_pin{18};
    uint8_t  intensity_pct{0};
    uint32_t duration_ms{0};
    uint8_t  pwm_hz{0};
};

HapticCommand make_haptic(AlertLevel lvl) {
    switch (lvl) {
        case AlertLevel::NOMINAL:  return {18,   0,    0,   0};
        case AlertLevel::ADVISORY: return {18,  30,  500,  40};
        case AlertLevel::CAUTION:  return {18,  60,  800,  60};
        case AlertLevel::PULLOVER: return {18,  85, 2000,  80};
        case AlertLevel::ESCALATE: return {18, 100, 3000, 100};
        default:                   return {18,   0,    0,   0};
    }
}

// ─────────────────────────────────────────────
// Main runtime
// ─────────────────────────────────────────────

class GuardianRuntime {
    // Ring buffer: capacity 64 frames, ~2s at 30fps
    using FrameBuffer = SPSCRingBuffer<SensorFrame, 64>;

    FrameBuffer        _buffer;
    SafetyFSM          _fsm;
    std::atomic<bool>  _running{true};
    std::atomic<AlertLevel> _current_alert{AlertLevel::NOMINAL};

    // Stats
    std::atomic<uint64_t> _frames_processed{0};
    std::atomic<uint64_t> _frames_dropped{0};
    std::atomic<uint64_t> _inference_latency_us{0};

public:
    GuardianRuntime() = default;

    /**
     * Acquisition thread: reads sensors at 30Hz → pushes to ring buffer.
     * On real hardware: reads GPIO/I2C/UART sensors here.
     */
    void acquisition_loop() {
        uint32_t drive_seconds = 0;
        auto next_tick = std::chrono::steady_clock::now();
        const auto tick_interval = std::chrono::microseconds(33'333);  // 30fps

        while (_running.load(std::memory_order_relaxed)) {
            SensorFrame frame;
            frame.timestamp_us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()
                ).count()
            );
            frame.drive_seconds = ++drive_seconds;

            // TODO(hardware): replace with real sensor reads:
            // frame.hrv_rmssd  = ecg_sensor.read_hrv();
            // frame.ear        = camera.compute_ear();
            // frame.g_peak     = imu.read_g_peak();
            // frame.speed_kph  = gps.read_speed_kph();

            // Simulation placeholder
            frame.hrv_rmssd = 45.0f - 0.01f * drive_seconds;
            frame.ear       = 0.28f - 0.0001f * drive_seconds;
            frame.speed_kph = 60.0f;

            if (!_buffer.try_push(frame)) {
                ++_frames_dropped;
            }

            next_tick += tick_interval;
            std::this_thread::sleep_until(next_tick);
        }
    }

    /**
     * Inference thread: reads ring buffer → classify → FSM → haptic.
     * On real hardware: also calls TensorRT TCN inference here.
     */
    void inference_loop() {
        while (_running.load(std::memory_order_relaxed)) {
            auto maybe_frame = _buffer.try_pop();
            if (!maybe_frame) {
                std::this_thread::sleep_for(std::chrono::microseconds(500));
                continue;
            }

            const auto& frame = *maybe_frame;
            const auto t0 = std::chrono::steady_clock::now();

            // Classify impairment
            AlertLevel raw_alert = classify_frame(frame);

            // FSM with hysteresis
            AlertLevel stable_alert = _fsm.step(raw_alert);
            _current_alert.store(stable_alert, std::memory_order_release);

            // Haptic output
            HapticCommand haptic = make_haptic(stable_alert);

            const auto t1 = std::chrono::steady_clock::now();
            _inference_latency_us.store(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(),
                std::memory_order_relaxed
            );

            ++_frames_processed;

            // Log on state change
            if (stable_alert != AlertLevel::NOMINAL) {
                std::cout << "[GuardianRuntime] Alert="
                          << alert_label(stable_alert)
                          << " HRV=" << frame.hrv_rmssd
                          << " EAR=" << frame.ear
                          << " Haptic=" << static_cast<int>(haptic.intensity_pct) << "%"
                          << " latency=" << _inference_latency_us.load() << "us"
                          << "\n";
            }
        }
    }

    void start() {
        std::thread acq(&GuardianRuntime::acquisition_loop, this);
        std::thread inf(&GuardianRuntime::inference_loop, this);
        acq.detach();
        inf.detach();
    }

    void stop() { _running.store(false, std::memory_order_relaxed); }

    AlertLevel current_alert() const {
        return _current_alert.load(std::memory_order_acquire);
    }

    void print_stats() const {
        std::cout << "[GuardianRuntime] Stats:"
                  << " frames_processed=" << _frames_processed
                  << " frames_dropped=" << _frames_dropped
                  << " inference_latency_us=" << _inference_latency_us
                  << " buffer_size=" << _buffer.size()
                  << " alert=" << alert_label(current_alert())
                  << "\n";
    }
};

} // namespace guardian

// ─────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────

int main() {
    std::cout << "Guardian Drive C++17 Runtime v2.0\n"
              << "Akilan Manivannan & Akila Lourdes Miriyala Francis\n"
              << "LIU Brooklyn — MS Artificial Intelligence\n\n";

    guardian::GuardianRuntime runtime;
    runtime.start();

    // Run for 5 seconds then print stats
    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        runtime.print_stats();
    }

    runtime.stop();
    std::cout << "Runtime stopped.\n";
    return 0;
}
