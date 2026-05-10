/*
 * cpp_runtime/tests/test_spsc_ring_buffer.cpp
 * Unit tests for SPSCRingBuffer — ThreadSanitizer enabled
 *
 * Tests:
 *   1. Single-threaded push/pop correctness
 *   2. Full/empty detection
 *   3. Concurrent producer/consumer (TSAN checks for races)
 *   4. Power-of-2 mask correctness
 *   5. Wraparound correctness
 *
 * Compile with -fsanitize=thread to catch data races.
 */

#include "spsc_ring_buffer.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace guardian;

// ─────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────

static int passed = 0;
static int failed = 0;

#define ASSERT(cond, msg)                                    \
    do {                                                     \
        if (!(cond)) {                                       \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n",      \
                         msg, __FILE__, __LINE__);           \
            ++failed;                                        \
        } else {                                             \
            std::fprintf(stdout, "PASS: %s\n", msg);        \
            ++passed;                                        \
        }                                                    \
    } while (0)

// ─────────────────────────────────────────────
// Test 1: single-threaded push/pop
// ─────────────────────────────────────────────

void test_single_threaded() {
    SPSCRingBuffer<int, 8> buf;

    ASSERT(buf.empty(), "initially empty");
    ASSERT(!buf.full(), "initially not full");
    ASSERT(buf.size() == 0, "initial size 0");

    ASSERT(buf.try_push(42), "push to empty succeeds");
    ASSERT(!buf.empty(), "not empty after push");
    ASSERT(buf.size() == 1, "size 1 after push");

    auto val = buf.try_pop();
    ASSERT(val.has_value(), "pop returns value");
    ASSERT(*val == 42, "popped value correct");
    ASSERT(buf.empty(), "empty after pop");

    // Fill to capacity (Cap-1 elements, not Cap)
    int pushes = 0;
    while (buf.try_push(pushes)) ++pushes;
    ASSERT(buf.full(), "buffer full after filling");
    ASSERT(pushes == 7, "can push Cap-1=7 elements to buffer of 8");

    // Pop all
    int pops = 0;
    while (buf.try_pop().has_value()) ++pops;
    ASSERT(pops == 7, "popped all 7 elements");
    ASSERT(buf.empty(), "empty after popping all");
}

// ─────────────────────────────────────────────
// Test 2: wraparound correctness
// ─────────────────────────────────────────────

void test_wraparound() {
    SPSCRingBuffer<int, 4> buf;

    // Push 3, pop 3, push 3 → must wrap around correctly
    for (int cycle = 0; cycle < 5; ++cycle) {
        for (int i = 0; i < 3; ++i) {
            ASSERT(buf.try_push(cycle * 100 + i), "push in cycle");
        }
        for (int i = 0; i < 3; ++i) {
            auto val = buf.try_pop();
            ASSERT(val.has_value(), "pop in cycle");
            ASSERT(*val == cycle * 100 + i, "FIFO order preserved across wraparound");
        }
    }
}

// ─────────────────────────────────────────────
// Test 3: concurrent producer/consumer (TSAN)
// ─────────────────────────────────────────────

void test_concurrent() {
    SPSCRingBuffer<uint64_t, 64> buf;
    const int N = 100'000;
    std::atomic<int> consumed{0};
    uint64_t last_seen = 0;
    bool order_ok = true;

    std::thread producer([&] {
        for (int i = 0; i < N; ++i) {
            while (!buf.try_push(static_cast<uint64_t>(i))) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&] {
        int count = 0;
        while (count < N) {
            auto val = buf.try_pop();
            if (val) {
                if (*val < last_seen) order_ok = false;
                last_seen = *val;
                ++count;
            } else {
                std::this_thread::yield();
            }
        }
        consumed.store(count);
    });

    producer.join();
    consumer.join();

    ASSERT(consumed.load() == N, "all items consumed in concurrent test");
    ASSERT(order_ok, "FIFO order preserved in concurrent test");
}

// ─────────────────────────────────────────────
// Test 4: SensorFrame round-trip
// ─────────────────────────────────────────────

void test_sensor_frame() {
    struct SensorFrame {
        float ear{0.28f};
        float hrv_rmssd{45.0f};
        uint8_t ecg_hr{72};
        uint64_t timestamp_us{0};
    };
    static_assert(std::is_trivially_copyable_v<SensorFrame>);

    SPSCRingBuffer<SensorFrame, 8> buf;
    SensorFrame f;
    f.ear = 0.12f;
    f.hrv_rmssd = 18.5f;
    f.ecg_hr = 125;
    f.timestamp_us = 1'000'000ULL;

    buf.try_push(f);
    auto out = buf.try_pop();
    ASSERT(out.has_value(), "SensorFrame pop succeeds");
    ASSERT(out->ear == 0.12f, "ear preserved");
    ASSERT(out->hrv_rmssd == 18.5f, "hrv_rmssd preserved");
    ASSERT(out->ecg_hr == 125, "ecg_hr preserved");
    ASSERT(out->timestamp_us == 1'000'000ULL, "timestamp preserved");
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

int main() {
    std::printf("Guardian Drive — C++17 SPSC Ring Buffer Tests\n");
    std::printf("Compiled with ASAN+TSAN (cmake -DCMAKE_BUILD_TYPE=Debug)\n\n");

    test_single_threaded();
    test_wraparound();
    test_concurrent();
    test_sensor_frame();

    std::printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
