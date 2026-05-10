/*
 * cpp_runtime/include/spsc_ring_buffer.hpp
 * Lock-free Single-Producer Single-Consumer Ring Buffer
 *
 * Used in Guardian Drive C++17 inference runtime to pass sensor data
 * from acquisition thread to TensorRT inference thread without locking.
 *
 * Properties:
 *   - Wait-free for producer and consumer
 *   - Cache-line aligned to prevent false sharing
 *   - Power-of-2 capacity for fast modulo via bitmask
 *   - Sequentially consistent memory ordering on indices
 *   - Zero heap allocation after construction
 *
 * Reference:
 *   Dmitry Vjukov, "Single-Producer Single-Consumer Queue" (2010)
 *   https://www.1024cores.net/home/lock-free-algorithms/queues/unbounded-spsc-queue
 *
 * Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
 * LIU Brooklyn — MS Artificial Intelligence — C++17 Runtime
 */

#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <optional>
#include <type_traits>

namespace guardian {

/**
 * @brief Lock-free SPSC ring buffer.
 *
 * @tparam T    Element type (must be trivially copyable for wait-free operation)
 * @tparam Cap  Capacity in elements (must be power of 2, >= 2)
 *
 * Usage:
 *   SPSCRingBuffer<SensorFrame, 64> buf;
 *   // Producer thread:
 *   buf.try_push(frame);
 *   // Consumer thread:
 *   if (auto frame = buf.try_pop()) { process(*frame); }
 */
template <typename T, std::size_t Cap>
class SPSCRingBuffer {
    static_assert(Cap >= 2, "Capacity must be >= 2");
    static_assert((Cap & (Cap - 1)) == 0, "Capacity must be power of 2");
    static_assert(std::is_trivially_copyable_v<T>,
                  "T must be trivially copyable for lock-free operation");

    static constexpr std::size_t MASK = Cap - 1;
    static constexpr std::size_t CACHE_LINE = 64;

    // Separate cache lines to prevent false sharing between producer/consumer
    alignas(CACHE_LINE) std::atomic<std::size_t> _write_idx{0};
    alignas(CACHE_LINE) std::atomic<std::size_t> _read_idx{0};
    alignas(CACHE_LINE) std::array<T, Cap> _data{};

public:
    SPSCRingBuffer() = default;

    // Non-copyable, non-movable (contains atomics)
    SPSCRingBuffer(const SPSCRingBuffer&) = delete;
    SPSCRingBuffer& operator=(const SPSCRingBuffer&) = delete;

    /**
     * @brief Try to push one element (producer thread only).
     * @return true if pushed, false if buffer full.
     *
     * Memory ordering: release on _write_idx so consumer sees new element.
     */
    [[nodiscard]] bool try_push(const T& item) noexcept {
        const std::size_t write = _write_idx.load(std::memory_order_relaxed);
        const std::size_t next  = (write + 1) & MASK;
        if (next == _read_idx.load(std::memory_order_acquire)) {
            return false;  // Full
        }
        _data[write] = item;
        _write_idx.store(next, std::memory_order_release);
        return true;
    }

    /**
     * @brief Try to pop one element (consumer thread only).
     * @return Element if available, std::nullopt if empty.
     *
     * Memory ordering: acquire on _write_idx so we see producer's write.
     */
    [[nodiscard]] std::optional<T> try_pop() noexcept {
        const std::size_t read  = _read_idx.load(std::memory_order_relaxed);
        const std::size_t write = _write_idx.load(std::memory_order_acquire);
        if (read == write) {
            return std::nullopt;  // Empty
        }
        T item = _data[read];
        _read_idx.store((read + 1) & MASK, std::memory_order_release);
        return item;
    }

    /**
     * @brief Number of elements currently in buffer.
     * Approximate — may be stale by the time caller reads it.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        const std::size_t w = _write_idx.load(std::memory_order_acquire);
        const std::size_t r = _read_idx.load(std::memory_order_acquire);
        return (w - r) & MASK;
    }

    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] bool full()  const noexcept { return size() == Cap - 1; }
    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return Cap; }
};

} // namespace guardian
