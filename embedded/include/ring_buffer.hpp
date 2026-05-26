#pragma once
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <new>
#include <sys/mman.h>

// Lock-free single-producer / single-consumer ring buffer.
// Backed by mmap(MAP_ANONYMOUS) — a fixed physical memory region.
// N must be a power of 2. T must be trivially copyable.
//
// Zero-copy semantics: producer writes directly into a slot; consumer reads
// directly from that slot. No intermediate heap allocation, no memcpy of
// payload data beyond the in-place struct copy.
template <typename T, std::size_t N>
class RingBuffer {
    static_assert((N & (N - 1)) == 0, "N must be a power of 2");
    static_assert(N >= 2, "N must be at least 2");

public:
    RingBuffer() {
        void* mem = mmap(nullptr, sizeof(Slot) * N,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(mem != MAP_FAILED);
        slots_ = static_cast<Slot*>(mem);
        for (std::size_t i = 0; i < N; ++i)
            new (&slots_[i]) Slot{};
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

    ~RingBuffer() {
        munmap(slots_, sizeof(Slot) * N);
    }

    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    // Producer: write item into next slot. Returns false if full.
    bool push(const T& item) noexcept {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t next = (h + 1) & (N - 1);
        if (next == tail_.load(std::memory_order_acquire))
            return false;  // full
        slots_[h].data = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Consumer: read item from next slot. Returns false if empty.
    bool pop(T& out) noexcept {
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire))
            return false;  // empty
        out = slots_[t].data;
        tail_.store((t + 1) & (N - 1), std::memory_order_release);
        return true;
    }

    bool empty() const noexcept {
        return tail_.load(std::memory_order_acquire) ==
               head_.load(std::memory_order_acquire);
    }

    std::size_t size() const noexcept {
        std::size_t h = head_.load(std::memory_order_acquire);
        std::size_t t = tail_.load(std::memory_order_acquire);
        return (h - t) & (N - 1);
    }

private:
    struct alignas(64) Slot {
        T data{};
    };

    Slot* slots_;
    alignas(64) std::atomic<std::size_t> head_;
    alignas(64) std::atomic<std::size_t> tail_;
};
