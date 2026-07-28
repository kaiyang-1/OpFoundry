#pragma once
// Host-side parallel_for on plain std::thread. Keeps the harness free of an OpenMP
// runtime, which a locally built clang (HIP_CLANG_PATH) usually does not ship.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <thread>
#include <vector>

namespace pa {

// Worker count, overridable via PA_NUM_THREADS (the OMP_NUM_THREADS equivalent).
inline unsigned num_threads() {
    if (const char* env = std::getenv("PA_NUM_THREADS")) {
        const int v = std::atoi(env);
        if (v > 0) return static_cast<unsigned>(v);
    }
    const unsigned hw = std::thread::hardware_concurrency();
    return hw ? hw : 1u;
}

// Calls body(begin, end, tid) over chunks of [0, n) claimed dynamically. Dynamic
// claiming matters for the reference kernel, where per-query cost follows the KV length.
template<class Body>
void parallel_chunks(size_t n, size_t grain, Body body) {
    if (n == 0) return;
    grain = std::max<size_t>(grain, 1);
    const size_t num_chunks = (n + grain - 1) / grain;
    const unsigned nthreads = static_cast<unsigned>(std::min<size_t>(num_threads(), num_chunks));
    if (nthreads <= 1) {
        body(size_t{0}, n, 0u);
        return;
    }

    std::atomic<size_t> next_chunk{0};
    auto worker = [&](unsigned tid) {
        for (size_t c = next_chunk.fetch_add(1, std::memory_order_relaxed); c < num_chunks;
             c = next_chunk.fetch_add(1, std::memory_order_relaxed)) {
            const size_t begin = c * grain;
            body(begin, std::min(begin + grain, n), tid);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(nthreads - 1);
    for (unsigned t = 1; t < nthreads; ++t) pool.emplace_back(worker, t);
    worker(0);  // the calling thread takes part too
    for (auto& t : pool) t.join();
}

// Enough chunks per thread to smooth out uneven iteration cost.
inline size_t default_grain(size_t n) {
    const size_t target_chunks = size_t(num_threads()) * 8;
    return std::max<size_t>(1, (n + target_chunks - 1) / target_chunks);
}

template<class Body>
void parallel_for(size_t n, Body body) {
    parallel_chunks(n, default_grain(n), [&](size_t begin, size_t end, unsigned) {
        for (size_t i = begin; i < end; ++i) body(i);
    });
}

}  // namespace pa
