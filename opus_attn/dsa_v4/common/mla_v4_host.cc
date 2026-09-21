#include <hip/hip_fp8.h>
#include <opus/hip_minimal.hpp>
#include <algorithm>
#include <random>
#include <iostream>
#include <limits>
#include <numeric>
#include <memory>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <type_traits>
#include "common/mla_v4_parallel.h"

#if defined(MLA_V4_ARCH_GFX950)
#include "gfx950/mla_v4_traits.h"

template<class Traits>
__global__ void opus_mla_v4_prefill_a16w16_16mx1_16nx4_kernel(opus_mla_v4_prefill_kargs kargs);
template<class Traits>
__global__ void opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernel(opus_mla_v4_prefill_kargs kargs);
template<class Traits>
__global__ void opus_mla_v4_prefill_a8w8_16mx1_16nx4_kernel(opus_mla_v4_prefill_fp8_kargs kargs);
template<class Traits>
__global__ void opus_mla_v4_prefill_a8w8_16mx8_32nx1_kernel(opus_mla_v4_prefill_fp8_kargs kargs);

template<int Q, int KV, int D, int NW, class DT, class DO>
inline void mla_v4_prefill_launch(opus_mla_v4_prefill_a16w16_16mx1_16nx4_traits<Q, KV, D, NW, DT, DO>,
                                  const opus_mla_v4_prefill_kargs& kargs, dim3 grid, dim3 block) {
    opus_mla_v4_prefill_a16w16_16mx1_16nx4_kernel<opus_mla_v4_prefill_a16w16_16mx1_16nx4_traits<Q, KV, D, NW, DT, DO>><<<grid, block>>>(kargs);
}
template<int Q, int KV, int D, int NW, class DT, class DO>
inline void mla_v4_prefill_launch(opus_mla_v4_prefill_a16w16_16mx8_32nx1_traits<Q, KV, D, NW, DT, DO>,
                                  const opus_mla_v4_prefill_kargs& kargs, dim3 grid, dim3 block) {
    opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernel<opus_mla_v4_prefill_a16w16_16mx8_32nx1_traits<Q, KV, D, NW, DT, DO>><<<grid, block>>>(kargs);
}
template<int Q, int KV, int NW, class NOPE, class ROPE, class DO>
inline void mla_v4_prefill_launch(opus_mla_v4_prefill_a8w8_16mx1_16nx4_traits<Q, KV, NW, NOPE, ROPE, DO>,
                                  const opus_mla_v4_prefill_fp8_kargs& kargs, dim3 grid, dim3 block) {
    opus_mla_v4_prefill_a8w8_16mx1_16nx4_kernel<opus_mla_v4_prefill_a8w8_16mx1_16nx4_traits<Q, KV, NW, NOPE, ROPE, DO>><<<grid, block>>>(kargs);
}
template<int Q, int KV, int NW, class NOPE, class ROPE, class DO>
inline void mla_v4_prefill_launch(opus_mla_v4_prefill_a8w8_16mx8_32nx1_traits<Q, KV, NW, NOPE, ROPE, DO>,
                                  const opus_mla_v4_prefill_fp8_kargs& kargs, dim3 grid, dim3 block) {
    opus_mla_v4_prefill_a8w8_16mx8_32nx1_kernel<opus_mla_v4_prefill_a8w8_16mx8_32nx1_traits<Q, KV, NW, NOPE, ROPE, DO>><<<grid, block>>>(kargs);
}

#elif defined(MLA_V4_ARCH_GFX1250)
#include "gfx1250/mla_v4_traits.h"

template<class Traits>
__global__ void opus_mla_v4_prefill_a16w16_16mx4_64nx1_kernel(opus_mla_v4_prefill_kargs kargs);
template<class Traits>
__global__ void opus_mla_v4_prefill_a16w16_16mx1_16nx4_kernel(opus_mla_v4_prefill_kargs kargs);
template<class Traits>
__global__ void opus_mla_v4_prefill_a16w16_32mx1_16nx4_kernel(opus_mla_v4_prefill_kargs kargs);
template<class Traits>
__global__ void opus_mla_v4_prefill_a8w8_16mx4_64nx1_kernel(opus_mla_v4_prefill_fp8_kargs kargs);
template<class Traits>
__global__ void opus_mla_v4_prefill_a8w8_16mx1_16nx4_kernel(opus_mla_v4_prefill_fp8_kargs kargs);
template<class Traits>
__global__ void opus_mla_v4_prefill_a8w8_32mx1_16nx4_kernel(opus_mla_v4_prefill_fp8_kargs kargs);

constexpr int mla_v4_max_cluster_y = 2;

inline int mla_v4_pick_cluster_y(int num_h_blocks) {
    for (int c = mla_v4_max_cluster_y; c > 1; c >>= 1)
        if (num_h_blocks % c == 0) return c;
    return 1;
}

template<int CY, class Kernel, class KArgs>
inline void mla_v4_prefill_launch_clustered(Kernel kernel, const KArgs& kargs, dim3 grid, dim3 block) {
    if constexpr (CY == 1) {
        kernel<<<grid, block>>>(kargs);
    } else {
        if (grid.y % CY != 0) {
            fprintf(stderr, "CLUSTER_Y=%d does not divide grid.y=%u; refusing to launch a masked gather\n",
                    CY, grid.y);
            exit(1);
        }
        hipLaunchAttribute attr{};
        attr.id = hipLaunchAttributeClusterDimension;
        attr.val.clusterDim.x = 1;
        attr.val.clusterDim.y = CY;
        attr.val.clusterDim.z = 1;

        hipLaunchConfig_t cfg{};
        cfg.gridDim          = grid;
        cfg.blockDim         = block;
        cfg.dynamicSmemBytes = 0;
        cfg.stream           = nullptr;
        cfg.attrs            = &attr;
        cfg.numAttrs         = 1;
        hipError_t e = hipLaunchKernelEx(&cfg, kernel, kargs);
        if (e != hipSuccess) {
            fprintf(stderr, "cluster launch (1,%d,1) failed: %s\n", CY, hipGetErrorString(e));
            exit(1);
        }
    }
}

template<int Q, int KV, int D, int NW, int CY, class DT, class DO>
inline void mla_v4_prefill_launch(opus_mla_v4_prefill_a16w16_16mx4_64nx1_traits<Q, KV, D, NW, CY, DT, DO>,
                                  const opus_mla_v4_prefill_kargs& kargs, dim3 grid, dim3 block) {
    using Traits = opus_mla_v4_prefill_a16w16_16mx4_64nx1_traits<Q, KV, D, NW, CY, DT, DO>;
    mla_v4_prefill_launch_clustered<CY>(opus_mla_v4_prefill_a16w16_16mx4_64nx1_kernel<Traits>, kargs, grid, block);
}
template<int Q, int KV, int D, int NW, class DT, class DO>
inline void mla_v4_prefill_launch(opus_mla_v4_prefill_a16w16_16mx1_16nx4_traits<Q, KV, D, NW, DT, DO>,
                                  const opus_mla_v4_prefill_kargs& kargs, dim3 grid, dim3 block) {
    opus_mla_v4_prefill_a16w16_16mx1_16nx4_kernel<opus_mla_v4_prefill_a16w16_16mx1_16nx4_traits<Q, KV, D, NW, DT, DO>><<<grid, block>>>(kargs);
}
template<int Q, int KV, int D, int NW, class DT, class DO>
inline void mla_v4_prefill_launch(opus_mla_v4_prefill_a16w16_32mx1_16nx4_traits<Q, KV, D, NW, DT, DO>,
                                  const opus_mla_v4_prefill_kargs& kargs, dim3 grid, dim3 block) {
    opus_mla_v4_prefill_a16w16_32mx1_16nx4_kernel<opus_mla_v4_prefill_a16w16_32mx1_16nx4_traits<Q, KV, D, NW, DT, DO>><<<grid, block>>>(kargs);
}
template<int Q, int KV, int NW, int CY, class NOPE, class ROPE, class DO>
inline void mla_v4_prefill_launch(opus_mla_v4_prefill_a8w8_16mx4_64nx1_traits<Q, KV, NW, CY, NOPE, ROPE, DO>,
                                  const opus_mla_v4_prefill_fp8_kargs& kargs, dim3 grid, dim3 block) {
    using Traits = opus_mla_v4_prefill_a8w8_16mx4_64nx1_traits<Q, KV, NW, CY, NOPE, ROPE, DO>;
    mla_v4_prefill_launch_clustered<CY>(opus_mla_v4_prefill_a8w8_16mx4_64nx1_kernel<Traits>, kargs, grid, block);
}
template<int Q, int KV, int NW, class NOPE, class ROPE, class DO>
inline void mla_v4_prefill_launch(opus_mla_v4_prefill_a8w8_16mx1_16nx4_traits<Q, KV, NW, NOPE, ROPE, DO>,
                                  const opus_mla_v4_prefill_fp8_kargs& kargs, dim3 grid, dim3 block) {
    opus_mla_v4_prefill_a8w8_16mx1_16nx4_kernel<opus_mla_v4_prefill_a8w8_16mx1_16nx4_traits<Q, KV, NW, NOPE, ROPE, DO>><<<grid, block>>>(kargs);
}
template<int Q, int KV, int NW, class NOPE, class ROPE, class DO>
inline void mla_v4_prefill_launch(opus_mla_v4_prefill_a8w8_32mx1_16nx4_traits<Q, KV, NW, NOPE, ROPE, DO>,
                                  const opus_mla_v4_prefill_fp8_kargs& kargs, dim3 grid, dim3 block) {
    opus_mla_v4_prefill_a8w8_32mx1_16nx4_kernel<opus_mla_v4_prefill_a8w8_32mx1_16nx4_traits<Q, KV, NW, NOPE, ROPE, DO>><<<grid, block>>>(kargs);
}
#else
#  error "No target arch defined. The Makefile passes MLA_V4_ARCH_<ARCH> from ARCH (e.g. ARCH=gfx950)."
#endif

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_HIP_KERNEL_LAUNCH() CHECK_HIP(hipGetLastError())

template<class T>
struct dev_buf {
    T* ptr = nullptr;

    explicit dev_buf(size_t count) {
        CHECK_HIP(hipMalloc(&ptr, std::max<size_t>(count, 1) * sizeof(T)));
    }
    dev_buf(const dev_buf&) = delete;
    dev_buf& operator=(const dev_buf&) = delete;
    ~dev_buf() { if (ptr) (void)hipFree(ptr); }

    void upload(const T* src, size_t count) {
        if (count) CHECK_HIP(hipMemcpy(ptr, src, count * sizeof(T), hipMemcpyHostToDevice));
    }
    void download(T* dst, size_t count) const {
        if (count) CHECK_HIP(hipMemcpy(dst, ptr, count * sizeof(T), hipMemcpyDeviceToHost));
    }
    void zero(size_t count) {
        if (count) CHECK_HIP(hipMemset(ptr, 0, count * sizeof(T)));
    }
    operator T*() const { return ptr; }
};

// Fixed grain, so a chunk spans the same elements at any MLA_V4_NUM_THREADS. Hashed
// rather than seed+index, which would make Q rows bit-identical to KV rows.
static constexpr size_t MLA_V4_INIT_GRAIN = 65536;
static constexpr uint64_t MLA_V4_SEED = 2026;

inline std::mt19937 mla_v4_rng(uint64_t seed, uint64_t index) {
    uint64_t x = seed + 0x9E3779B97F4A7C15ull * (index + 1);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return std::mt19937(static_cast<uint32_t>(x ^ (x >> 31)));
}

template<typename T>
void rand_vector(T* ptr, size_t size, uint64_t seed) {
    mla_v4::parallel_for((size + MLA_V4_INIT_GRAIN - 1) / MLA_V4_INIT_GRAIN, [&](size_t c) {
        const size_t begin = c * MLA_V4_INIT_GRAIN;
        const size_t end = std::min(begin + MLA_V4_INIT_GRAIN, size);
        std::mt19937 gen = mla_v4_rng(seed, c);
        std::normal_distribution<float> dis(0.0f, 1.0f);
        for (size_t i = begin; i < end; i++) {
            ptr[i] = static_cast<T>(dis(gen));
        }
    });
}

template<class Traits>
void init_fp8_dsa_split(typename Traits::D_NOPE* nope_ptr,
                        typename Traits::D_ROPE* rope_ptr, size_t rows, uint64_t seed) {
    using D_ROPE = typename Traits::D_ROPE;
    constexpr int NOPE_PADDED = Traits::D_NOPE_PADDED_SIZE;
    constexpr int NOPE        = Traits::D_NOPE_SIZE;
    constexpr int SCALE       = NOPE / 32;
    constexpr int BLOCK       = NOPE / SCALE;
    constexpr int ROPE        = Traits::D_ROPE_SIZE;
    static_assert(NOPE + SCALE <= NOPE_PADDED, "NoPE + scales exceed padded row");

    mla_v4::parallel_for((rows + MLA_V4_INIT_GRAIN - 1) / MLA_V4_INIT_GRAIN, [&](size_t c) {
        const size_t begin = c * MLA_V4_INIT_GRAIN;
        const size_t end = std::min(begin + MLA_V4_INIT_GRAIN, rows);
        std::mt19937 gen = mla_v4_rng(seed, c);
        std::normal_distribution<float> dis(0.0f, 1.0f);
        std::uniform_int_distribution<int> scale_exp_dis(0, 3);
        for (size_t r = begin; r < end; r++) {
            unsigned char* nbase = reinterpret_cast<unsigned char*>(nope_ptr) + r * NOPE_PADDED;
            auto* nope = reinterpret_cast<__hip_fp8_e4m3*>(nbase);
            unsigned char* scale = nbase + NOPE;
            for (int b = 0; b < SCALE; b++) {
                const int e = scale_exp_dis(gen);
                scale[b] = static_cast<unsigned char>(e + 127);
                const float inv_scale = std::ldexp(1.0f, -e);
                for (int i = 0; i < BLOCK; i++)
                    nope[b * BLOCK + i] = static_cast<__hip_fp8_e4m3>(dis(gen) * inv_scale);
            }
            // 0xFF is NaN as both e4m3 and E8M0: the kernel masks the pad off itself.
            for (int i = NOPE + SCALE; i < NOPE_PADDED; i++) nbase[i] = 0xFF;
            D_ROPE* rope = rope_ptr + r * ROPE;
            for (int i = 0; i < ROPE; i++) rope[i] = static_cast<D_ROPE>(dis(gen));
        }
    });
}

// Leave the pages scattered: sorting gives the gather a near-sequential read that no
// real page table provides.
void init_kv_page_table(std::vector<int>& kv_indptr,
                        std::vector<int>& kv_indices,
                        int N,
                        int pool_rows,
                        int topk,
                        bool causal,
                        uint64_t seed) {
    assert(N >= 0);
    assert(pool_rows >= 0);

    const int budget = (topk <= 0 || topk > pool_rows) ? pool_rows : topk;
    auto candidates = [&](int q) { return causal ? std::min(q + 1, pool_rows) : pool_rows; };

    kv_indptr.assign(N + 1, 0);
    for (int q = 0; q < N; ++q) {
        const int nnz = std::min(budget, candidates(q));
        const size_t end = static_cast<size_t>(kv_indptr[q]) + static_cast<size_t>(nnz);
        assert(end <= static_cast<size_t>(std::numeric_limits<int>::max()));
        kv_indptr[q + 1] = static_cast<int>(end);
    }

    kv_indices.resize(static_cast<size_t>(kv_indptr[N]));
    const size_t grain = std::max<size_t>(1, ceil_div(N, static_cast<int>(mla_v4::num_threads())));
    mla_v4::parallel_chunks(static_cast<size_t>(N), grain, [&](size_t begin, size_t end, unsigned) {
        std::vector<int> perm(static_cast<size_t>(pool_rows));
        std::iota(perm.begin(), perm.end(), 0);
        for (size_t q = begin; q < end; ++q) {
            int* row = kv_indices.data() + kv_indptr[q];
            const int nnz = kv_indptr[q + 1] - kv_indptr[q];
            const int range = candidates(static_cast<int>(q));
            std::mt19937 gen = mla_v4_rng(seed, static_cast<uint64_t>(q) + (1ull << 32));
            for (int i = 0; i < nnz; ++i) {
                const int j = i + static_cast<int>(gen() % static_cast<uint32_t>(range - i));
                std::swap(perm[i], perm[j]);
                row[i] = perm[i];
            }
            // Undo the swaps: the next row must draw from a clean identity permutation.
            for (int i = 0; i < nnz; ++i) perm[i] = i;
            for (int i = 0; i < nnz; ++i) perm[row[i]] = row[i];
        }
    });
}

inline std::vector<uint8_t> mla_v4_poison_mask(int pool_rows, const std::vector<int>& kv_indices) {
    std::vector<uint8_t> poison(static_cast<size_t>(pool_rows), 1);
    for (int idx : kv_indices) poison[static_cast<size_t>(idx)] = 0;
    return poison;
}

template<class Traits>
void poison_kv_rows(typename Traits::D_ATTN* kv, int pool_rows,
                    const std::vector<int>& kv_indices) {
    constexpr int ROW = Traits::D_TILE_SIZE;
    const auto nan = static_cast<typename Traits::D_ATTN>(std::numeric_limits<float>::quiet_NaN());
    const auto poison = mla_v4_poison_mask(pool_rows, kv_indices);
    for (size_t r = 0; r < poison.size(); ++r)
        if (poison[r]) std::fill_n(kv + r * ROW, ROW, nan);
}

template<class Traits>
void poison_kv_rows_fp8(typename Traits::D_NOPE* nope_ptr, typename Traits::D_ROPE* rope_ptr,
                        int pool_rows, const std::vector<int>& kv_indices) {
    constexpr int NOPE_PADDED = Traits::D_NOPE_PADDED_SIZE;
    constexpr int ROPE = Traits::D_ROPE_SIZE;
    const auto nan = static_cast<typename Traits::D_ROPE>(std::numeric_limits<float>::quiet_NaN());
    const auto poison = mla_v4_poison_mask(pool_rows, kv_indices);
    for (size_t r = 0; r < poison.size(); ++r) {
        if (!poison[r]) continue;
        std::memset(reinterpret_cast<unsigned char*>(nope_ptr) + r * NOPE_PADDED, 0xFF, NOPE_PADDED);
        std::fill_n(rope_ptr + r * ROPE, ROPE, nan);
    }
}

template<class Traits, class KArgs>
void benchmark_mla_v4_kernel(const KArgs& kargs, dim3 grid, dim3 block,
                             int total_kv_rows, int warmup = 100, int iterations = 50) {
    for (int i = 0; i < warmup; ++i) {
        mla_v4_prefill_launch(Traits{}, kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        mla_v4_prefill_launch(Traits{}, kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_time = 0;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const float avg_time = total_time / iterations;

    using D_ATTN = typename Traits::D_ATTN;
    using D_OUT  = typename Traits::D_OUT;
    constexpr int D_HEAD = Traits::D_HEAD_SIZE;

    const double flops = 4.0 * kargs.H * total_kv_rows * D_HEAD;
    const double tflops = flops / (avg_time * 1e-3) / 1e12;

    size_t row_bytes;
    if constexpr (std::is_same_v<KArgs, opus_mla_v4_prefill_fp8_kargs>) {
        row_bytes = (size_t)Traits::D_NOPE_PADDED_SIZE * sizeof(typename Traits::D_NOPE)
                  + (size_t)Traits::D_ROPE_SIZE * sizeof(typename Traits::D_ROPE);
    } else {
        row_bytes = (size_t)Traits::D_TILE_SIZE * sizeof(D_ATTN);
    }
    const size_t q_bytes  = (size_t)kargs.N * kargs.H * row_bytes;
    const size_t o_bytes  = (size_t)kargs.N * kargs.H * D_HEAD * sizeof(D_OUT);
    const size_t kv_bytes = (size_t)total_kv_rows * row_bytes;
    const double tbps = double(q_bytes + o_bytes + kv_bytes) / (avg_time * 1e-3) / 1e12;

    printf("MLA-v4 Prefill Kernel Performance: avg_time=%.3f ms, %.2f TFlops, %.2f TB/s\n",
           avg_time, tflops, tbps);
}

template<typename DType>
bool validate_mla_v4_results(const DType* ref, const DType* gpu,
                             int N, int H, int D,
                             float rtol = 1e-2f, float atol = 1e-2f,
                             float tol_err_ratio = 0.05f) {
    const size_t total_elements = (size_t)N * H * D;
    constexpr size_t print_limit = 10;

    size_t total_errors = 0, printed = 0;
    bool any_nan = false;
    float max_abs_delta = 0.0f, ref_absmax = 0.0f;
    double sq_diff_sum = 0.0, ref_sq_sum = 0.0;

    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            const size_t offset = ((size_t)n * H + h) * D;
            for (int d = 0; d < D; d++) {
                const float ref_val = static_cast<float>(ref[offset + d]);
                const float gpu_val = static_cast<float>(gpu[offset + d]);
                const float delta   = std::abs(gpu_val - ref_val);

                ref_absmax  = std::max(ref_absmax, std::abs(ref_val));
                sq_diff_sum += double(delta) * double(delta);
                ref_sq_sum  += double(ref_val) * double(ref_val);

                // Ref too: a NaN delta compares false and would pass silently.
                const bool nan_inf = std::isnan(gpu_val) || std::isinf(gpu_val)
                                  || std::isnan(ref_val) || std::isinf(ref_val);
                any_nan |= nan_inf;
                if (nan_inf || delta > atol + rtol * std::abs(ref_val)) {
                    total_errors++;
                    max_abs_delta = std::max(max_abs_delta, delta);
                    if (printed++ < print_limit)
                        printf("  mismatch [n=%d,h=%d,d=%d] ref=%.6f gpu=%.6f delta=%.6f\n",
                               n, h, d, ref_val, gpu_val, delta);
                }
            }
        }
    }

    const double err_ratio    = double(total_errors) / double(total_elements);
    const double nrms         = std::sqrt(sq_diff_sum / std::max(ref_sq_sum, 1e-12));
    const bool   catastrophic = any_nan || max_abs_delta > 0.5f * ref_absmax;
    const bool   all_valid    = !catastrophic && err_ratio <= tol_err_ratio;

    printf("  rtol=%.0e atol=%.0e | max_abs_delta=%.6f nrms=%.3e | mismatch %zu/%zu (%.2f%%)\n",
           rtol, atol, max_abs_delta, nrms, total_errors, total_elements, 100.0 * err_ratio);
    if (all_valid)
        printf("✓ Validation passed (checked %zu elements)\n", total_elements);
    else if (catastrophic)
        printf("✗ Validation failed (catastrophic: %s, max_abs_delta=%.6f)\n",
               any_nan ? "NaN/Inf" : "delta > 0.5*max|ref|", max_abs_delta);
    else
        printf("✗ Validation failed (mismatch ratio %.2f%% > %.2f%%)\n",
               100.0 * err_ratio, 100.0 * tol_err_ratio);

    return all_valid;
}

template<class Traits>
inline void decode_dsa_row_bf16(const typename Traits::D_ATTN* row, float* out) {
    constexpr int D_HEAD = Traits::D_HEAD_SIZE;
    for (int d = 0; d < D_HEAD; d++) out[d] = static_cast<float>(row[d]);
}

template<class Traits>
inline void decode_dsa_row_fp8(const typename Traits::D_NOPE* nrow,
                               const typename Traits::D_ROPE* rrow, float* out) {
    constexpr int NOPE = Traits::D_NOPE_SIZE;
    constexpr int ROPE = Traits::D_ROPE_SIZE;
    const auto* base = reinterpret_cast<const unsigned char*>(nrow);
    const auto* nope = reinterpret_cast<const __hip_fp8_e4m3*>(base);
    const unsigned char* scale = base + NOPE;
    for (int d = 0; d < NOPE; d++)
        out[d] = static_cast<float>(nope[d]) * std::ldexp(1.0f, int(scale[d / 32]) - 127);
    for (int j = 0; j < ROPE; j++)
        out[NOPE + j] = static_cast<float>(rrow[j]);
}

template<class Traits>
inline void mla_v4_attention_compute(const float* q_dense, const float* kv_dense, int num_rows,
                                     float sink, typename Traits::D_OUT* o_row) {
    using O_t = typename Traits::D_OUT;
    constexpr int D_HEAD = Traits::D_HEAD_SIZE;
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D_HEAD));

    std::vector<float> scores(num_rows);
    for (int p = 0; p < num_rows; p++) {
        const float* k = kv_dense + (size_t)p * D_HEAD;
        float dot = 0.0f;
        for (int d = 0; d < D_HEAD; d++) dot += q_dense[d] * k[d];
        scores[p] = dot * softmax_scale;
    }
    float max_score = std::max(*std::max_element(scores.begin(), scores.end()), sink);
    float sum_exp = 0.0f;
    for (int p = 0; p < num_rows; p++) { scores[p] = std::exp(scores[p] - max_score); sum_exp += scores[p]; }
    sum_exp += std::exp(sink - max_score);
    for (int p = 0; p < num_rows; p++)
        scores[p] = static_cast<float>(static_cast<bf16_t>(scores[p] / sum_exp));
    for (int d = 0; d < D_HEAD; d++) {
        float acc = 0.0f;
        for (int p = 0; p < num_rows; p++) acc += scores[p] * kv_dense[(size_t)p * D_HEAD + d];
        o_row[d] = static_cast<O_t>(acc);
    }
}

template<class Traits>
void mla_v4_attention_ref(
    const typename Traits::D_ATTN* Q,
    const typename Traits::D_ATTN* UnifiedKV,
    const typename Traits::D_ATTN* KV,
    const float* AttnSink,
    typename Traits::D_OUT* O,
    const int* kv_indptr_prefix,
    const int* kv_indices_prefix,
    const int* kv_indptr_extend,
    const int* kv_indices_extend,
    int N, int H)
{
    using O_t = typename Traits::D_OUT;
    constexpr int D_HEAD = Traits::D_HEAD_SIZE;
    constexpr int ROW    = Traits::D_TILE_SIZE;
    const int stride_qo_n = H * ROW;
    const int stride_qo_h = ROW;
    const int stride_kv_page = ROW;
    const int o_stride_n = H * D_HEAD;
    const int o_stride_h = D_HEAD;

    mla_v4::parallel_for((size_t)H * N, [&](size_t idx) {
        const int h = static_cast<int>(idx / N);
        const int i = static_cast<int>(idx % N);
        const int prefix_begin = kv_indptr_prefix[i];
        const int extend_begin = kv_indptr_extend[i];
        const int num_prefix = kv_indptr_prefix[i + 1] - prefix_begin;
        const int num_extend = kv_indptr_extend[i + 1] - extend_begin;
        const int num_rows   = num_prefix + num_extend;

        O_t* o_row = O + (size_t)i * o_stride_n + h * o_stride_h;
        if (num_rows <= 0) {
            for (int d = 0; d < D_HEAD; d++) o_row[d] = static_cast<O_t>(0.0f);
            return;
        }

        std::vector<float> q_dense(D_HEAD);
        decode_dsa_row_bf16<Traits>(Q + (size_t)i * stride_qo_n + h * stride_qo_h, q_dense.data());

        std::vector<float> kv_dense((size_t)num_rows * D_HEAD);
        for (int p = 0; p < num_prefix; p++)
            decode_dsa_row_bf16<Traits>(UnifiedKV + (size_t)kv_indices_prefix[prefix_begin + p] * stride_kv_page,
                                          kv_dense.data() + (size_t)p * D_HEAD);
        for (int p = 0; p < num_extend; p++)
            decode_dsa_row_bf16<Traits>(KV + (size_t)kv_indices_extend[extend_begin + p] * stride_kv_page,
                                          kv_dense.data() + (size_t)(num_prefix + p) * D_HEAD);

        mla_v4_attention_compute<Traits>(q_dense.data(), kv_dense.data(), num_rows, AttnSink[h], o_row);
    });
}

template<class Traits>
void mla_v4_attention_ref_fp8(
    const typename Traits::D_NOPE* Q_nope, const typename Traits::D_ROPE* Q_rope,
    const typename Traits::D_NOPE* UKV_nope, const typename Traits::D_ROPE* UKV_rope,
    const typename Traits::D_NOPE* KV_nope, const typename Traits::D_ROPE* KV_rope,
    const float* AttnSink,
    typename Traits::D_OUT* O,
    const int* kv_indptr_prefix, const int* kv_indices_prefix,
    const int* kv_indptr_extend, const int* kv_indices_extend,
    int N, int H)
{
    using O_t = typename Traits::D_OUT;
    constexpr int D_HEAD = Traits::D_HEAD_SIZE;
    constexpr int NOPE_PADDED = Traits::D_NOPE_PADDED_SIZE;
    constexpr int ROPE = Traits::D_ROPE_SIZE;
    const int o_stride_n = H * D_HEAD;
    const int o_stride_h = D_HEAD;

    mla_v4::parallel_for((size_t)H * N, [&](size_t idx) {
        const int h = static_cast<int>(idx / N);
        const int i = static_cast<int>(idx % N);
        const int prefix_begin = kv_indptr_prefix[i];
        const int extend_begin = kv_indptr_extend[i];
        const int num_prefix = kv_indptr_prefix[i + 1] - prefix_begin;
        const int num_extend = kv_indptr_extend[i + 1] - extend_begin;
        const int num_rows   = num_prefix + num_extend;

        O_t* o_row = O + (size_t)i * o_stride_n + h * o_stride_h;
        if (num_rows <= 0) {
            for (int d = 0; d < D_HEAD; d++) o_row[d] = static_cast<O_t>(0.0f);
            return;
        }

        std::vector<float> q_dense(D_HEAD);
        const size_t q_row = (size_t)i * H + h;
        decode_dsa_row_fp8<Traits>(Q_nope + q_row * NOPE_PADDED, Q_rope + q_row * ROPE, q_dense.data());

        std::vector<float> kv_dense((size_t)num_rows * D_HEAD);
        for (int p = 0; p < num_prefix; p++) {
            const int kv_row = kv_indices_prefix[prefix_begin + p];
            decode_dsa_row_fp8<Traits>(UKV_nope + (size_t)kv_row * NOPE_PADDED, UKV_rope + (size_t)kv_row * ROPE,
                                         kv_dense.data() + (size_t)p * D_HEAD);
        }
        for (int p = 0; p < num_extend; p++) {
            const int kv_row = kv_indices_extend[extend_begin + p];
            decode_dsa_row_fp8<Traits>(KV_nope + (size_t)kv_row * NOPE_PADDED, KV_rope + (size_t)kv_row * ROPE,
                                         kv_dense.data() + (size_t)(num_prefix + p) * D_HEAD);
        }

        mla_v4_attention_compute<Traits>(q_dense.data(), kv_dense.data(), num_rows, AttnSink[h], o_row);
    });
}

template<class Traits>
int run_mla_v4_prefill_case(int H, int N, int total_pages, int total_tokens,
                            int topk, bool verify) {
    using DType = typename Traits::D_ATTN;
    using OType = typename Traits::D_OUT;
    constexpr bool is_fp8 = std::is_same_v<DType, fp8_t> || std::is_same_v<DType, bf8_t>;
    const char* precision = is_fp8 ? "NoPE=fp8, RoPE=bf16" : "NoPE=bf16, RoPE=bf16";
    printf("MLA-v4 Prefill Attention: H_Q=%d, N=%d, D=%d, %s, total_pages=%d, total_tokens=%d, topk=%d\n",
           H, N, Traits::D_HEAD_SIZE, precision, total_pages, total_tokens, topk);

    constexpr int D_HEAD = Traits::D_HEAD_SIZE;
    const size_t o_size = (size_t)N * H * D_HEAD;

    auto host_attn_sink = std::make_unique<float[]>(H);
    auto host_o_ref = std::make_unique<OType[]>(o_size);
    auto host_o_gpu = std::make_unique<OType[]>(o_size);
    const uint64_t seed_sink = MLA_V4_SEED + 1, seed_q = MLA_V4_SEED + 2, seed_ukv = MLA_V4_SEED + 3,
                   seed_kv = MLA_V4_SEED + 4, seed_idx_prefix = MLA_V4_SEED + 5,
                   seed_idx_extend = MLA_V4_SEED + 6;
    rand_vector(host_attn_sink.get(), H, seed_sink);

    std::vector<int> host_kv_indptr_prefix, host_kv_indices_prefix;
    std::vector<int> host_kv_indptr_extend, host_kv_indices_extend;
    init_kv_page_table(host_kv_indptr_prefix, host_kv_indices_prefix, N, total_pages, topk,
                       /*causal=*/false, seed_idx_prefix);
    init_kv_page_table(host_kv_indptr_extend, host_kv_indices_extend, N, total_tokens, topk,
                       /*causal=*/true, seed_idx_extend);
    const size_t total_kv_indices = host_kv_indices_prefix.size() + host_kv_indices_extend.size();
    assert(total_kv_indices <= static_cast<size_t>(std::numeric_limits<int>::max()));
    const int total_kv_rows = static_cast<int>(total_kv_indices);

    dev_buf<float> dev_attn_sink(H);
    dev_buf<OType> dev_o(o_size);
    dev_buf<int> dev_kv_indptr_prefix(host_kv_indptr_prefix.size());
    dev_buf<int> dev_kv_indices_prefix(host_kv_indices_prefix.size());
    dev_buf<int> dev_kv_indptr_extend(host_kv_indptr_extend.size());
    dev_buf<int> dev_kv_indices_extend(host_kv_indices_extend.size());
    dev_o.zero(o_size);
    dev_attn_sink.upload(host_attn_sink.get(), H);
    dev_kv_indptr_prefix.upload(host_kv_indptr_prefix.data(), host_kv_indptr_prefix.size());
    dev_kv_indices_prefix.upload(host_kv_indices_prefix.data(), host_kv_indices_prefix.size());
    dev_kv_indptr_extend.upload(host_kv_indptr_extend.data(), host_kv_indptr_extend.size());
    dev_kv_indices_extend.upload(host_kv_indices_extend.data(), host_kv_indices_extend.size());

    const int num_h_blocks = ceil_div(H, Traits::Q_TILE_SIZE * Traits::T_M);
    dim3 grid(N, num_h_blocks, 1);
    dim3 block(Traits::BLOCK_SIZE);
    printf("MLA-v4 kernel launch config: grid=(%d,%d,%d), block=%d, smem=%zu bytes\n",
           grid.x, grid.y, grid.z, (int)block.x, Traits::smem_size_bytes());

    int rc = 0;
    auto verify_and_bench = [&](const auto& kargs) {
        mla_v4_prefill_launch(Traits{}, kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();
        if (verify) {
            printf("\nValidating GPU results against CPU reference...\n");
            dev_o.download(host_o_gpu.get(), o_size);
            bool all_valid = validate_mla_v4_results<OType>(host_o_ref.get(), host_o_gpu.get(), N, H, D_HEAD);
            printf("\n[Overall] %s\n", all_valid ? "✓ GPU KERNEL VALID" : "✗ GPU KERNEL FAILED");
            if (!all_valid) rc = 1;
        }
        if (!rc) {
            printf("\n");
            benchmark_mla_v4_kernel<Traits>(kargs, grid, block, total_kv_rows);
            printf("\n");
        }
    };

    if constexpr (is_fp8) {
        using D_NOPE = typename Traits::D_NOPE;
        using D_ROPE = typename Traits::D_ROPE;
        constexpr int NOPE_PADDED = Traits::D_NOPE_PADDED_SIZE;
        constexpr int ROPE = Traits::D_ROPE_SIZE;
        const size_t q_nope_size = (size_t)N * H * NOPE_PADDED, q_rope_size = (size_t)N * H * ROPE;
        const size_t ukv_nope_size = (size_t)total_pages * NOPE_PADDED, ukv_rope_size = (size_t)total_pages * ROPE;
        const size_t kv_nope_size = (size_t)total_tokens * NOPE_PADDED, kv_rope_size = (size_t)total_tokens * ROPE;

        auto host_q_nope = std::make_unique<D_NOPE[]>(q_nope_size);
        auto host_q_rope = std::make_unique<D_ROPE[]>(q_rope_size);
        auto host_ukv_nope = std::make_unique<D_NOPE[]>(ukv_nope_size);
        auto host_ukv_rope = std::make_unique<D_ROPE[]>(ukv_rope_size);
        auto host_kv_nope = std::make_unique<D_NOPE[]>(kv_nope_size);
        auto host_kv_rope = std::make_unique<D_ROPE[]>(kv_rope_size);
        init_fp8_dsa_split<Traits>(host_q_nope.get(), host_q_rope.get(), (size_t)N * H, seed_q);
        init_fp8_dsa_split<Traits>(host_ukv_nope.get(), host_ukv_rope.get(), (size_t)total_pages, seed_ukv);
        init_fp8_dsa_split<Traits>(host_kv_nope.get(), host_kv_rope.get(), (size_t)total_tokens, seed_kv);
        poison_kv_rows_fp8<Traits>(host_ukv_nope.get(), host_ukv_rope.get(), total_pages,
                                   host_kv_indices_prefix);
        poison_kv_rows_fp8<Traits>(host_kv_nope.get(), host_kv_rope.get(), total_tokens,
                                   host_kv_indices_extend);

        dev_buf<D_NOPE> dev_q_nope(q_nope_size);
        dev_buf<D_ROPE> dev_q_rope(q_rope_size);
        dev_buf<D_NOPE> dev_ukv_nope(ukv_nope_size);
        dev_buf<D_ROPE> dev_ukv_rope(ukv_rope_size);
        dev_buf<D_NOPE> dev_kv_nope(kv_nope_size);
        dev_buf<D_ROPE> dev_kv_rope(kv_rope_size);
        dev_q_nope.upload(host_q_nope.get(), q_nope_size);
        dev_q_rope.upload(host_q_rope.get(), q_rope_size);
        dev_ukv_nope.upload(host_ukv_nope.get(), ukv_nope_size);
        dev_ukv_rope.upload(host_ukv_rope.get(), ukv_rope_size);
        dev_kv_nope.upload(host_kv_nope.get(), kv_nope_size);
        dev_kv_rope.upload(host_kv_rope.get(), kv_rope_size);

        if (verify)
            mla_v4_attention_ref_fp8<Traits>(host_q_nope.get(), host_q_rope.get(), host_ukv_nope.get(), host_ukv_rope.get(),
                                           host_kv_nope.get(), host_kv_rope.get(), host_attn_sink.get(), host_o_ref.get(),
                                           host_kv_indptr_prefix.data(), host_kv_indices_prefix.data(),
                                           host_kv_indptr_extend.data(), host_kv_indices_extend.data(), N, H);

        opus_mla_v4_prefill_fp8_kargs kargs{};
        kargs.q_nope_ptr = dev_q_nope;
        kargs.q_rope_ptr = dev_q_rope;
        kargs.unified_kv_nope_ptr = dev_ukv_nope;
        kargs.unified_kv_rope_ptr = dev_ukv_rope;
        kargs.kv_nope_ptr = dev_kv_nope;
        kargs.kv_rope_ptr = dev_kv_rope;
        kargs.attn_sink_ptr = dev_attn_sink;
        kargs.out_ptr = dev_o;
        kargs.kv_indptr_prefix = dev_kv_indptr_prefix;
        kargs.kv_indices_prefix = dev_kv_indices_prefix;
        kargs.kv_indptr_extend = dev_kv_indptr_extend;
        kargs.kv_indices_extend = dev_kv_indices_extend;
        kargs.N = N;
        kargs.H = H;
        kargs.total_pages = total_pages;
        kargs.total_tokens = total_tokens;
        kargs.stride_q_nope_n = H * NOPE_PADDED;
        kargs.stride_q_nope_h = NOPE_PADDED;
        kargs.stride_q_rope_n = H * ROPE;
        kargs.stride_q_rope_h = ROPE;
        kargs.stride_o_n = H * D_HEAD;
        kargs.stride_o_h = D_HEAD;
        kargs.stride_kv_nope_page = NOPE_PADDED;
        kargs.stride_kv_rope_page = ROPE;
        kargs.softmax_scale = 1.0f / std::sqrt(static_cast<float>(D_HEAD));

        verify_and_bench(kargs);
    } else {
        constexpr int D = Traits::D_TILE_SIZE;
        const size_t q_size = (size_t)N * H * D;
        const size_t unified_kv_size = (size_t)total_pages * D;
        const size_t kv_size = (size_t)total_tokens * D;

        auto host_q = std::make_unique<DType[]>(q_size);
        auto host_unified_kv = std::make_unique<DType[]>(unified_kv_size);
        auto host_kv = std::make_unique<DType[]>(kv_size);
        rand_vector(host_q.get(), q_size, seed_q);
        rand_vector(host_unified_kv.get(), unified_kv_size, seed_ukv);
        rand_vector(host_kv.get(), kv_size, seed_kv);
        poison_kv_rows<Traits>(host_unified_kv.get(), total_pages, host_kv_indices_prefix);
        poison_kv_rows<Traits>(host_kv.get(), total_tokens, host_kv_indices_extend);

        dev_buf<DType> dev_q(q_size);
        dev_buf<DType> dev_unified_kv(unified_kv_size);
        dev_buf<DType> dev_kv(kv_size);
        dev_q.upload(host_q.get(), q_size);
        dev_unified_kv.upload(host_unified_kv.get(), unified_kv_size);
        dev_kv.upload(host_kv.get(), kv_size);

        if (verify)
            mla_v4_attention_ref<Traits>(host_q.get(), host_unified_kv.get(), host_kv.get(), host_attn_sink.get(), host_o_ref.get(),
                                       host_kv_indptr_prefix.data(), host_kv_indices_prefix.data(),
                                       host_kv_indptr_extend.data(), host_kv_indices_extend.data(), N, H);

        opus_mla_v4_prefill_kargs kargs{};
        kargs.q_ptr = dev_q;
        kargs.unified_kv_ptr = dev_unified_kv;
        kargs.kv_ptr = dev_kv;
        kargs.attn_sink_ptr = dev_attn_sink;
        kargs.out_ptr = dev_o;
        kargs.kv_indptr_prefix = dev_kv_indptr_prefix;
        kargs.kv_indices_prefix = dev_kv_indices_prefix;
        kargs.kv_indptr_extend = dev_kv_indptr_extend;
        kargs.kv_indices_extend = dev_kv_indices_extend;
        kargs.N = N;
        kargs.H = H;
        kargs.D = D;
        kargs.total_pages = total_pages;
        kargs.total_tokens = total_tokens;
        kargs.stride_qo_n = H * D;
        kargs.stride_qo_h = D;
        kargs.stride_kv_page = D;
        kargs.softmax_scale = 1.0f / std::sqrt(static_cast<float>(D_HEAD));

        verify_and_bench(kargs);
    }

    return rc;
}

// 64 MiB of bf16 KV, far past the 4 MiB L2, so the gather is not cache-served.
static constexpr int MLA_V4_DEFAULT_TOTAL_PAGES = 65536;

int main(int argc, char** argv) {
    int H = 128;
    int N = 4096;
    int topk = 1024;
    int total_pages = -1;
    int total_tokens = -1;

    bool verify = false;
    bool use_fp8 = false;
    auto parse_val = [](const char* arg, const char* flag) -> const char* {
        size_t len = std::strlen(flag);
        if (std::strncmp(arg, flag, len) == 0) {
            if (arg[len] == '=') return arg + len + 1;
            if (arg[len] == '\0') return reinterpret_cast<const char*>(1);
        }
        return nullptr;
    };
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const char* val;
        if (std::strcmp(arg, "--verify") == 0) { verify = true; continue; }
        if ((val = parse_val(arg, "-dtype"))) {
            const char* dtype_str = (val == reinterpret_cast<const char*>(1))
                                        ? (i + 1 < argc ? argv[++i] : "")
                                        : val;
            if (std::strcmp(dtype_str, "fp8") == 0) { use_fp8 = true; }
            else if (std::strcmp(dtype_str, "bf16") == 0) { use_fp8 = false; }
            else {
                std::cerr << "-dtype must be 'bf16' or 'fp8', got '" << dtype_str << "'\n";
                return 1;
            }
            continue;
        }
        auto try_parse = [&](int& target, const char* flag) {
            if ((val = parse_val(arg, flag))) {
                if (val == reinterpret_cast<const char*>(1)) { if (i + 1 < argc) target = std::atoi(argv[++i]); }
                else target = std::atoi(val);
                return true;
            }
            return false;
        };
        if (try_parse(H, "-h_q")) continue;
        if (try_parse(N, "-n")) continue;
        if (try_parse(topk, "-topk")) continue;
        if (try_parse(total_pages, "-total_pages")) continue;
        if (try_parse(total_tokens, "-total_tokens")) continue;
        std::cerr << "unknown argument '" << arg << "'\n";
        return 1;
    }
    if (total_pages < 0)
        total_pages = topk > 0 ? std::max({N, topk, MLA_V4_DEFAULT_TOTAL_PAGES}) : N;
    if (total_tokens < 0) total_tokens = N;
    if (topk <= 0) topk = std::max(total_pages, total_tokens);

    if (H <= 0 || N <= 0 || total_pages < 0 || total_tokens < 0) {
        std::cerr << "Invalid parameters. H_Q,N must be positive and "
                     "total_pages,total_tokens non-negative.\n";
        return 1;
    }

#define RUN_CASE(...)                                                          \
    run_mla_v4_prefill_case<__VA_ARGS__>(H, N, total_pages, total_tokens, topk, verify)

#if defined(MLA_V4_ARCH_GFX950)
    if (use_fp8) {
        return H <= 32
            ? RUN_CASE(opus_mla_v4_prefill_a8w8_16mx1_16nx4_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>)
            : RUN_CASE(opus_mla_v4_prefill_a8w8_16mx8_32nx1_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>);
    }
    return H <= 32
        ? RUN_CASE(opus_mla_v4_prefill_a16w16_16mx1_16nx4_traits<16, 64, 512, 4, bf16_t, bf16_t>)
        : RUN_CASE(opus_mla_v4_prefill_a16w16_16mx8_32nx1_traits<16, 32, 512, 8, bf16_t, bf16_t>);
#elif defined(MLA_V4_ARCH_GFX1250)
    const int cluster_y = mla_v4_pick_cluster_y(ceil_div(H, 64));
    if (use_fp8) {
        if (H <= 16) {
            return RUN_CASE(opus_mla_v4_prefill_a8w8_16mx1_16nx4_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>);
        }
        if (H <= 32) {
            return RUN_CASE(opus_mla_v4_prefill_a8w8_32mx1_16nx4_traits<32, 64, 4, fp8_t, bf16_t, bf16_t>);
        }
        switch (cluster_y) {
            case 2: return RUN_CASE(opus_mla_v4_prefill_a8w8_16mx4_64nx1_traits<16, 64, 4, 2, fp8_t, bf16_t, bf16_t>);
            default: return RUN_CASE(opus_mla_v4_prefill_a8w8_16mx4_64nx1_traits<16, 64, 4, 1, fp8_t, bf16_t, bf16_t>);
        }
    }
    if (H <= 16) {
        return RUN_CASE(opus_mla_v4_prefill_a16w16_16mx1_16nx4_traits<16, 64, 512, 4, bf16_t, bf16_t>);
    }
    if (H <= 32) {
        return RUN_CASE(opus_mla_v4_prefill_a16w16_32mx1_16nx4_traits<32, 64, 512, 4, bf16_t, bf16_t>);
    }
    switch (cluster_y) {
        case 2: return RUN_CASE(opus_mla_v4_prefill_a16w16_16mx4_64nx1_traits<16, 64, 512, 4, 2, bf16_t, bf16_t>);
        default: return RUN_CASE(opus_mla_v4_prefill_a16w16_16mx4_64nx1_traits<16, 64, 512, 4, 1, bf16_t, bf16_t>);
    }
#endif
}
