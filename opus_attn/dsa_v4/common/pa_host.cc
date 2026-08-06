// Host-only: benchmark harness, CPU reference, main()
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
#include "common/pa_parallel.h"

#if defined(PA_ARCH_GFX950)
#include "gfx950/pa_traits.h"

template<class Traits>
__global__ void pa_prefill_16mx1_16nx4_kernel(pa_kargs kargs);
template<class Traits>
__global__ void pa_prefill_16mx8_32nx1_kernel(pa_kargs kargs);
template<class Traits>
__global__ void pa_prefill_16mx1_16nx4_fp8_kernel(pa_fp8_kargs kargs);
template<class Traits>
__global__ void pa_prefill_16mx8_32nx1_fp8_kernel(pa_fp8_kargs kargs);

// Launch wrappers — overloaded on the trait type so each selects its own kernel.
template<int Q, int KV, int D, int NW, class DT, class DO>
inline void pa_launch(pa_16mx1_16nx4_traits<Q, KV, D, NW, DT, DO>,
                      const pa_kargs& kargs, dim3 grid, dim3 block) {
    pa_prefill_16mx1_16nx4_kernel<pa_16mx1_16nx4_traits<Q, KV, D, NW, DT, DO>><<<grid, block>>>(kargs);
}
template<int Q, int KV, int D, int NW, class DT, class DO>
inline void pa_launch(pa_16mx8_32nx1_traits<Q, KV, D, NW, DT, DO>,
                      const pa_kargs& kargs, dim3 grid, dim3 block) {
    pa_prefill_16mx8_32nx1_kernel<pa_16mx8_32nx1_traits<Q, KV, D, NW, DT, DO>><<<grid, block>>>(kargs);
}
template<int Q, int KV, int NW, class NOPE, class ROPE, class DO>
inline void pa_launch(pa_16mx1_16nx4_fp8_traits<Q, KV, NW, NOPE, ROPE, DO>,
                      const pa_fp8_kargs& kargs, dim3 grid, dim3 block) {
    pa_prefill_16mx1_16nx4_fp8_kernel<pa_16mx1_16nx4_fp8_traits<Q, KV, NW, NOPE, ROPE, DO>><<<grid, block>>>(kargs);
}
template<int Q, int KV, int NW, class NOPE, class ROPE, class DO>
inline void pa_launch(pa_16mx8_32nx1_fp8_traits<Q, KV, NW, NOPE, ROPE, DO>,
                      const pa_fp8_kargs& kargs, dim3 grid, dim3 block) {
    pa_prefill_16mx8_32nx1_fp8_kernel<pa_16mx8_32nx1_fp8_traits<Q, KV, NW, NOPE, ROPE, DO>><<<grid, block>>>(kargs);
}

#elif defined(PA_ARCH_GFX1250)
#include "gfx1250/pa_traits.h"

template<class Traits>
__global__ void pa_prefill_16mx4_64nx1_kernel(pa_kargs kargs);
template<class Traits>
__global__ void pa_prefill_16mx4_64nx1_fp8_kernel(pa_fp8_kargs kargs);

template<int Q, int KV, int D, int NW, class DT, class DO>
inline void pa_launch(pa_16mx4_64nx1_traits<Q, KV, D, NW, DT, DO>,
                      const pa_kargs& kargs, dim3 grid, dim3 block) {
    pa_prefill_16mx4_64nx1_kernel<pa_16mx4_64nx1_traits<Q, KV, D, NW, DT, DO>><<<grid, block>>>(kargs);
}
template<int Q, int KV, int NW, class NOPE, class ROPE, class DO>
inline void pa_launch(pa_16mx4_64nx1_fp8_traits<Q, KV, NW, NOPE, ROPE, DO>,
                      const pa_fp8_kargs& kargs, dim3 grid, dim3 block) {
    pa_prefill_16mx4_64nx1_fp8_kernel<pa_16mx4_64nx1_fp8_traits<Q, KV, NW, NOPE, ROPE, DO>><<<grid, block>>>(kargs);
}
#else
#  error "No target arch defined. The Makefile passes PA_ARCH_<ARCH> from ARCH (e.g. ARCH=gfx950)."
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

// Fill a contiguous vector with random values
template<typename T>
void rand_vector(T* ptr, size_t size, float min_val = 0.0f, float max_val = 1.0f) {
    pa::parallel_chunks(size, pa::default_grain(size), [&](size_t begin, size_t end, unsigned) {
        std::random_device rd;
        std::mt19937 gen(rd() + static_cast<uint32_t>(begin));
        std::uniform_real_distribution<float> dis(min_val, max_val);
        for (size_t i = begin; i < end; i++) {
            ptr[i] = static_cast<T>(dis(gen));
        }
    });
}

// Initialize the split DSA fp8 streams. The NoPE stream packs, per row of
// D_NOPE_PADDED_SIZE fp8 slots: [ NoPE fp8 (D_NOPE_SIZE) | E8M0 block scales
// (D_NOPE_SIZE/32) | fp8 zero-pad ]. The RoPE stream holds D_ROPE_SIZE bf16.
template<class PATraits>
void init_fp8_dsa_split(typename PATraits::D_NOPE* nope_ptr,
                        typename PATraits::D_ROPE* rope_ptr, size_t rows) {
    using D_ROPE = typename PATraits::D_ROPE;
    constexpr int NOPE_PADDED = PATraits::D_NOPE_PADDED_SIZE;  // fp8 slots/row (512)
    constexpr int NOPE        = PATraits::D_NOPE_SIZE;         // NoPE fp8 elements (448)
    constexpr int SCALE       = NOPE / 32;                     // E8M0 scales (14)
    constexpr int ROPE        = PATraits::D_ROPE_SIZE;         // RoPE bf16 elements (64)
    static_assert(NOPE + SCALE <= NOPE_PADDED, "NoPE + scales exceed padded row");

    pa::parallel_chunks(rows, pa::default_grain(rows), [&](size_t begin, size_t end, unsigned) {
        std::random_device rd;
        std::mt19937 gen(rd() + static_cast<uint32_t>(begin));
        std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
        std::uniform_real_distribution<float> scale_dis(-4.0f, 4.0f);
        for (size_t r = begin; r < end; r++) {
            unsigned char* nbase = reinterpret_cast<unsigned char*>(nope_ptr) + r * NOPE_PADDED;
            auto* nope = reinterpret_cast<__hip_fp8_e4m3*>(nbase);
            for (int i = 0; i < NOPE; i++) nope[i] = static_cast<__hip_fp8_e4m3>(dis(gen));
            unsigned char* scale = nbase + NOPE;
            for (int i = 0; i < SCALE; i++) {
                float s = std::exp2(scale_dis(gen));
                uint32_t bits; std::memcpy(&bits, &s, sizeof(bits));
                scale[i] = static_cast<unsigned char>((bits >> 23) & 0xFF);
            }
            for (int i = NOPE + SCALE; i < NOPE_PADDED; i++) nbase[i] = 0;
            D_ROPE* rope = rope_ptr + r * ROPE;
            for (int i = 0; i < ROPE; i++) rope[i] = static_cast<D_ROPE>(dis(gen));
        }
    });
}

void init_sparse_kv_indices(std::vector<int>& kv_indptr,
                            std::vector<int>& kv_indices,
                            int N,
                            int total_pages,
                            int kv_tile_size,
                            uint32_t seed = 1234) {
    assert(N >= 0);
    assert(total_pages > 0);
    assert(kv_tile_size > 0);

    kv_indptr.assign(N + 1, 0);
    kv_indices.clear();

    std::mt19937 gen(seed);
    std::vector<int> pages(total_pages);
    std::iota(pages.begin(), pages.end(), 0);

    auto clamp_len = [&](int len) {
        return std::max(0, std::min(len, total_pages));
    };

    const std::vector<int> boundary_lengths = {
        0,
        1,
        kv_tile_size - 1,
        kv_tile_size,
        kv_tile_size + 1,
        2 * kv_tile_size,
        2 * kv_tile_size + 1,
        total_pages
    };
    std::uniform_int_distribution<int> random_len(0, total_pages);

    for (int q = 0; q < N; ++q) {
        int nnz = 0;
        if (q < static_cast<int>(boundary_lengths.size())) {
            nnz = clamp_len(boundary_lengths[q]);
        } else {
            nnz = random_len(gen);
        }

        std::shuffle(pages.begin(), pages.end(), gen);
        const size_t seg_begin = kv_indices.size();
        kv_indices.insert(kv_indices.end(), pages.begin(), pages.begin() + nnz);
        std::sort(kv_indices.begin() + seg_begin, kv_indices.end());
        assert(kv_indices.size() <= static_cast<size_t>(std::numeric_limits<int>::max()));
        kv_indptr[q + 1] = static_cast<int>(kv_indices.size());
    }

    assert(kv_indptr.front() == 0);
    assert(kv_indptr.back() == static_cast<int>(kv_indices.size()));
    for (int q = 0; q < N; ++q) {
        assert(kv_indptr[q] <= kv_indptr[q + 1]);
        for (int p = kv_indptr[q]; p < kv_indptr[q + 1]; ++p) {
            assert(kv_indices[p] >= 0 && kv_indices[p] < total_pages);
            if (p > kv_indptr[q])
                assert(kv_indices[p] >= kv_indices[p - 1]);
        }
    }
}

void init_dense_kv_indices(std::vector<int>& kv_indptr,
                           std::vector<int>& kv_indices,
                           int N,
                           int total_pages) {
    assert(N >= 0);
    assert(total_pages > 0);
    const size_t total_indices = static_cast<size_t>(N) * total_pages;
    assert(total_indices <= static_cast<size_t>(std::numeric_limits<int>::max()));

    kv_indptr.resize(N + 1);
    kv_indices.resize(total_indices);

    for (int q = 0; q <= N; ++q) {
        kv_indptr[q] = static_cast<int>(static_cast<size_t>(q) * total_pages);
    }
    for (int q = 0; q < N; ++q) {
        const size_t row_begin = static_cast<size_t>(q) * total_pages;
        for (int page = 0; page < total_pages; ++page) {
            kv_indices[row_begin + page] = page;
        }
    }
}

// Benchmark PA kernel performance with warm-up and timing
template<class Traits, class KArgs>
void benchmark_pa_kernel(const KArgs& kargs, dim3 grid, dim3 block,
                          int indices_prefix_sum, int warmup = 100, int iterations = 50) {
    for (int i = 0; i < warmup; ++i) {
        pa_launch(Traits{}, kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        pa_launch(Traits{}, kargs, grid, block);
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
    constexpr int D_HEAD = Traits::D_HEAD_SIZE;  // logical head dim (NoPE + RoPE), e.g. 512

    // FLOPs: per (query, kv, head) -> QK^T (2*D_HEAD) + PV (2*D_HEAD).
    const double flops = 4.0 * kargs.H * indices_prefix_sum * D_HEAD;
    const double tflops = flops / (avg_time * 1e-3) / 1e12;

    // Bandwidth: Q read (packed row) + O write (bf16) + KV read (packed row), each its own dtype.
    size_t row_bytes;
    if constexpr (std::is_same_v<KArgs, pa_fp8_kargs>) {
        row_bytes = (size_t)Traits::D_NOPE_PADDED_SIZE * sizeof(typename Traits::D_NOPE)
                  + (size_t)Traits::D_ROPE_SIZE * sizeof(typename Traits::D_ROPE);
    } else {
        row_bytes = (size_t)Traits::D_TILE_SIZE * sizeof(D_ATTN);
    }
    const size_t q_bytes  = (size_t)kargs.N * kargs.H * row_bytes;
    const size_t o_bytes  = (size_t)kargs.N * kargs.H * D_HEAD * sizeof(D_OUT);
    const size_t kv_bytes = (size_t)indices_prefix_sum * row_bytes;
    const double tbps = double(q_bytes + o_bytes + kv_bytes) / (avg_time * 1e-3) / 1e12;

    printf("PA Prefill Kernel Performance: avg_time=%.3f ms, %.2f TFlops, %.2f TB/s\n",
           avg_time, tflops, tbps);
}

// Validate PA GPU results against CPU reference.
template<typename DType>
bool validate_pa_results(const DType* ref, const DType* gpu,
                          int N, int H, int D,
                          float rtol = 1e-2f, float atol = 1e-2f,
                          float tol_err_ratio = 0.05f) {
    const size_t total_elements = (size_t)N * H * D;
    constexpr size_t printNum = 10;

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

                const bool nan_inf = std::isnan(gpu_val) || std::isinf(gpu_val);
                any_nan |= nan_inf;
                if (nan_inf || delta > atol + rtol * std::abs(ref_val)) {
                    total_errors++;
                    max_abs_delta = std::max(max_abs_delta, delta);
                    if (printed++ < printNum)
                        printf("  mismatch [n=%d,h=%d,d=%d] ref=%.6f gpu=%.6f delta=%.6f\n",
                               n, h, d, ref_val, gpu_val, delta);
                }
            }
        }
    }

    const double err_ratio    = double(total_errors) / double(total_elements);
    const double nrms         = std::sqrt(sq_diff_sum / std::max(ref_sq_sum, 1e-12));  // ||gpu-ref|| / ||ref||
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

// Reconstruct one bf16 stored row into a dense float[D_HEAD_SIZE].
template<class PATraits>
inline void decode_dsa_row_bf16(const typename PATraits::D_ATTN* row, float* out) {
    constexpr int D_HEAD = PATraits::D_HEAD_SIZE;
    for (int d = 0; d < D_HEAD; d++) out[d] = static_cast<float>(row[d]);
}

// Reconstruct one split fp8 row (NoPE+scales fp8 stream, RoPE bf16 stream) into dense float.
template<class PATraits>
inline void decode_dsa_row_fp8(const typename PATraits::D_NOPE* nrow,
                               const typename PATraits::D_ROPE* rrow, float* out) {
    constexpr int NOPE = PATraits::D_NOPE_SIZE;
    constexpr int ROPE = PATraits::D_ROPE_SIZE;
    const auto* base = reinterpret_cast<const unsigned char*>(nrow);
    const auto* nope = reinterpret_cast<const __hip_fp8_e4m3*>(base);
    const unsigned char* scale = base + NOPE;  // raw E8M0 bytes
    for (int d = 0; d < NOPE; d++)
        out[d] = static_cast<float>(nope[d]) * std::ldexp(1.0f, int(scale[d / 32]) - 127);
    for (int j = 0; j < ROPE; j++)
        out[NOPE + j] = static_cast<float>(rrow[j]);
}

// ─── CPU reference: Paged Attention (PA) ──────────────────────────
//
// Sparse scaled-dot-product attention over two CSR ranges:
//   prefix rows index UnifiedKV[total_pages, D]
//   extend rows index KV[total_tokens, D]
//   O[i,h,:] = softmax(Q[i,h,:] @ concat(prefix, extend)^T * softmax_scale) @ concat(prefix, extend)
//
// Softmax + PV for one (query, head) over already-dequantized dense rows.
template<class PATraits>
inline void pa_attention_compute(const float* q_dense, const float* kv_dense, int num_rows,
                                 float sink, typename PATraits::D_OUT* o_row) {
    using O_t = typename PATraits::D_OUT;
    constexpr int D_HEAD = PATraits::D_HEAD_SIZE;
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
        scores[p] = static_cast<float>(static_cast<bf16_t>(scores[p] / sum_exp));  // P rounded to bf16
    for (int d = 0; d < D_HEAD; d++) {
        float acc = 0.0f;
        for (int p = 0; p < num_rows; p++) acc += scores[p] * kv_dense[(size_t)p * D_HEAD + d];
        o_row[d] = static_cast<O_t>(acc);
    }
}

template<class PATraits>
void pa_attention_ref(
    const typename PATraits::D_ATTN* Q,         // [N, H, ROW]  (ROW = D_TILE_SIZE storage stride)
    const typename PATraits::D_ATTN* UnifiedKV, // [total_pages, ROW]
    const typename PATraits::D_ATTN* KV,        // [total_tokens, ROW]
    const float*  AttnSink,                     // [H]
    typename PATraits::D_OUT* O,                // [N, H, D_HEAD]
    const int* kv_indptr_prefix,
    const int* kv_indices_prefix,
    const int* kv_indptr_extend,
    const int* kv_indices_extend,
    int N, int H)
{
    using O_t = typename PATraits::D_OUT;
    constexpr int D_HEAD = PATraits::D_HEAD_SIZE;
    constexpr int ROW    = PATraits::D_TILE_SIZE;
    const int stride_qo_n = H * ROW;
    const int stride_qo_h = ROW;
    const int stride_kv_page = ROW;
    const int o_stride_n = H * D_HEAD;
    const int o_stride_h = D_HEAD;

    pa::parallel_for((size_t)H * N, [&](size_t idx) {
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
        decode_dsa_row_bf16<PATraits>(Q + (size_t)i * stride_qo_n + h * stride_qo_h, q_dense.data());

        std::vector<float> kv_dense((size_t)num_rows * D_HEAD);
        for (int p = 0; p < num_prefix; p++)
            decode_dsa_row_bf16<PATraits>(UnifiedKV + (size_t)kv_indices_prefix[prefix_begin + p] * stride_kv_page,
                                          kv_dense.data() + (size_t)p * D_HEAD);
        for (int p = 0; p < num_extend; p++)
            decode_dsa_row_bf16<PATraits>(KV + (size_t)kv_indices_extend[extend_begin + p] * stride_kv_page,
                                          kv_dense.data() + (size_t)(num_prefix + p) * D_HEAD);

        pa_attention_compute<PATraits>(q_dense.data(), kv_dense.data(), num_rows, AttnSink[h], o_row);
    });
}

// fp8 reference operating on the split NoPE (fp8) and RoPE (bf16) streams.
template<class PATraits>
void pa_attention_ref_fp8(
    const typename PATraits::D_NOPE* Q_nope, const typename PATraits::D_ROPE* Q_rope,
    const typename PATraits::D_NOPE* UKV_nope, const typename PATraits::D_ROPE* UKV_rope,
    const typename PATraits::D_NOPE* KV_nope, const typename PATraits::D_ROPE* KV_rope,
    const float* AttnSink,
    typename PATraits::D_OUT* O,
    const int* kv_indptr_prefix, const int* kv_indices_prefix,
    const int* kv_indptr_extend, const int* kv_indices_extend,
    int N, int H)
{
    using O_t = typename PATraits::D_OUT;
    constexpr int D_HEAD = PATraits::D_HEAD_SIZE;
    constexpr int NOPE_PADDED = PATraits::D_NOPE_PADDED_SIZE;
    constexpr int ROPE = PATraits::D_ROPE_SIZE;
    const int o_stride_n = H * D_HEAD;
    const int o_stride_h = D_HEAD;

    pa::parallel_for((size_t)H * N, [&](size_t idx) {
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
        decode_dsa_row_fp8<PATraits>(Q_nope + q_row * NOPE_PADDED, Q_rope + q_row * ROPE, q_dense.data());

        std::vector<float> kv_dense((size_t)num_rows * D_HEAD);
        for (int p = 0; p < num_prefix; p++) {
            const int kv_row = kv_indices_prefix[prefix_begin + p];
            decode_dsa_row_fp8<PATraits>(UKV_nope + (size_t)kv_row * NOPE_PADDED, UKV_rope + (size_t)kv_row * ROPE,
                                         kv_dense.data() + (size_t)p * D_HEAD);
        }
        for (int p = 0; p < num_extend; p++) {
            const int kv_row = kv_indices_extend[extend_begin + p];
            decode_dsa_row_fp8<PATraits>(KV_nope + (size_t)kv_row * NOPE_PADDED, KV_rope + (size_t)kv_row * ROPE,
                                         kv_dense.data() + (size_t)(num_prefix + p) * D_HEAD);
        }

        pa_attention_compute<PATraits>(q_dense.data(), kv_dense.data(), num_rows, AttnSink[h], o_row);
    });
}

// ─── main ───────────────────────────────────────────────────────────────────

template<class PATraits>
int run_pa_case(int H, int N, int total_pages, int total_tokens,
                bool verify, bool dense_kv) {
    using DType = typename PATraits::D_ATTN;   // input storage dtype (fp8 packed, or bf16)
    using OType = typename PATraits::D_OUT;     // output dtype (default bf16)
    constexpr bool is_fp8 = std::is_same_v<DType, fp8_t> || std::is_same_v<DType, bf8_t>;
    const char* precision = is_fp8 ? "NoPE=fp8, RoPE=bf16" : "NoPE=bf16, RoPE=bf16";
    printf("PA Prefill Attention: H_Q=%d, N=%d, D=%d, %s, total_pages=%d, total_tokens=%d\n",
           H, N, PATraits::D_HEAD_SIZE, precision, total_pages, total_tokens);

    constexpr int D_HEAD = PATraits::D_HEAD_SIZE;
    const size_t o_size = (size_t)N * H * D_HEAD;

    auto host_attn_sink = std::make_unique<float[]>(H);
    auto host_o_ref = std::make_unique<OType[]>(o_size);
    auto host_o_gpu = std::make_unique<OType[]>(o_size);
    rand_vector(host_attn_sink.get(), H, -2.f, 2.f);

    std::vector<int> host_kv_indptr_prefix, host_kv_indices_prefix;
    std::vector<int> host_kv_indptr_extend, host_kv_indices_extend;
    if (dense_kv) {
        init_dense_kv_indices(host_kv_indptr_prefix, host_kv_indices_prefix, N, total_pages);
        init_dense_kv_indices(host_kv_indptr_extend, host_kv_indices_extend, N, total_tokens);
    } else {
        init_sparse_kv_indices(host_kv_indptr_prefix, host_kv_indices_prefix, N, total_pages, PATraits::KV_TILE_SIZE, 1234);
        init_sparse_kv_indices(host_kv_indptr_extend, host_kv_indices_extend, N, total_tokens, PATraits::KV_TILE_SIZE, 5678);
    }
    const size_t total_kv_indices = host_kv_indices_prefix.size() + host_kv_indices_extend.size();
    assert(total_kv_indices <= static_cast<size_t>(std::numeric_limits<int>::max()));
    const int indices_prefix_sum = static_cast<int>(total_kv_indices);

    float *dev_attn_sink;
    OType *dev_o;
    int *dev_kv_indptr_prefix, *dev_kv_indices_prefix, *dev_kv_indptr_extend, *dev_kv_indices_extend;
    const size_t kv_indices_prefix_alloc_size = std::max<size_t>(host_kv_indices_prefix.size(), 1);
    const size_t kv_indices_extend_alloc_size = std::max<size_t>(host_kv_indices_extend.size(), 1);
    CHECK_HIP(hipMalloc(&dev_attn_sink, H * sizeof(float)));
    CHECK_HIP(hipMalloc(&dev_o, o_size * sizeof(OType)));
    CHECK_HIP(hipMemset(dev_o, 0, o_size * sizeof(OType)));
    CHECK_HIP(hipMalloc(&dev_kv_indptr_prefix, host_kv_indptr_prefix.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&dev_kv_indices_prefix, kv_indices_prefix_alloc_size * sizeof(int)));
    CHECK_HIP(hipMalloc(&dev_kv_indptr_extend, host_kv_indptr_extend.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&dev_kv_indices_extend, kv_indices_extend_alloc_size * sizeof(int)));
    CHECK_HIP(hipMemcpy(dev_attn_sink, host_attn_sink.get(), H * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_kv_indptr_prefix, host_kv_indptr_prefix.data(), host_kv_indptr_prefix.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_kv_indptr_extend, host_kv_indptr_extend.data(), host_kv_indptr_extend.size() * sizeof(int), hipMemcpyHostToDevice));
    if (!host_kv_indices_prefix.empty())
        CHECK_HIP(hipMemcpy(dev_kv_indices_prefix, host_kv_indices_prefix.data(), host_kv_indices_prefix.size() * sizeof(int), hipMemcpyHostToDevice));
    if (!host_kv_indices_extend.empty())
        CHECK_HIP(hipMemcpy(dev_kv_indices_extend, host_kv_indices_extend.data(), host_kv_indices_extend.size() * sizeof(int), hipMemcpyHostToDevice));

    const int num_h_blocks = ceil_div(H, PATraits::Q_TILE_SIZE * PATraits::T_M);
    dim3 grid(N, num_h_blocks, 1);
    dim3 block(PATraits::BLOCK_SIZE);
    printf("PA kernel launch config: grid=(%d,%d,%d), block=%d (NUM_WARPS=%d), smem=%zu bytes (K/V tiles)\n",
           grid.x, grid.y, grid.z, (int)block.x, PATraits::NUM_WARPS, PATraits::smem_size_bytes());

    int rc = 0;
    auto verify_and_bench = [&](const auto& kargs) {
        pa_launch(PATraits{}, kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();
        if (verify) {
            printf("\nValidating GPU results against CPU reference...\n");
            CHECK_HIP(hipMemcpy(host_o_gpu.get(), dev_o, o_size * sizeof(OType), hipMemcpyDeviceToHost));
            bool all_valid = validate_pa_results<OType>(host_o_ref.get(), host_o_gpu.get(), N, H, D_HEAD);
            printf("\n[Overall] %s\n", all_valid ? "✓ GPU KERNEL VALID" : "✗ GPU KERNEL FAILED");
            if (!all_valid) rc = 1;
        }
        if (!rc) {
            printf("\n");
            benchmark_pa_kernel<PATraits>(kargs, grid, block, indices_prefix_sum);
            printf("\n");
        }
    };

    if constexpr (is_fp8) {
        using D_NOPE = typename PATraits::D_NOPE;
        using D_ROPE = typename PATraits::D_ROPE;
        constexpr int NOPE_PADDED = PATraits::D_NOPE_PADDED_SIZE;
        constexpr int ROPE = PATraits::D_ROPE_SIZE;
        const size_t q_nope_size = (size_t)N * H * NOPE_PADDED, q_rope_size = (size_t)N * H * ROPE;
        const size_t ukv_nope_size = (size_t)total_pages * NOPE_PADDED, ukv_rope_size = (size_t)total_pages * ROPE;
        const size_t kv_nope_size = (size_t)total_tokens * NOPE_PADDED, kv_rope_size = (size_t)total_tokens * ROPE;

        auto host_q_nope = std::make_unique<D_NOPE[]>(q_nope_size);
        auto host_q_rope = std::make_unique<D_ROPE[]>(q_rope_size);
        auto host_ukv_nope = std::make_unique<D_NOPE[]>(ukv_nope_size);
        auto host_ukv_rope = std::make_unique<D_ROPE[]>(ukv_rope_size);
        auto host_kv_nope = std::make_unique<D_NOPE[]>(kv_nope_size);
        auto host_kv_rope = std::make_unique<D_ROPE[]>(kv_rope_size);
        init_fp8_dsa_split<PATraits>(host_q_nope.get(), host_q_rope.get(), (size_t)N * H);
        init_fp8_dsa_split<PATraits>(host_ukv_nope.get(), host_ukv_rope.get(), (size_t)total_pages);
        init_fp8_dsa_split<PATraits>(host_kv_nope.get(), host_kv_rope.get(), (size_t)total_tokens);

        D_NOPE *dev_q_nope, *dev_ukv_nope, *dev_kv_nope;
        D_ROPE *dev_q_rope, *dev_ukv_rope, *dev_kv_rope;
        CHECK_HIP(hipMalloc(&dev_q_nope, q_nope_size * sizeof(D_NOPE)));
        CHECK_HIP(hipMalloc(&dev_q_rope, q_rope_size * sizeof(D_ROPE)));
        CHECK_HIP(hipMalloc(&dev_ukv_nope, ukv_nope_size * sizeof(D_NOPE)));
        CHECK_HIP(hipMalloc(&dev_ukv_rope, ukv_rope_size * sizeof(D_ROPE)));
        CHECK_HIP(hipMalloc(&dev_kv_nope, kv_nope_size * sizeof(D_NOPE)));
        CHECK_HIP(hipMalloc(&dev_kv_rope, kv_rope_size * sizeof(D_ROPE)));
        CHECK_HIP(hipMemcpy(dev_q_nope, host_q_nope.get(), q_nope_size * sizeof(D_NOPE), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(dev_q_rope, host_q_rope.get(), q_rope_size * sizeof(D_ROPE), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(dev_ukv_nope, host_ukv_nope.get(), ukv_nope_size * sizeof(D_NOPE), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(dev_ukv_rope, host_ukv_rope.get(), ukv_rope_size * sizeof(D_ROPE), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(dev_kv_nope, host_kv_nope.get(), kv_nope_size * sizeof(D_NOPE), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(dev_kv_rope, host_kv_rope.get(), kv_rope_size * sizeof(D_ROPE), hipMemcpyHostToDevice));

        if (verify)
            pa_attention_ref_fp8<PATraits>(host_q_nope.get(), host_q_rope.get(), host_ukv_nope.get(), host_ukv_rope.get(),
                                           host_kv_nope.get(), host_kv_rope.get(), host_attn_sink.get(), host_o_ref.get(),
                                           host_kv_indptr_prefix.data(), host_kv_indices_prefix.data(),
                                           host_kv_indptr_extend.data(), host_kv_indices_extend.data(), N, H);

        pa_fp8_kargs kargs{};
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

        CHECK_HIP(hipFree(dev_q_nope));   CHECK_HIP(hipFree(dev_q_rope));
        CHECK_HIP(hipFree(dev_ukv_nope)); CHECK_HIP(hipFree(dev_ukv_rope));
        CHECK_HIP(hipFree(dev_kv_nope));  CHECK_HIP(hipFree(dev_kv_rope));
    } else {
        constexpr int D = PATraits::D_TILE_SIZE;
        const size_t q_size = (size_t)N * H * D;
        const size_t unified_kv_size = (size_t)total_pages * D;
        const size_t kv_size = (size_t)total_tokens * D;

        auto host_q = std::make_unique<DType[]>(q_size);
        auto host_unified_kv = std::make_unique<DType[]>(unified_kv_size);
        auto host_kv = std::make_unique<DType[]>(kv_size);
        rand_vector(host_q.get(), q_size, -2.f, 2.f);
        rand_vector(host_unified_kv.get(), unified_kv_size, -2.f, 2.f);
        rand_vector(host_kv.get(), kv_size, -2.f, 2.f);

        DType *dev_q, *dev_unified_kv, *dev_kv;
        CHECK_HIP(hipMalloc(&dev_q, q_size * sizeof(DType)));
        CHECK_HIP(hipMalloc(&dev_unified_kv, unified_kv_size * sizeof(DType)));
        CHECK_HIP(hipMalloc(&dev_kv, kv_size * sizeof(DType)));
        CHECK_HIP(hipMemcpy(dev_q, host_q.get(), q_size * sizeof(DType), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(dev_unified_kv, host_unified_kv.get(), unified_kv_size * sizeof(DType), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(dev_kv, host_kv.get(), kv_size * sizeof(DType), hipMemcpyHostToDevice));

        if (verify)
            pa_attention_ref<PATraits>(host_q.get(), host_unified_kv.get(), host_kv.get(), host_attn_sink.get(), host_o_ref.get(),
                                       host_kv_indptr_prefix.data(), host_kv_indices_prefix.data(),
                                       host_kv_indptr_extend.data(), host_kv_indices_extend.data(), N, H);

        pa_kargs kargs{};
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

        CHECK_HIP(hipFree(dev_q));
        CHECK_HIP(hipFree(dev_unified_kv));
        CHECK_HIP(hipFree(dev_kv));
    }

    CHECK_HIP(hipFree(dev_attn_sink));
    CHECK_HIP(hipFree(dev_o));
    CHECK_HIP(hipFree(dev_kv_indptr_prefix));
    CHECK_HIP(hipFree(dev_kv_indices_prefix));
    CHECK_HIP(hipFree(dev_kv_indptr_extend));
    CHECK_HIP(hipFree(dev_kv_indices_extend));

    return rc;
}

int main(int argc, char** argv) {
    int H = 128;   // query heads
    int N = 1024;  // sequence length
    int total_pages = -1; // rows in unified_kv; default N after parsing
    int total_tokens = -1; // rows in the per-fwd extend KV tensor; default N

    // Parse command line arguments.
    bool verify = false;
    bool dense_kv = false;
    bool use_fp8 = false;
    auto parse_val = [](const char* arg, const char* flag) -> const char* {
        size_t len = std::strlen(flag);
        if (std::strncmp(arg, flag, len) == 0) {
            if (arg[len] == '=') return arg + len + 1;       // -flag=value
            if (arg[len] == '\0') return reinterpret_cast<const char*>(1); // -flag value (next arg)
        }
        return nullptr;
    };
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const char* val;
        if (std::strcmp(arg, "--verify") == 0) { verify = true; continue; }
        if (std::strcmp(arg, "--dense") == 0) { dense_kv = true; continue; }
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
        if (try_parse(total_pages, "-total_pages")) continue;
        if (try_parse(total_tokens, "-total_tokens")) continue;
    }
    if (total_pages < 0) {
        total_pages = N;
    }
    if (total_tokens < 0) {
        total_tokens = N;
    }

    if (H <= 0 || N <= 0 || total_pages <= 0 || total_tokens <= 0) {
        std::cerr << "Invalid parameters. H_Q,N,total_pages,total_tokens must be positive.\n";
        return 1;
    }

#if defined(PA_ARCH_GFX950)
    if (use_fp8) {
        return H <= 32
            ? run_pa_case<pa_16mx1_16nx4_fp8_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>>(H, N, total_pages, total_tokens, verify, dense_kv)
            : run_pa_case<pa_16mx8_32nx1_fp8_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>>(H, N, total_pages, total_tokens, verify, dense_kv);
    }
    return H <= 32
        ? run_pa_case<pa_16mx1_16nx4_traits<16, 64, 512, 4, bf16_t, bf16_t>>(H, N, total_pages, total_tokens, verify, dense_kv)
        : run_pa_case<pa_16mx8_32nx1_traits<16, 32, 512, 8, bf16_t, bf16_t>>(H, N, total_pages, total_tokens, verify, dense_kv);
#elif defined(PA_ARCH_GFX1250)
    if (use_fp8) {
        return run_pa_case<pa_16mx4_64nx1_fp8_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>>(H, N, total_pages, total_tokens, verify, dense_kv);
    }
    return run_pa_case<pa_16mx4_64nx1_traits<16, 64, 512, 4, bf16_t, bf16_t>>(H, N, total_pages, total_tokens, verify, dense_kv);
#endif
}
