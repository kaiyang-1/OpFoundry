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
#include <omp.h>

#include "pa_defs.h"

// Declared in per-variant/per-dtype kernel instantiation TUs.
template<class Traits>
__global__ void pa_prefill_16mx1_16nx4_kernel(pa_kargs kargs);
template<class Traits>
__global__ void pa_prefill_16mx8_32nx1_kernel(pa_kargs kargs);
template<class Traits>
__global__ void pa_prefill_16mx1_16nx4_fp8_kernel(pa_kargs kargs);

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
template<int Q, int KV, int D, int NW, class NOPE, class ROPE, class DO>
inline void pa_launch(pa_16mx1_16nx4_fp8_traits<Q, KV, D, NW, NOPE, ROPE, DO>,
                      const pa_kargs& kargs, dim3 grid, dim3 block) {
    pa_prefill_16mx1_16nx4_fp8_kernel<pa_16mx1_16nx4_fp8_traits<Q, KV, D, NW, NOPE, ROPE, DO>><<<grid, block>>>(kargs);
}

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
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        #pragma omp for
        for (size_t i = 0; i < size; i++) {
            ptr[i] = static_cast<T>(dis(gen));
        }
    }
}

// Initialize a packed DeepSeek sparse-attention (DSA) fp8 tensor. Each row spans
// PATraits::D_TILE_SIZE bytes laid out as:
//   [ NoPE fp8 (D_NOPE_SIZE) | fp8 block scales (D_NOPE_SIZE/32) | fp8 zero-pad | RoPE bf16 (D_ROPE_SIZE) ]
// NoPE/RoPE filled with random values (via HIP fp8 / bf16 casts), scales set to 1.0.
template<class PATraits>
void init_fp8_dsa_tensor(typename PATraits::D_ATTN* ptr, size_t rows) {
    using D_ROPE = typename PATraits::D_ROPE;
    constexpr int ROW        = PATraits::D_TILE_SIZE;            // total bytes/row (e.g. 640)
    constexpr int NOPE       = PATraits::D_NOPE_SIZE;            // NoPE fp8 elements (448)
    constexpr int SCALE      = NOPE / 32;                        // fp8 scales, one per 32-elem block (14)
    constexpr int ROPE       = PATraits::D_ROPE_SIZE;            // RoPE bf16 elements (64)
    constexpr int ROPE_BYTES = ROPE * (int)sizeof(D_ROPE);      // 128
    constexpr int ZERO       = ROW - NOPE - SCALE - ROPE_BYTES; // fp8 zero pad (50)
    constexpr int ROPE_OFF   = NOPE + SCALE + ZERO;             // byte offset of RoPE (512)
    static_assert(ZERO >= 0, "DSA row layout does not fit in D_TILE_SIZE bytes");

    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
        std::uniform_real_distribution<float> scale_dis(-4.0f, 4.0f);  // log2 range for E8M0 block scales
        #pragma omp for
        for (size_t r = 0; r < rows; r++) {
            unsigned char* base = reinterpret_cast<unsigned char*>(ptr) + r * ROW;
            auto* nope = reinterpret_cast<__hip_fp8_e4m3*>(base);
            for (int i = 0; i < NOPE; i++) nope[i] = static_cast<__hip_fp8_e4m3>(dis(gen));
            // E8M0 block scales = biased exponent of a random fp32 (raw bytes, NOT e4m3);
            // (bits >> 23) & 0xFF is the E8M0 byte, dequant'd as 2^(byte-127).
            unsigned char* scale = base + NOPE;
            for (int i = 0; i < SCALE; i++) {
                float s = std::exp2(scale_dis(gen));
                uint32_t bits; std::memcpy(&bits, &s, sizeof(bits));
                scale[i] = static_cast<unsigned char>((bits >> 23) & 0xFF);
            }
            for (int i = 0; i < ZERO; i++) base[NOPE + SCALE + i] = 0;
            auto* rope = reinterpret_cast<D_ROPE*>(base + ROPE_OFF);
            for (int i = 0; i < ROPE; i++) rope[i] = static_cast<D_ROPE>(dis(gen));
        }
    }
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
        kv_indices.insert(kv_indices.end(), pages.begin(), pages.begin() + nnz);
        assert(kv_indices.size() <= static_cast<size_t>(std::numeric_limits<int>::max()));
        kv_indptr[q + 1] = static_cast<int>(kv_indices.size());
    }

    assert(kv_indptr.front() == 0);
    assert(kv_indptr.back() == static_cast<int>(kv_indices.size()));
    for (int q = 0; q < N; ++q) {
        assert(kv_indptr[q] <= kv_indptr[q + 1]);
        for (int p = kv_indptr[q]; p < kv_indptr[q + 1]; ++p) {
            assert(kv_indices[p] >= 0 && kv_indices[p] < total_pages);
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
template<class Traits>
void benchmark_pa_kernel(const pa_kargs& kargs, dim3 grid, dim3 block,
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
    //   sparse attention -> 4 * H * nnz(indices) * D
    const double flops = (4.0 * kargs.H * indices_prefix_sum * kargs.D);
    const double tflops = flops / (avg_time * 1e-3) / 1e12;

    const size_t qo_bytes = 2ull * kargs.N * kargs.H * kargs.D * sizeof(typename Traits::D_ATTN);
    const size_t kv_bytes = (size_t)indices_prefix_sum * kargs.D * sizeof(typename Traits::D_ATTN);
    const double tbps = double(qo_bytes + kv_bytes) / (avg_time * 1e-3) / 1e12;

    printf("PA Prefill Kernel Performance: avg_time=%.3f ms, %.2f TFlops, %.2f TB/s\n",
           avg_time, tflops, tbps);
}

// Validate PA GPU results against CPU reference
template<typename DType>
bool validate_pa_results(const DType* ref, const DType* gpu,
                          int N, int H, int D, float threshold = 5e-2f) {
    bool all_valid = true;
    size_t total_errors = 0;
    const size_t total_elements = (size_t)N * H * D;

    for (int i = 0; i < N; i++) {
        for (int h = 0; h < H; h++) {
            const size_t offset = ((size_t)i * H + h) * D;
            for (int d = 0; d < D; d++) {
                const float ref_val = static_cast<float>(ref[offset + d]);
                const float gpu_val = static_cast<float>(gpu[offset + d]);
                const float diff = std::abs(gpu_val - ref_val);
                if (std::isnan(gpu_val) || std::isinf(gpu_val) || diff > threshold) {
                    total_errors++;
                    all_valid = false;
                    if (total_errors <= 32)  // cap log spam; full count reported below
                        printf("  mismatch [n=%d,h=%d,d=%d] ref=%.6f gpu=%.6f diff=%.6f\n",
                               i, h, d, ref_val, gpu_val, diff);
                }
            }
        }
    }
    
    if (all_valid) {
        printf("✓ Full validation passed (checked %zu elements)\n", total_elements);
    } else {
        printf("✗ Validation failed with %zu/%zu total errors\n",
               total_errors, total_elements);
    }
    
    return all_valid;
}

// Reconstruct one stored row into a dense float[D_HEAD_SIZE].
template<class PATraits>
inline void dequant_dsa_row(const typename PATraits::D_ATTN* row, float* out) {
    using DType  = typename PATraits::D_ATTN;
    constexpr bool is_fp8 = std::is_same_v<DType, fp8_t> || std::is_same_v<DType, bf8_t>;
    constexpr int  D_HEAD = PATraits::D_HEAD_SIZE;
    if constexpr (is_fp8) {
        using D_ROPE = typename PATraits::D_ROPE;
        constexpr int NOPE       = PATraits::D_NOPE_SIZE;
        constexpr int ROPE       = PATraits::D_ROPE_SIZE;
        constexpr int ROPE_BYTES = ROPE * (int)sizeof(D_ROPE);
        constexpr int ROW        = PATraits::D_TILE_SIZE;
        constexpr int ROPE_OFF   = ROW - ROPE_BYTES;   // RoPE bf16 sits at the end of the row
        const auto*   base  = reinterpret_cast<const unsigned char*>(row);
        const auto*   nope  = reinterpret_cast<const __hip_fp8_e4m3*>(base);
        const unsigned char* scale = base + NOPE;       // raw E8M0 bytes
        const auto*   rope  = reinterpret_cast<const D_ROPE*>(base + ROPE_OFF);
        for (int d = 0; d < NOPE; d++)
            out[d] = static_cast<float>(nope[d]) * std::ldexp(1.0f, int(scale[d / 32]) - 127);
        for (int j = 0; j < ROPE; j++)
            out[NOPE + j] = static_cast<float>(rope[j]);
    } else {
        for (int d = 0; d < D_HEAD; d++) out[d] = static_cast<float>(row[d]);
    }
}

// ─── CPU reference: Paged Attention (PA) ──────────────────────────
//
// Sparse scaled-dot-product attention over two CSR ranges:
//   prefix rows index UnifiedKV[total_pages, D]
//   extend rows index KV[total_tokens, D]
//   O[i,h,:] = softmax(Q[i,h,:] @ concat(prefix, extend)^T * softmax_scale) @ concat(prefix, extend)
//
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
    constexpr int D_HEAD = PATraits::D_HEAD_SIZE;   // logical head dim
    constexpr int ROW    = PATraits::D_TILE_SIZE;   // storage row stride (DType units)
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D_HEAD));

    const int stride_qo_n = H * ROW;
    const int stride_qo_h = ROW;
    const int stride_kv_page = ROW;
    const int o_stride_n = H * D_HEAD;
    const int o_stride_h = D_HEAD;

    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < N; i++) {
            const int prefix_begin = kv_indptr_prefix[i];
            const int prefix_end   = kv_indptr_prefix[i + 1];
            const int extend_begin = kv_indptr_extend[i];
            const int extend_end   = kv_indptr_extend[i + 1];
            const int num_prefix = prefix_end - prefix_begin;
            const int num_extend = extend_end - extend_begin;
            const int num_rows   = num_prefix + num_extend;

            O_t* o_row = O + (size_t)i * o_stride_n + h * o_stride_h;
            if (num_rows <= 0) {
                for (int d = 0; d < D_HEAD; d++) o_row[d] = static_cast<O_t>(0.0f);
                continue;
            }

            // Reconstruct Q and every K/V row to dense fp32 once (K == V; scores & PV share them).
            std::vector<float> q_dense(D_HEAD);
            dequant_dsa_row<PATraits>(Q + (size_t)i * stride_qo_n + h * stride_qo_h, q_dense.data());

            std::vector<float> kv_dense((size_t)num_rows * D_HEAD);
            for (int p = 0; p < num_prefix; p++) {
                const int kv_row = kv_indices_prefix[prefix_begin + p];
                dequant_dsa_row<PATraits>(UnifiedKV + (size_t)kv_row * stride_kv_page,
                                          kv_dense.data() + (size_t)p * D_HEAD);
            }
            for (int p = 0; p < num_extend; p++) {
                const int kv_row = kv_indices_extend[extend_begin + p];
                dequant_dsa_row<PATraits>(KV + (size_t)kv_row * stride_kv_page,
                                          kv_dense.data() + (size_t)(num_prefix + p) * D_HEAD);
            }

            // ---- Scores S[p] = (Q . K[p]) * softmax_scale ----
            std::vector<float> scores(num_rows);
            for (int p = 0; p < num_rows; p++) {
                const float* k = kv_dense.data() + (size_t)p * D_HEAD;
                float dot = 0.0f;
                for (int d = 0; d < D_HEAD; d++) dot += q_dense[d] * k[d];
                scores[p] = dot * softmax_scale;
            }

            // ---- Softmax with per-head sink in the denominator only ----
            float max_score = std::max(*std::max_element(scores.begin(), scores.end()), AttnSink[h]);
            float sum_exp = 0.0f;
            for (int p = 0; p < num_rows; p++) { scores[p] = std::exp(scores[p] - max_score); sum_exp += scores[p]; }
            sum_exp += std::exp(AttnSink[h] - max_score);
            std::vector<float> p_row(num_rows);
            for (int p = 0; p < num_rows; p++)
                p_row[p] = static_cast<float>(static_cast<bf16_t>(scores[p] / sum_exp));  // P rounded to bf16 (matches kernel)

            // ---- Output O[d] = sum_p P[p] * V[p][d]  (V == K) ----
            for (int d = 0; d < D_HEAD; d++) {
                float acc = 0.0f;
                for (int p = 0; p < num_rows; p++) acc += p_row[p] * kv_dense[(size_t)p * D_HEAD + d];
                o_row[d] = static_cast<O_t>(acc);
            }
        }
    }
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

    // D = storage row (fp8 packs scales+pad, so D_TILE_SIZE > D_HEAD); D_HEAD = output width.
    constexpr int D = PATraits::D_TILE_SIZE;
    constexpr int D_HEAD = PATraits::D_HEAD_SIZE;
    const size_t q_size = (size_t)N * H * D;
    const size_t unified_kv_size = (size_t)total_pages * D;
    const size_t kv_size = (size_t)total_tokens * D;
    const size_t o_size = (size_t)N * H * D_HEAD;

    // Allocate host memory
    auto host_q = std::make_unique<DType[]>(q_size);
    auto host_unified_kv = std::make_unique<DType[]>(unified_kv_size);
    auto host_kv = std::make_unique<DType[]>(kv_size);
    auto host_attn_sink = std::make_unique<float[]>(H);
    auto host_o_ref = std::make_unique<OType[]>(o_size);
    auto host_o_gpu = std::make_unique<OType[]>(o_size);
    std::vector<int> host_kv_indptr_prefix;
    std::vector<int> host_kv_indices_prefix;
    std::vector<int> host_kv_indptr_extend;
    std::vector<int> host_kv_indices_extend;

    // Initialize with random data
    if constexpr (is_fp8) {
        // Fill the packed DSA rows (NoPE fp8 + scales + zero-pad + RoPE bf16).
        init_fp8_dsa_tensor<PATraits>(host_q.get(), (size_t)N * H);
        init_fp8_dsa_tensor<PATraits>(host_unified_kv.get(), (size_t)total_pages);
        init_fp8_dsa_tensor<PATraits>(host_kv.get(), (size_t)total_tokens);
        rand_vector(host_attn_sink.get(), H, -2.f, 2.f);
    } else {
        rand_vector(host_q.get(), q_size, -2.f, 2.f);
        rand_vector(host_unified_kv.get(), unified_kv_size, -2.f, 2.f);
        rand_vector(host_kv.get(), kv_size, -2.f, 2.f);
        rand_vector(host_attn_sink.get(), H, -2.f, 2.f);
    }
    if (dense_kv) {
        init_dense_kv_indices(host_kv_indptr_prefix, host_kv_indices_prefix, N, total_pages);
        init_dense_kv_indices(host_kv_indptr_extend, host_kv_indices_extend, N, total_tokens);
    } else {
        init_sparse_kv_indices(host_kv_indptr_prefix,
                               host_kv_indices_prefix,
                               N,
                               total_pages,
                               PATraits::KV_TILE_SIZE,
                               1234);
        init_sparse_kv_indices(host_kv_indptr_extend,
                               host_kv_indices_extend,
                               N,
                               total_tokens,
                               PATraits::KV_TILE_SIZE,
                               5678);
    }
    const size_t total_kv_indices = host_kv_indices_prefix.size() + host_kv_indices_extend.size();
    assert(total_kv_indices <= static_cast<size_t>(std::numeric_limits<int>::max()));
    const int indices_prefix_sum = static_cast<int>(total_kv_indices);

    // Allocate device memory
    DType *dev_q, *dev_unified_kv, *dev_kv;
    OType *dev_o;
    float *dev_attn_sink;
    int *dev_kv_indptr_prefix, *dev_kv_indices_prefix, *dev_kv_indptr_extend, *dev_kv_indices_extend;
    const size_t kv_indices_prefix_alloc_size = std::max<size_t>(host_kv_indices_prefix.size(), 1);
    const size_t kv_indices_extend_alloc_size = std::max<size_t>(host_kv_indices_extend.size(), 1);
    CHECK_HIP(hipMalloc(&dev_q, q_size * sizeof(DType)));
    CHECK_HIP(hipMalloc(&dev_unified_kv, unified_kv_size * sizeof(DType)));
    CHECK_HIP(hipMalloc(&dev_kv, kv_size * sizeof(DType)));
    CHECK_HIP(hipMalloc(&dev_attn_sink, H * sizeof(float)));
    CHECK_HIP(hipMalloc(&dev_o, o_size * sizeof(OType)));
    CHECK_HIP(hipMemset(dev_o, 0, o_size * sizeof(OType)));
    CHECK_HIP(hipMalloc(&dev_kv_indptr_prefix, host_kv_indptr_prefix.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&dev_kv_indices_prefix, kv_indices_prefix_alloc_size * sizeof(int)));
    CHECK_HIP(hipMalloc(&dev_kv_indptr_extend, host_kv_indptr_extend.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&dev_kv_indices_extend, kv_indices_extend_alloc_size * sizeof(int)));

    CHECK_HIP(hipMemcpy(dev_q, host_q.get(), q_size * sizeof(DType), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_unified_kv, host_unified_kv.get(), unified_kv_size * sizeof(DType), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_kv, host_kv.get(), kv_size * sizeof(DType), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_attn_sink, host_attn_sink.get(), H * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_kv_indptr_prefix, host_kv_indptr_prefix.data(), host_kv_indptr_prefix.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_kv_indptr_extend, host_kv_indptr_extend.data(), host_kv_indptr_extend.size() * sizeof(int), hipMemcpyHostToDevice));
    if (!host_kv_indices_prefix.empty()) {
        CHECK_HIP(hipMemcpy(dev_kv_indices_prefix, host_kv_indices_prefix.data(), host_kv_indices_prefix.size() * sizeof(int), hipMemcpyHostToDevice));
    }
    if (!host_kv_indices_extend.empty()) {
        CHECK_HIP(hipMemcpy(dev_kv_indices_extend, host_kv_indices_extend.data(), host_kv_indices_extend.size() * sizeof(int), hipMemcpyHostToDevice));
    }

    // Setup kernel arguments
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
    kargs.stride_q_n = H * D;
    kargs.stride_q_h = D;
    kargs.stride_o_n = H * D_HEAD;
    kargs.stride_o_h = D_HEAD;
    kargs.stride_kv_page = D;
    kargs.softmax_scale = 1.0f / std::sqrt(static_cast<float>(PATraits::D_HEAD_SIZE));

    const int num_h_blocks = ceil_div(H, PATraits::Q_TILE_SIZE * PATraits::T_M);
    dim3 grid(N, num_h_blocks, 1);
    dim3 block(PATraits::BLOCK_SIZE);

    printf("PA kernel launch config: grid=(%d,%d,%d), block=%d (NUM_WARPS=%d), smem=%zu bytes (K/V tiles)\n",
           grid.x, grid.y, grid.z, (int)block.x, PATraits::NUM_WARPS, PATraits::smem_size_bytes());

    pa_launch(PATraits{}, kargs, grid, block);
    CHECK_HIP_KERNEL_LAUNCH();

    int rc = 0;
    if (verify) {
        printf("\nValidating GPU results against CPU reference...\n");
        CHECK_HIP(hipMemcpy(host_o_gpu.get(), dev_o, o_size * sizeof(OType), hipMemcpyDeviceToHost));
        pa_attention_ref<PATraits>(host_q.get(), host_unified_kv.get(), host_kv.get(), host_attn_sink.get(), host_o_ref.get(),
                                   host_kv_indptr_prefix.data(), host_kv_indices_prefix.data(),
                                   host_kv_indptr_extend.data(), host_kv_indices_extend.data(),
                                   N, H);

        bool all_valid = validate_pa_results<OType>(host_o_ref.get(), host_o_gpu.get(), N, H, D_HEAD);
        printf("\n[Overall] %s\n", all_valid ? "✓ GPU KERNEL VALID" : "✗ GPU KERNEL FAILED");
        if (!all_valid) rc = 1;
    }

    if (!rc) {
        printf("\n");
        benchmark_pa_kernel<PATraits>(kargs, grid, block, indices_prefix_sum);
        printf("\n");
    }

    // Cleanup
    CHECK_HIP(hipFree(dev_q));
    CHECK_HIP(hipFree(dev_unified_kv));
    CHECK_HIP(hipFree(dev_kv));
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

    // Parse command line arguments. Supports: -n 16384 and -n=16384.
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

    if (use_fp8) {
        return run_pa_case<pa_16mx1_16nx4_fp8_traits<16, 64, 640, 4, fp8_t, bf16_t, bf16_t>>(H, N, total_pages, total_tokens, verify, dense_kv);
    }
    // Dispatch by query-head count: h_q <= 32 favors the 16mx1_16nx4 layout,
    // otherwise the 16mx8_32nx1 layout. Both are correct for any H > 0.
    return H <= 32
        ? run_pa_case<pa_16mx1_16nx4_traits<16, 64, 512, 4, bf16_t, bf16_t>>(H, N, total_pages, total_tokens, verify, dense_kv)
        : run_pa_case<pa_16mx8_32nx1_traits<16, 32, 512, 8, bf16_t, bf16_t>>(H, N, total_pages, total_tokens, verify, dense_kv);
}
