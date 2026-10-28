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

#include "mla_v41_defs.h"
#include "mla_v41_parallel.h"

template<class Traits>
__global__ void opus_mla_v41_prefill_a8w8_16mx4_64nx1_kernel(typename Traits::kargs_type kargs);
template<class Traits>
__global__ void opus_mla_v41_prefill_a8w8_16mx1_16nx4_kernel(typename Traits::kargs_type kargs);

template<int Q, int KV, int NW, class DA, class DP, class DO>
inline void mla_v41_prefill_launch(opus_mla_v41_prefill_a8w8_16mx4_64nx1_traits<Q, KV, NW, DA, DP, DO>,
                                   const opus_mla_v41_prefill_kargs& kargs, dim3 grid, dim3 block) {
    opus_mla_v41_prefill_a8w8_16mx4_64nx1_kernel<opus_mla_v41_prefill_a8w8_16mx4_64nx1_traits<Q, KV, NW, DA, DP, DO>><<<grid, block>>>(kargs);
}
template<int Q, int KV, int NW, class DA, class DP, class DO>
inline void mla_v41_prefill_launch(opus_mla_v41_prefill_a8w8_16mx1_16nx4_traits<Q, KV, NW, DA, DP, DO>,
                                   const opus_mla_v41_prefill_kargs& kargs, dim3 grid, dim3 block) {
    opus_mla_v41_prefill_a8w8_16mx1_16nx4_kernel<opus_mla_v41_prefill_a8w8_16mx1_16nx4_traits<Q, KV, NW, DA, DP, DO>><<<grid, block>>>(kargs);
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

inline uint8_t float_to_fp8_e4m3(float f) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, f);
    const uint8_t  sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
    const uint32_t mag  = bits & 0x7FFFFFFFu;

    if (mag >= 0x7F800000u) return static_cast<uint8_t>(sign | 0x7Fu);
    if (mag >= 0x43E00000u) return static_cast<uint8_t>(sign | 0x7Eu);

    const int exp = static_cast<int>(mag >> 23) - 127;
    uint8_t payload;
    uint32_t dropped, halfway;
    if (exp >= -6) {
        const uint32_t mant = mag & 0x7FFFFFu;
        payload = static_cast<uint8_t>(((exp + 7) << 3) | (mant >> 20));
        dropped = mant & 0xFFFFFu;
        halfway = 0x80000u;
    } else {
        const int shift = -exp - 6;
        if (shift > 11) return sign;
        const uint32_t mant  = (mag & 0x7FFFFFu) | 0x800000u;
        const uint32_t width = 20 + shift;
        payload = static_cast<uint8_t>(mant >> width);
        dropped = mant & ((1u << width) - 1);
        halfway = 1u << (width - 1);
    }
    if (dropped > halfway || (dropped == halfway && (payload & 1))) payload++;
    return static_cast<uint8_t>(sign | payload);
}

inline float fp8_e4m3_to_float(uint8_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x80u) << 24;
    const uint32_t exp  = (bits >> 3) & 0x0Fu;
    const uint32_t mant = bits & 0x07u;

    if (exp == 0x0Fu && mant == 0x07u) return __builtin_bit_cast(float, sign | 0x7FC00000u);
    if (exp == 0) {
        const float m = static_cast<float>(mant) * 0x1p-9f;
        return (bits & 0x80u) ? -m : m;
    }
    return __builtin_bit_cast(float, sign | ((exp + 120) << 23) | (mant << 20));
}

// Fixed grain keeps the generated data independent of MLA_V41_NUM_THREADS.
static constexpr size_t MLA_V41_INIT_GRAIN = 65536;
static constexpr uint64_t MLA_V41_SEED = 2026;

inline std::mt19937 mla_v41_rng(uint64_t seed, uint64_t index) {
    uint64_t x = seed + 0x9E3779B97F4A7C15ull * (index + 1);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return std::mt19937(static_cast<uint32_t>(x ^ (x >> 31)));
}

template<typename T>
void rand_vector(T* ptr, size_t size, uint64_t seed) {
    mla_v41::parallel_for((size + MLA_V41_INIT_GRAIN - 1) / MLA_V41_INIT_GRAIN, [&](size_t c) {
        const size_t begin = c * MLA_V41_INIT_GRAIN;
        const size_t end = std::min(begin + MLA_V41_INIT_GRAIN, size);
        std::mt19937 gen = mla_v41_rng(seed, c);
        std::normal_distribution<float> dis(0.0f, 1.0f);
        for (size_t i = begin; i < end; i++) {
            ptr[i] = static_cast<T>(dis(gen));
        }
    });
}

template<class Traits>
void init_mxfp8_rows(uint8_t* data_ptr, uint8_t* scale_ptr, size_t rows, uint64_t seed) {
    constexpr int D     = Traits::D_HEAD_SIZE;
    constexpr int SCALE = Traits::D_SCALE_SIZE;
    constexpr int BLOCK = Traits::D_SCALE_BLOCK;

    mla_v41::parallel_for((rows + MLA_V41_INIT_GRAIN - 1) / MLA_V41_INIT_GRAIN, [&](size_t c) {
        const size_t begin = c * MLA_V41_INIT_GRAIN;
        const size_t end = std::min(begin + MLA_V41_INIT_GRAIN, rows);
        std::mt19937 gen = mla_v41_rng(seed, c);
        std::normal_distribution<float> dis(0.0f, 1.0f);
        std::uniform_int_distribution<int> scale_exp_dis(0, 3);
        for (size_t r = begin; r < end; r++) {
            uint8_t* data  = data_ptr + r * D;
            uint8_t* scale = scale_ptr + r * SCALE;
            for (int b = 0; b < SCALE; b++) {
                const int e = scale_exp_dis(gen);
                scale[b] = static_cast<uint8_t>(e + 127);
                const float inv_scale = std::ldexp(1.0f, -e);
                for (int i = 0; i < BLOCK; i++)
                    data[b * BLOCK + i] = float_to_fp8_e4m3(dis(gen) * inv_scale);
            }
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
    const size_t grain = std::max<size_t>(1, ceil_div(N, static_cast<int>(mla_v41::num_threads())));
    mla_v41::parallel_chunks(static_cast<size_t>(N), grain, [&](size_t begin, size_t end, unsigned) {
        std::vector<int> perm(static_cast<size_t>(pool_rows));
        std::iota(perm.begin(), perm.end(), 0);
        for (size_t q = begin; q < end; ++q) {
            int* row = kv_indices.data() + kv_indptr[q];
            const int nnz = kv_indptr[q + 1] - kv_indptr[q];
            const int range = candidates(static_cast<int>(q));
            std::mt19937 gen = mla_v41_rng(seed, static_cast<uint64_t>(q) + (1ull << 32));
            for (int i = 0; i < nnz; ++i) {
                const int j = i + static_cast<int>(gen() % static_cast<uint32_t>(range - i));
                std::swap(perm[i], perm[j]);
                row[i] = perm[i];
            }
            for (int i = 0; i < nnz; ++i) perm[i] = i;
            for (int i = 0; i < nnz; ++i) perm[row[i]] = row[i];
        }
    });
}

// 0xFF is NaN as both e4m3 and E8M0, so a gather of an unselected row poisons O.
template<class Traits>
void poison_kv_rows(uint8_t* kv, uint8_t* kv_scale, int pool_rows, const std::vector<int>& kv_indices) {
    constexpr int D     = Traits::D_HEAD_SIZE;
    constexpr int SCALE = Traits::D_SCALE_SIZE;
    std::vector<uint8_t> selected(static_cast<size_t>(pool_rows), 0);
    for (int idx : kv_indices) selected[static_cast<size_t>(idx)] = 1;
    for (size_t r = 0; r < selected.size(); ++r) {
        if (selected[r]) continue;
        std::memset(kv + r * D, 0xFF, D);
        std::memset(kv_scale + r * SCALE, 0xFF, SCALE);
    }
}

template<class Traits>
void benchmark_mla_v41_kernel(const opus_mla_v41_prefill_kargs& kargs, dim3 grid, dim3 block,
                              int total_kv_rows, int warmup = 100, int iterations = 50) {
    for (int i = 0; i < warmup; ++i) {
        mla_v41_prefill_launch(Traits{}, kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        mla_v41_prefill_launch(Traits{}, kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_time = 0;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const float avg_time = total_time / iterations;

    constexpr int D = Traits::D_HEAD_SIZE;
    constexpr size_t row_bytes = D * sizeof(typename Traits::D_ATTN) + Traits::D_SCALE_SIZE;

    const double flops = 4.0 * kargs.H * total_kv_rows * D;
    const double tflops = flops / (avg_time * 1e-3) / 1e12;

    const size_t q_bytes  = (size_t)kargs.N * kargs.H * row_bytes;
    const size_t o_bytes  = (size_t)kargs.N * kargs.H * D * sizeof(typename Traits::D_OUT);
    const size_t kv_bytes = (size_t)total_kv_rows * row_bytes;
    const double tbps = double(q_bytes + o_bytes + kv_bytes) / (avg_time * 1e-3) / 1e12;

    printf("MLA-v4.1 Prefill Kernel Performance: avg_time=%.3f ms, %.2f TFlops, %.2f TB/s\n",
           avg_time, tflops, tbps);
}

template<typename DType>
bool validate_mla_v41_results(const DType* ref, const DType* gpu,
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
inline void dequant_mxfp8_row(const uint8_t* row, const uint8_t* scale, float* out) {
    constexpr int D = Traits::D_HEAD_SIZE;
    for (int d = 0; d < D; d++)
        out[d] = fp8_e4m3_to_float(row[d]) * std::ldexp(1.0f, int(scale[d / Traits::D_SCALE_BLOCK]) - 127);
}

template<class Traits>
inline void mla_v41_attention_compute(const float* q_dense, const float* kv_dense, int num_rows,
                                      float sink, typename Traits::D_OUT* o_row) {
    using O_t = typename Traits::D_OUT;
    constexpr int D = Traits::D_HEAD_SIZE;
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> scores(num_rows);
    for (int p = 0; p < num_rows; p++) {
        const float* k = kv_dense + (size_t)p * D;
        float dot = 0.0f;
        for (int d = 0; d < D; d++) dot += q_dense[d] * k[d];
        scores[p] = dot * softmax_scale;
    }
    float max_score = std::max(*std::max_element(scores.begin(), scores.end()), sink);
    float sum_exp = 0.0f;
    for (int p = 0; p < num_rows; p++) { scores[p] = std::exp(scores[p] - max_score); sum_exp += scores[p]; }
    sum_exp += std::exp(sink - max_score);
    for (int p = 0; p < num_rows; p++)
        scores[p] = static_cast<float>(static_cast<typename Traits::D_PV>(scores[p] / sum_exp));
    for (int d = 0; d < D; d++) {
        float acc = 0.0f;
        for (int p = 0; p < num_rows; p++) acc += scores[p] * kv_dense[(size_t)p * D + d];
        o_row[d] = static_cast<O_t>(acc);
    }
}

template<class Traits>
void mla_v41_attention_ref(
    const uint8_t* Q, const uint8_t* Q_scale,
    const uint8_t* UnifiedKV, const uint8_t* UnifiedKV_scale,
    const uint8_t* KV, const uint8_t* KV_scale,
    const float* AttnSink,
    typename Traits::D_OUT* O,
    const int* kv_indptr_prefix,
    const int* kv_indices_prefix,
    const int* kv_indptr_extend,
    const int* kv_indices_extend,
    int N, int H)
{
    using O_t = typename Traits::D_OUT;
    constexpr int D     = Traits::D_HEAD_SIZE;
    constexpr int SCALE = Traits::D_SCALE_SIZE;

    mla_v41::parallel_for((size_t)H * N, [&](size_t idx) {
        const int h = static_cast<int>(idx / N);
        const int i = static_cast<int>(idx % N);
        const int prefix_begin = kv_indptr_prefix[i];
        const int extend_begin = kv_indptr_extend[i];
        const int num_prefix = kv_indptr_prefix[i + 1] - prefix_begin;
        const int num_extend = kv_indptr_extend[i + 1] - extend_begin;
        const int num_rows   = num_prefix + num_extend;

        O_t* o_row = O + ((size_t)i * H + h) * D;
        if (num_rows <= 0) {
            for (int d = 0; d < D; d++) o_row[d] = static_cast<O_t>(0.0f);
            return;
        }

        std::vector<float> q_dense(D);
        const size_t q_row = (size_t)i * H + h;
        dequant_mxfp8_row<Traits>(Q + q_row * D, Q_scale + q_row * SCALE, q_dense.data());

        std::vector<float> kv_dense((size_t)num_rows * D);
        for (int p = 0; p < num_prefix; p++) {
            const size_t kv_row = kv_indices_prefix[prefix_begin + p];
            dequant_mxfp8_row<Traits>(UnifiedKV + kv_row * D, UnifiedKV_scale + kv_row * SCALE,
                                      kv_dense.data() + (size_t)p * D);
        }
        for (int p = 0; p < num_extend; p++) {
            const size_t kv_row = kv_indices_extend[extend_begin + p];
            dequant_mxfp8_row<Traits>(KV + kv_row * D, KV_scale + kv_row * SCALE,
                                      kv_dense.data() + (size_t)(num_prefix + p) * D);
        }

        const float sink = AttnSink ? AttnSink[h] : -std::numeric_limits<float>::infinity();
        mla_v41_attention_compute<Traits>(q_dense.data(), kv_dense.data(), num_rows, sink, o_row);
    });
}

template<class Traits>
int run_mla_v41_prefill_case(int H, int N, int total_pages, int total_tokens,
                             int topk, bool verify, bool with_sink) {
    using OType = typename Traits::D_OUT;
    constexpr int D     = Traits::D_HEAD_SIZE;
    constexpr int SCALE = Traits::D_SCALE_SIZE;
    printf("MLA-v4.1 Prefill Attention: H_Q=%d, N=%d, D=%d, mxfp8, total_pages=%d, total_tokens=%d, topk=%d%s\n",
           H, N, D, total_pages, total_tokens, topk, with_sink ? "" : ", no sink");

    const size_t q_rows = (size_t)N * H;
    const size_t o_size = q_rows * D;

    auto host_attn_sink        = std::make_unique<float[]>(H);
    auto host_q                = std::make_unique<uint8_t[]>(q_rows * D);
    auto host_q_scale          = std::make_unique<uint8_t[]>(q_rows * SCALE);
    auto host_unified_kv       = std::make_unique<uint8_t[]>((size_t)total_pages * D);
    auto host_unified_kv_scale = std::make_unique<uint8_t[]>((size_t)total_pages * SCALE);
    auto host_kv               = std::make_unique<uint8_t[]>((size_t)total_tokens * D);
    auto host_kv_scale         = std::make_unique<uint8_t[]>((size_t)total_tokens * SCALE);
    auto host_o_ref            = std::make_unique<OType[]>(o_size);
    auto host_o_gpu            = std::make_unique<OType[]>(o_size);
    const uint64_t seed_sink = MLA_V41_SEED + 1, seed_q = MLA_V41_SEED + 2, seed_ukv = MLA_V41_SEED + 3,
                   seed_kv = MLA_V41_SEED + 4, seed_idx_prefix = MLA_V41_SEED + 5,
                   seed_idx_extend = MLA_V41_SEED + 6;
    rand_vector(host_attn_sink.get(), H, seed_sink);
    init_mxfp8_rows<Traits>(host_q.get(), host_q_scale.get(), q_rows, seed_q);
    init_mxfp8_rows<Traits>(host_unified_kv.get(), host_unified_kv_scale.get(), (size_t)total_pages, seed_ukv);
    init_mxfp8_rows<Traits>(host_kv.get(), host_kv_scale.get(), (size_t)total_tokens, seed_kv);

    std::vector<int> host_kv_indptr_prefix, host_kv_indices_prefix;
    std::vector<int> host_kv_indptr_extend, host_kv_indices_extend;
    init_kv_page_table(host_kv_indptr_prefix, host_kv_indices_prefix, N, total_pages, topk,
                       /*causal=*/false, seed_idx_prefix);
    init_kv_page_table(host_kv_indptr_extend, host_kv_indices_extend, N, total_tokens, topk,
                       /*causal=*/true, seed_idx_extend);
    const size_t total_kv_indices = host_kv_indices_prefix.size() + host_kv_indices_extend.size();
    assert(total_kv_indices <= static_cast<size_t>(std::numeric_limits<int>::max()));
    const int total_kv_rows = static_cast<int>(total_kv_indices);

    poison_kv_rows<Traits>(host_unified_kv.get(), host_unified_kv_scale.get(), total_pages, host_kv_indices_prefix);
    poison_kv_rows<Traits>(host_kv.get(), host_kv_scale.get(), total_tokens, host_kv_indices_extend);

    if (verify)
        mla_v41_attention_ref<Traits>(host_q.get(), host_q_scale.get(),
                                      host_unified_kv.get(), host_unified_kv_scale.get(),
                                      host_kv.get(), host_kv_scale.get(),
                                      with_sink ? host_attn_sink.get() : nullptr,
                                      host_o_ref.get(),
                                      host_kv_indptr_prefix.data(), host_kv_indices_prefix.data(),
                                      host_kv_indptr_extend.data(), host_kv_indices_extend.data(), N, H);

    dev_buf<float> dev_attn_sink(H);
    dev_buf<uint8_t> dev_q(q_rows * D);
    dev_buf<uint8_t> dev_q_scale(q_rows * SCALE);
    dev_buf<uint8_t> dev_unified_kv((size_t)total_pages * D);
    dev_buf<uint8_t> dev_unified_kv_scale((size_t)total_pages * SCALE);
    dev_buf<uint8_t> dev_kv((size_t)total_tokens * D);
    dev_buf<uint8_t> dev_kv_scale((size_t)total_tokens * SCALE);
    dev_buf<OType> dev_o(o_size);
    dev_buf<int> dev_kv_indptr_prefix(host_kv_indptr_prefix.size());
    dev_buf<int> dev_kv_indices_prefix(host_kv_indices_prefix.size());
    dev_buf<int> dev_kv_indptr_extend(host_kv_indptr_extend.size());
    dev_buf<int> dev_kv_indices_extend(host_kv_indices_extend.size());
    dev_attn_sink.upload(host_attn_sink.get(), H);
    dev_q.upload(host_q.get(), q_rows * D);
    dev_q_scale.upload(host_q_scale.get(), q_rows * SCALE);
    dev_unified_kv.upload(host_unified_kv.get(), (size_t)total_pages * D);
    dev_unified_kv_scale.upload(host_unified_kv_scale.get(), (size_t)total_pages * SCALE);
    dev_kv.upload(host_kv.get(), (size_t)total_tokens * D);
    dev_kv_scale.upload(host_kv_scale.get(), (size_t)total_tokens * SCALE);
    dev_o.zero(o_size);
    dev_kv_indptr_prefix.upload(host_kv_indptr_prefix.data(), host_kv_indptr_prefix.size());
    dev_kv_indices_prefix.upload(host_kv_indices_prefix.data(), host_kv_indices_prefix.size());
    dev_kv_indptr_extend.upload(host_kv_indptr_extend.data(), host_kv_indptr_extend.size());
    dev_kv_indices_extend.upload(host_kv_indices_extend.data(), host_kv_indices_extend.size());

    opus_mla_v41_prefill_kargs kargs{};
    kargs.q_ptr = dev_q;
    kargs.q_scale_ptr = dev_q_scale;
    kargs.unified_kv_ptr = dev_unified_kv;
    kargs.unified_kv_scale_ptr = dev_unified_kv_scale;
    kargs.kv_ptr = dev_kv;
    kargs.kv_scale_ptr = dev_kv_scale;
    kargs.attn_sink_ptr = with_sink ? dev_attn_sink.ptr : nullptr;
    kargs.out_ptr = dev_o;
    kargs.kv_indptr_prefix = dev_kv_indptr_prefix;
    kargs.kv_indices_prefix = dev_kv_indices_prefix;
    kargs.kv_indptr_extend = dev_kv_indptr_extend;
    kargs.kv_indices_extend = dev_kv_indices_extend;
    kargs.N = N;
    kargs.H = H;
    kargs.total_pages = total_pages;
    kargs.total_tokens = total_tokens;
    kargs.stride_q_n = H * D;
    kargs.stride_q_h = D;
    kargs.stride_q_scale_n = H * SCALE;
    kargs.stride_q_scale_h = SCALE;
    kargs.stride_o_n = H * D;
    kargs.stride_o_h = D;
    kargs.stride_kv_page = D;
    kargs.stride_kv_scale_page = SCALE;
    kargs.softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));

    const int num_h_blocks = ceil_div(H, Traits::Q_TILE_SIZE * Traits::T_M);
    dim3 grid(N, num_h_blocks, 1);
    dim3 block(Traits::BLOCK_SIZE);
    printf("MLA-v4.1 kernel launch config: grid=(%d,%d,%d), block=%d\n",
           grid.x, grid.y, grid.z, (int)block.x);

    mla_v41_prefill_launch(Traits{}, kargs, grid, block);
    CHECK_HIP_KERNEL_LAUNCH();
    if (verify) {
        printf("\nValidating GPU results against CPU reference...\n");
        dev_o.download(host_o_gpu.get(), o_size);
        const bool all_valid = validate_mla_v41_results<OType>(host_o_ref.get(), host_o_gpu.get(), N, H, D);
        printf("\n[Overall] %s\n", all_valid ? "✓ GPU KERNEL VALID" : "✗ GPU KERNEL FAILED");
        if (!all_valid) return 1;
    }
    printf("\n");
    benchmark_mla_v41_kernel<Traits>(kargs, grid, block, total_kv_rows);
    printf("\n");
    return 0;
}

static constexpr int MLA_V41_DEFAULT_TOTAL_PAGES = 65536;

int main(int argc, char** argv) {
    int H = 128;
    int N = 4096;
    int topk = 1024;
    int total_pages = -1;
    int total_tokens = -1;
    bool verify = false;
    bool with_sink = true;

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
        if (std::strcmp(arg, "-no_sink") == 0) { with_sink = false; continue; }
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
        total_pages = topk > 0 ? std::max({N, topk, MLA_V41_DEFAULT_TOTAL_PAGES}) : N;
    if (total_tokens < 0) total_tokens = N;
    if (topk <= 0) topk = std::max(total_pages, total_tokens);

    if (H <= 0 || N <= 0 || total_pages < 0 || total_tokens < 0) {
        std::cerr << "Invalid parameters. H_Q,N must be positive and "
                     "total_pages,total_tokens non-negative.\n";
        return 1;
    }

    if (H <= 16)
        return run_mla_v41_prefill_case<opus_mla_v41_prefill_a8w8_16mx1_16nx4_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>>(
            H, N, total_pages, total_tokens, topk, verify, with_sink);
    return run_mla_v41_prefill_case<opus_mla_v41_prefill_a8w8_16mx4_64nx1_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>>(
        H, N, total_pages, total_tokens, topk, verify, with_sink);
}
