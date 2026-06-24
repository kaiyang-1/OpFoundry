#include <hip/hip_fp8.h>
#include <opus/hip_minimal.hpp>
#include <algorithm>
#include <random>
#include <iostream>
#include <limits>
#include <numeric>
#include <memory>
#include <vector>
#include <bit>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <omp.h>

#include "defs.h"

template<class Traits>
__global__ void dsa_v32_decode_16mx8_32nx1_fp8_kernel(dsa_kargs kargs);
template<class Traits>
__global__ void get_mla_metadata_kernel(dsa_kargs kargs);
template<class Traits, int HEADS_PER_BLOCK = 8>
__global__ void mla_combine_kernel(dsa_kargs kargs);

static constexpr int DSA_V32_COMBINE_HEADS_PER_BLOCK = 8;

template<class Traits>
inline void dsa_v32_launch_pipeline(Traits, const dsa_kargs& kargs,
                                    dim3 grid_main, dim3 block_main, bool run_metadata = true) {
    if (run_metadata)
        get_mla_metadata_kernel<Traits><<<dim3(1), dim3(Traits::WARP_SIZE), (2 * kargs.B + 1) * (int)sizeof(int)>>>(kargs);
    dsa_v32_decode_16mx8_32nx1_fp8_kernel<Traits><<<grid_main, block_main>>>(kargs);
    const int n_head_blocks = ceil_div(kargs.H, DSA_V32_COMBINE_HEADS_PER_BLOCK);
    dim3 grid_combine(kargs.B, n_head_blocks, 1);
    dim3 block_combine(DSA_V32_COMBINE_HEADS_PER_BLOCK * Traits::WARP_SIZE);
    mla_combine_kernel<Traits><<<grid_combine, block_combine>>>(kargs);
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

template<class PATraits>
void init_fp8_dsa_split(typename PATraits::D_NOPE* nope_ptr,
                        uint8_t* scale_ptr,
                        typename PATraits::D_ROPE* rope_ptr, size_t rows) {
    using D_ROPE = typename PATraits::D_ROPE;
    constexpr int NOPE  = PATraits::D_NOPE_SIZE;
    constexpr int SCALE = PATraits::D_SCALE_SIZE;
    constexpr int ROPE  = PATraits::D_ROPE_SIZE;

    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
        std::uniform_real_distribution<float> scale_dis(-4.0f, 4.0f);
        #pragma omp for
        for (size_t r = 0; r < rows; r++) {
            auto* nope = reinterpret_cast<__hip_fp8_e4m3*>(nope_ptr) + r * NOPE;
            for (int i = 0; i < NOPE; i++) nope[i] = static_cast<__hip_fp8_e4m3>(dis(gen));
            uint8_t* scale = scale_ptr + r * SCALE;
            for (int i = 0; i < SCALE; i++) {
                float s = std::exp2(scale_dis(gen));
                const uint32_t bits = std::bit_cast<uint32_t>(s);
                scale[i] = static_cast<uint8_t>((bits >> 23) & 0xFF);
            }
            D_ROPE* rope = rope_ptr + r * ROPE;
            for (int i = 0; i < ROPE; i++) rope[i] = static_cast<D_ROPE>(dis(gen));
        }
    }
}

void init_sparse_kv_indices(std::vector<int>& kv_indptr,
                            std::vector<int>& kv_indices,
                            int B,
                            int total_pages,
                            int kv_tile_size,
                            uint32_t seed = 1234) {
    assert(B >= 0);
    assert(total_pages > 0);
    assert(kv_tile_size > 0);

    kv_indptr.assign(B + 1, 0);
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

    for (int q = 0; q < B; ++q) {
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
    for (int q = 0; q < B; ++q) {
        assert(kv_indptr[q] <= kv_indptr[q + 1]);
        for (int p = kv_indptr[q]; p < kv_indptr[q + 1]; ++p) {
            assert(kv_indices[p] >= 0 && kv_indices[p] < total_pages);
        }
    }
}

void init_dense_kv_indices(std::vector<int>& kv_indptr,
                           std::vector<int>& kv_indices,
                           int B,
                           int total_pages) {
    assert(B >= 0);
    assert(total_pages > 0);
    const size_t total_indices = static_cast<size_t>(B) * total_pages;
    assert(total_indices <= static_cast<size_t>(std::numeric_limits<int>::max()));

    kv_indptr.resize(B + 1);
    kv_indices.resize(total_indices);

    for (int q = 0; q <= B; ++q) {
        kv_indptr[q] = static_cast<int>(static_cast<size_t>(q) * total_pages);
    }
    for (int q = 0; q < B; ++q) {
        const size_t row_begin = static_cast<size_t>(q) * total_pages;
        for (int page = 0; page < total_pages; ++page) {
            kv_indices[row_begin + page] = page;
        }
    }
}

template<class Traits, class KArgs>
void benchmark_dsa_v32_kernel(const KArgs& kargs, dim3 grid, dim3 block,
                              int indices_prefix_sum, int warmup = 100, int iterations = 50) {

    get_mla_metadata_kernel<Traits><<<dim3(1), dim3(Traits::WARP_SIZE), (2 * kargs.B + 1) * (int)sizeof(int)>>>(kargs);
    CHECK_HIP_KERNEL_LAUNCH();
    CHECK_HIP(hipDeviceSynchronize());

    for (int i = 0; i < warmup; ++i) {
        dsa_v32_launch_pipeline(Traits{}, kargs, grid, block, false);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        dsa_v32_launch_pipeline(Traits{}, kargs, grid, block, false);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_time = 0;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const float avg_time = total_time / iterations;

    using D_NOPE = typename Traits::D_NOPE;
    using D_ROPE = typename Traits::D_ROPE;
    using D_OUT  = typename Traits::D_OUT;
    constexpr int D_QK = Traits::D_HEAD_SIZE;
    constexpr int D_V  = Traits::D_NOPE_SIZE;

    const double flops = 2.0 * kargs.H * indices_prefix_sum * (D_QK + D_V);
    const double tflops = flops / (avg_time * 1e-3) / 1e12;

    constexpr size_t row_bytes = Traits::D_NOPE_SIZE * sizeof(D_NOPE)
                               + Traits::D_SCALE_SIZE * sizeof(uint8_t)
                               + Traits::D_ROPE_SIZE * sizeof(D_ROPE);
    const size_t q_bytes  = (size_t)kargs.B * kargs.H * row_bytes;
    const size_t o_bytes  = (size_t)kargs.B * kargs.H * D_V * sizeof(D_OUT);
    const size_t kv_bytes = (size_t)indices_prefix_sum * row_bytes;
    const double tbps = double(q_bytes + o_bytes + kv_bytes) / (avg_time * 1e-3) / 1e12;

    printf("DSA v3.2 Decode Kernel Performance: avg_time=%.3f ms, %.2f TFlops, %.2f TB/s\n",
           avg_time, tflops, tbps);
}

template<typename DType>
bool validate_dsa_v32_results(const DType* ref, const DType* gpu,
                          int B, int H, int D,
                          float rtol = 1e-2f, float atol = 1e-2f,
                          float tol_err_ratio = 0.05f) {
    const size_t total_elements = (size_t)B * H * D;
    constexpr size_t printNum = 10;

    size_t total_errors = 0, printed = 0;
    bool any_nan = false;
    float max_abs_delta = 0.0f, ref_absmax = 0.0f;
    double sq_diff_sum = 0.0, ref_sq_sum = 0.0;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            const size_t offset = ((size_t)b * H + h) * D;
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
                        printf("  mismatch [b=%d,h=%d,d=%d] ref=%.6f gpu=%.6f delta=%.6f\n",
                               b, h, d, ref_val, gpu_val, delta);
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

template<class PATraits>
inline void dequant_dsa_row_fp8(const typename PATraits::D_NOPE* nrow,
                                const uint8_t* srow,
                                const typename PATraits::D_ROPE* rrow, float* out) {
    constexpr int NOPE = PATraits::D_NOPE_SIZE;
    constexpr int ROPE = PATraits::D_ROPE_SIZE;
    const auto* nope = reinterpret_cast<const __hip_fp8_e4m3*>(nrow);
    for (int d = 0; d < NOPE; d++)
        out[d] = static_cast<float>(nope[d]) * std::ldexp(1.0f, int(srow[d / 32]) - 127);
    for (int j = 0; j < ROPE; j++)
        out[NOPE + j] = static_cast<float>(rrow[j]);
}

template<class PATraits>
inline void dsa_v32_attention_compute(const float* q_dense, const float* kv_dense, int num_rows,
                                      typename PATraits::D_OUT* o_row) {
    using O_t = typename PATraits::D_OUT;
    constexpr int D_QK = PATraits::D_HEAD_SIZE;
    constexpr int D_V  = PATraits::D_NOPE_SIZE;
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D_QK));

    std::vector<float> scores(num_rows);
    for (int p = 0; p < num_rows; p++) {
        const float* k = kv_dense + (size_t)p * D_QK;
        float dot = 0.0f;
        for (int d = 0; d < D_QK; d++) dot += q_dense[d] * k[d];
        scores[p] = dot * softmax_scale;
    }
    float max_score = *std::max_element(scores.begin(), scores.end());
    float sum_exp = 0.0f;
    for (int p = 0; p < num_rows; p++) { scores[p] = std::exp(scores[p] - max_score); sum_exp += scores[p]; }
    for (int p = 0; p < num_rows; p++)
        scores[p] = static_cast<float>(static_cast<bf16_t>(scores[p] / sum_exp));
    for (int d = 0; d < D_V; d++) {
        float acc = 0.0f;
        for (int p = 0; p < num_rows; p++) acc += scores[p] * kv_dense[(size_t)p * D_QK + d];
        o_row[d] = static_cast<O_t>(acc);
    }
}

template<class PATraits>
void dsa_v32_attention_ref_fp8(
    const typename PATraits::D_NOPE* Q_nope, const uint8_t* Q_scale, const typename PATraits::D_ROPE* Q_rope,
    const typename PATraits::D_NOPE* KV_nope, const uint8_t* KV_scale, const typename PATraits::D_ROPE* KV_rope,
    typename PATraits::D_OUT* O,
    const int* kv_indptr, const int* kv_indices,
    int B, int H)
{
    using O_t = typename PATraits::D_OUT;
    constexpr int D_HEAD = PATraits::D_NOPE_SIZE;
    constexpr int D_QK   = PATraits::D_HEAD_SIZE;
    constexpr int NOPE   = PATraits::D_NOPE_SIZE;
    constexpr int SCALE  = PATraits::D_SCALE_SIZE;
    constexpr int ROPE   = PATraits::D_ROPE_SIZE;
    const int o_stride_n = H * D_HEAD;
    const int o_stride_h = D_HEAD;

    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < B; i++) {
            const int kv_begin = kv_indptr[i];
            const int num_rows = kv_indptr[i + 1] - kv_begin;

            O_t* o_row = O + (size_t)i * o_stride_n + h * o_stride_h;
            if (num_rows <= 0) {
                for (int d = 0; d < D_HEAD; d++) o_row[d] = static_cast<O_t>(0.0f);
                continue;
            }

            std::vector<float> q_dense(D_QK);
            const size_t q_row = (size_t)i * H + h;
            dequant_dsa_row_fp8<PATraits>(Q_nope + q_row * NOPE, Q_scale + q_row * SCALE, Q_rope + q_row * ROPE, q_dense.data());

            std::vector<float> kv_dense((size_t)num_rows * D_QK);
            for (int p = 0; p < num_rows; p++) {
                const int kv_row = kv_indices[kv_begin + p];
                dequant_dsa_row_fp8<PATraits>(KV_nope + (size_t)kv_row * NOPE, KV_scale + (size_t)kv_row * SCALE,
                                              KV_rope + (size_t)kv_row * ROPE,
                                              kv_dense.data() + (size_t)p * D_QK);
            }

            dsa_v32_attention_compute<PATraits>(q_dense.data(), kv_dense.data(), num_rows, o_row);
        }
    }
}

template<class PATraits>
int run_dsa_v32_case(int H, int B, int total_tokens,
                bool verify, bool dense_kv) {
    using OType = typename PATraits::D_OUT;
    printf("DSA v3.2 Decode Attention: H_Q=%d, B=%d, D_QK=%d, D_V=%d, NoPE=fp8, RoPE=bf16, total_tokens=%d\n",
           H, B, PATraits::D_HEAD_SIZE, PATraits::D_NOPE_SIZE, total_tokens);

    constexpr int D_HEAD = PATraits::D_NOPE_SIZE;
    const size_t o_size = (size_t)B * H * D_HEAD;

    auto host_o_ref = std::make_unique<OType[]>(o_size);
    auto host_o_gpu = std::make_unique<OType[]>(o_size);

    std::vector<int> host_kv_indptr, host_kv_indices;
    if (dense_kv) {
        init_dense_kv_indices(host_kv_indptr, host_kv_indices, B, total_tokens);
    } else {
        init_sparse_kv_indices(host_kv_indptr, host_kv_indices, B, total_tokens, PATraits::KV_TILE_SIZE, 5678);
    }
    const size_t total_kv_indices = host_kv_indices.size();
    assert(total_kv_indices <= static_cast<size_t>(std::numeric_limits<int>::max()));
    const int total_kv_count = static_cast<int>(total_kv_indices);

    OType *dev_o;
    int *dev_kv_indptr, *dev_kv_indices;
    const size_t kv_indices_alloc_size = std::max<size_t>(host_kv_indices.size(), 1);
    CHECK_HIP(hipMalloc(&dev_o, o_size * sizeof(OType)));
    CHECK_HIP(hipMalloc(&dev_kv_indptr, host_kv_indptr.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&dev_kv_indices, kv_indices_alloc_size * sizeof(int)));
    CHECK_HIP(hipMemcpy(dev_kv_indptr, host_kv_indptr.data(), host_kv_indptr.size() * sizeof(int), hipMemcpyHostToDevice));
    if (!host_kv_indices.empty())
        CHECK_HIP(hipMemcpy(dev_kv_indices, host_kv_indices.data(), host_kv_indices.size() * sizeof(int), hipMemcpyHostToDevice));

    const int num_h_blocks = ceil_div(H, PATraits::Q_TILE_SIZE * PATraits::T_M);
    const int num_parts = std::max(1, DSA_V32_NUM_CU / num_h_blocks);
    dim3 grid(num_parts, num_h_blocks, 1);
    dim3 block(PATraits::BLOCK_SIZE);
    printf("DSA v3.2 split-KV launch config: main grid=(%d,%d,%d) block=%d, num_parts=%d\n",
           grid.x, grid.y, grid.z, block.x, num_parts);

    const int total_splits = B + num_parts;
    DsaSchedMeta* dev_sched_meta; int* dev_num_splits;
    float *dev_o_accum, *dev_lse_accum;
    CHECK_HIP(hipMalloc(&dev_sched_meta, num_parts * sizeof(DsaSchedMeta)));
    CHECK_HIP(hipMalloc(&dev_num_splits, (B + 1) * sizeof(int)));
    CHECK_HIP(hipMalloc(&dev_o_accum, (size_t)total_splits * H * PATraits::D_NOPE_SIZE * sizeof(float)));
    CHECK_HIP(hipMalloc(&dev_lse_accum, (size_t)total_splits * H * sizeof(float)));

    int rc = 0;
    auto verify_and_bench = [&](const auto& kargs) {
        dsa_v32_launch_pipeline(PATraits{}, kargs, grid, block, true);
        CHECK_HIP_KERNEL_LAUNCH();
        if (verify) {
            printf("\nValidating GPU results against CPU reference...\n");
            CHECK_HIP(hipMemcpy(host_o_gpu.get(), dev_o, o_size * sizeof(OType), hipMemcpyDeviceToHost));
            bool all_valid = validate_dsa_v32_results<OType>(host_o_ref.get(), host_o_gpu.get(), B, H, D_HEAD);
            printf("\n[Overall] %s\n", all_valid ? "✓ GPU KERNEL VALID" : "✗ GPU KERNEL FAILED");
            if (!all_valid) rc = 1;
        }
        if (!rc) {
            printf("\n");
            benchmark_dsa_v32_kernel<PATraits>(kargs, grid, block, total_kv_count);
            printf("\n");
        }
    };

    using D_NOPE = typename PATraits::D_NOPE;
    using D_ROPE = typename PATraits::D_ROPE;
    constexpr int NOPE = PATraits::D_NOPE_SIZE;
    constexpr int SCALE = PATraits::D_SCALE_SIZE;
    constexpr int ROPE = PATraits::D_ROPE_SIZE;
    const size_t q_nope_size = (size_t)B * H * NOPE, q_rope_size = (size_t)B * H * ROPE;
    const size_t kv_nope_size = (size_t)total_tokens * NOPE, kv_rope_size = (size_t)total_tokens * ROPE;
    const size_t q_scale_size = (size_t)B * H * SCALE, kv_scale_size = (size_t)total_tokens * SCALE;

    auto host_q_nope = std::make_unique<D_NOPE[]>(q_nope_size);
    auto host_q_scale = std::make_unique<uint8_t[]>(q_scale_size);
    auto host_q_rope = std::make_unique<D_ROPE[]>(q_rope_size);
    auto host_kv_nope = std::make_unique<D_NOPE[]>(kv_nope_size);
    auto host_kv_scale = std::make_unique<uint8_t[]>(kv_scale_size);
    auto host_kv_rope = std::make_unique<D_ROPE[]>(kv_rope_size);
    init_fp8_dsa_split<PATraits>(host_q_nope.get(), host_q_scale.get(), host_q_rope.get(), (size_t)B * H);
    init_fp8_dsa_split<PATraits>(host_kv_nope.get(), host_kv_scale.get(), host_kv_rope.get(), (size_t)total_tokens);

    D_NOPE *dev_q_nope, *dev_kv_nope;
    D_ROPE *dev_q_rope, *dev_kv_rope;
    uint8_t *dev_q_scale, *dev_kv_scale;
    CHECK_HIP(hipMalloc(&dev_q_nope, q_nope_size * sizeof(D_NOPE)));
    CHECK_HIP(hipMalloc(&dev_q_scale, q_scale_size * sizeof(uint8_t)));
    CHECK_HIP(hipMalloc(&dev_q_rope, q_rope_size * sizeof(D_ROPE)));
    CHECK_HIP(hipMalloc(&dev_kv_nope, kv_nope_size * sizeof(D_NOPE)));
    CHECK_HIP(hipMalloc(&dev_kv_scale, kv_scale_size * sizeof(uint8_t)));
    CHECK_HIP(hipMalloc(&dev_kv_rope, kv_rope_size * sizeof(D_ROPE)));
    CHECK_HIP(hipMemcpy(dev_q_nope, host_q_nope.get(), q_nope_size * sizeof(D_NOPE), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_q_scale, host_q_scale.get(), q_scale_size * sizeof(uint8_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_q_rope, host_q_rope.get(), q_rope_size * sizeof(D_ROPE), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_kv_nope, host_kv_nope.get(), kv_nope_size * sizeof(D_NOPE), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_kv_scale, host_kv_scale.get(), kv_scale_size * sizeof(uint8_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_kv_rope, host_kv_rope.get(), kv_rope_size * sizeof(D_ROPE), hipMemcpyHostToDevice));

    if (verify)
        dsa_v32_attention_ref_fp8<PATraits>(host_q_nope.get(), host_q_scale.get(), host_q_rope.get(),
                                            host_kv_nope.get(), host_kv_scale.get(), host_kv_rope.get(),
                                            host_o_ref.get(),
                                            host_kv_indptr.data(), host_kv_indices.data(), B, H);

    dsa_kargs kargs{};
    kargs.q_nope_ptr = dev_q_nope;
    kargs.q_scale_ptr = dev_q_scale;
    kargs.q_rope_ptr = dev_q_rope;
    kargs.kv_nope_ptr = dev_kv_nope;
    kargs.kv_scale_ptr = dev_kv_scale;
    kargs.kv_rope_ptr = dev_kv_rope;
    kargs.out_ptr = dev_o;
    kargs.kv_indptr = dev_kv_indptr;
    kargs.kv_indices = dev_kv_indices;
    kargs.sched_meta = dev_sched_meta;
    kargs.num_splits = dev_num_splits;
    kargs.o_accum = dev_o_accum;
    kargs.lse_accum = dev_lse_accum;
    kargs.num_parts = num_parts;
    kargs.B = B;
    kargs.H = H;
    kargs.total_tokens = total_tokens;
    kargs.stride_q_nope_b = H * NOPE;
    kargs.stride_q_nope_h = NOPE;
    kargs.stride_q_scale_b = H * SCALE;
    kargs.stride_q_scale_h = SCALE;
    kargs.stride_q_rope_b = H * ROPE;
    kargs.stride_q_rope_h = ROPE;
    kargs.stride_o_b = H * D_HEAD;
    kargs.stride_o_h = D_HEAD;
    kargs.stride_kv_nope_page = NOPE;
    kargs.stride_kv_scale_page = SCALE;
    kargs.stride_kv_rope_page = ROPE;
    kargs.softmax_scale = 1.0f / std::sqrt(static_cast<float>(PATraits::D_HEAD_SIZE));

    verify_and_bench(kargs);

    CHECK_HIP(hipFree(dev_q_nope));   CHECK_HIP(hipFree(dev_q_rope));   CHECK_HIP(hipFree(dev_q_scale));
    CHECK_HIP(hipFree(dev_kv_nope));  CHECK_HIP(hipFree(dev_kv_rope));  CHECK_HIP(hipFree(dev_kv_scale));

    CHECK_HIP(hipFree(dev_sched_meta));
    CHECK_HIP(hipFree(dev_num_splits));
    CHECK_HIP(hipFree(dev_o_accum));
    CHECK_HIP(hipFree(dev_lse_accum));

    CHECK_HIP(hipFree(dev_o));
    CHECK_HIP(hipFree(dev_kv_indptr));
    CHECK_HIP(hipFree(dev_kv_indices));

    return rc;
}

int main(int argc, char** argv) {
    int H = 128;
    int B = 128;
    int total_tokens = -1;

    bool verify = false;
    bool dense_kv = false;
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
        if (std::strcmp(arg, "--dense") == 0) { dense_kv = true; continue; }
        auto try_parse = [&](int& target, const char* flag) {
            if ((val = parse_val(arg, flag))) {
                if (val == reinterpret_cast<const char*>(1)) { if (i + 1 < argc) target = std::atoi(argv[++i]); }
                else target = std::atoi(val);
                return true;
            }
            return false;
        };
        if (try_parse(H, "-h_q")) continue;
        if (try_parse(B, "-b")) continue;
        if (try_parse(total_tokens, "-total_tokens")) continue;
    }
    if (total_tokens < 0) {
        total_tokens = B;
    }

    if (H <= 0 || B <= 0 || total_tokens <= 0) {
        std::cerr << "Invalid parameters. H_Q,B,total_tokens must be positive.\n";
        return 1;
    }

    return run_dsa_v32_case<dsa_v32_16mx8_32nx1_fp8_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>>(H, B, total_tokens, verify, dense_kv);
}
