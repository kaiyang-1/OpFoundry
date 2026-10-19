#pragma once

#include <cstdint>

using bf16_t = __bf16;
using fp8_t  = _BitInt(8);

struct opus_mla_v41_prefill_kargs {
    const void* __restrict__ q_ptr;                // [N, H, D] e4m3
    const void* __restrict__ q_scale_ptr;          // [N, H, D / 32] e8m0
    const void* __restrict__ unified_kv_ptr;       // [total_pages, D], prefix source
    const void* __restrict__ unified_kv_scale_ptr; // [total_pages, D / 32]
    const void* __restrict__ kv_ptr;               // [total_tokens, D], extend source
    const void* __restrict__ kv_scale_ptr;         // [total_tokens, D / 32]
    const void* __restrict__ attn_sink_ptr;        // [H] fp32, may be nullptr
    void* __restrict__ out_ptr;                    // [N, H, D] bf16
    void* __restrict__ lse_ptr;                    // [N, H] fp32, may be nullptr
    const int* __restrict__ kv_indptr_prefix;      // [N + 1]
    const int* __restrict__ kv_indices_prefix;     // [nnz_prefix]
    const int* __restrict__ kv_indptr_extend;      // [N + 1]
    const int* __restrict__ kv_indices_extend;     // [nnz_extend]
    int N;
    int H;
    int total_pages;
    int total_tokens;
    int stride_q_n;
    int stride_q_h;
    int stride_q_scale_n;
    int stride_q_scale_h;
    int stride_o_n;
    int stride_o_h;
    int stride_lse_n;
    int stride_kv_page;
    int stride_kv_scale_page;
    float softmax_scale;
};

__host__ __device__ constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

#if defined(MLA_V41_ARCH_GFX950)
#include "gfx950/mla_v41_traits.h"
#else
#error "no MLA_V41_ARCH_* defined; build through the Makefile"
#endif
