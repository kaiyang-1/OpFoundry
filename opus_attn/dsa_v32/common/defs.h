#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

using bf16_t = __bf16;
using fp8_t  = _BitInt(8);
using bf8_t  = unsigned _BitInt(8);

inline fp8_t float_to_fp8_e4m3(float f) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, f);
    const uint8_t  sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
    const uint32_t mag  = bits & 0x7FFFFFFFu;

    if (mag >= 0x7F800000u) return __builtin_bit_cast(fp8_t, static_cast<uint8_t>(sign | 0x7Fu));
    if (mag >= 0x43E00000u) return __builtin_bit_cast(fp8_t, static_cast<uint8_t>(sign | 0x7Eu));

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
        if (shift > 11) return __builtin_bit_cast(fp8_t, sign);
        const uint32_t mant  = (mag & 0x7FFFFFu) | 0x800000u;
        const uint32_t width = 20 + shift;
        payload = static_cast<uint8_t>(mant >> width);
        dropped = mant & ((1u << width) - 1);
        halfway = 1u << (width - 1);
    }
    if (dropped > halfway || (dropped == halfway && (payload & 1))) payload++;
    return __builtin_bit_cast(fp8_t, static_cast<uint8_t>(sign | payload));
}

inline float fp8_e4m3_to_float(fp8_t v) {
    const uint8_t  bits = __builtin_bit_cast(uint8_t, v);
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

static constexpr float MLA_DECODE_LN_2 = 0.69314718055994531f;
static constexpr float MLA_DECODE_LOG2_E = 1.4426950408889634f;

static constexpr int MLA_DECODE_NUM_CU = 256;
static constexpr int MLA_DECODE_KV_GRANULARITY = 16;
static constexpr int MLA_DECODE_FIXED_OVERHEAD = 16;
static constexpr int MLA_DECODE_PACKED_QO_LEN_PER_WG = 128;
static constexpr int MLA_DECODE_MAX_SPLIT_PER_BATCH = 32;
static constexpr float MLA_DECODE_SPLIT_COEF = 1.2f;

struct opus_mla_decode_work_info {
    int batch_idx;
    int partial_slot;
    int qo_start;
    int qo_end;
    int kv_start;
    int kv_end;
    int kv_offset;
    int _pad;
};

struct opus_mla_decode_mxfp8_kargs {
    const void* __restrict__ q_nope_ptr;
    const void* __restrict__ q_scale_ptr;
    const void* __restrict__ q_rope_ptr;
    const void* __restrict__ kv_nope_ptr;
    const void* __restrict__ kv_scale_ptr;
    const void* __restrict__ kv_rope_ptr;
    void* __restrict__ out_ptr;
    void* __restrict__ lse_ptr;
    void* __restrict__ o_accum;
    void* __restrict__ lse_accum;

    const int* __restrict__ q_indptr;
    const int* __restrict__ kv_indptr;
    const int* __restrict__ kv_indices;
    const int* __restrict__ work_indptr;
    const opus_mla_decode_work_info* __restrict__ work_info_set;

    int H;
    int total_tokens;
    int stride_q_nope_b;
    int stride_q_nope_h;
    int stride_q_scale_b;
    int stride_q_scale_h;
    int stride_q_rope_b;
    int stride_q_rope_h;
    int stride_o_b;
    int stride_o_h;
    int stride_kv_nope_page;
    int stride_kv_scale_page;
    int stride_kv_rope_page;
    float softmax_scale;
};

struct opus_mla_decode_kargs {
    const void* __restrict__ q_ptr;
    const void* __restrict__ kv_ptr;
    void* __restrict__ out_ptr;
    void* __restrict__ lse_ptr;
    void* __restrict__ o_accum;
    void* __restrict__ lse_accum;

    const int* __restrict__ q_indptr;
    const int* __restrict__ kv_indptr;
    const int* __restrict__ kv_indices;
    const int* __restrict__ work_indptr;
    const opus_mla_decode_work_info* __restrict__ work_info_set;

    int H;
    int total_tokens;
    int stride_q_b;
    int stride_q_h;
    int stride_o_b;
    int stride_o_h;
    int stride_kv_page;
    float softmax_scale;
};

// Harness-only: in production aiter owns both stages, so these are not ABI.
struct opus_mla_decode_metadata_kargs {
    const int* __restrict__ qo_indptr;
    const int* __restrict__ kv_indptr;

    int* __restrict__ work_indptr;
    opus_mla_decode_work_info* __restrict__ work_info_set;
    int* __restrict__ reduce_indptr;
    int* __restrict__ reduce_final_map;
    int* __restrict__ reduce_partial_map;

    int B;
    int H;
    int num_cu;
    int num_splits;
    int uni_seqlen_qo;
    int kv_granularity;
    int fixed_overhead;
    int tail_done_threshold;
    int reduce_indptr_size;
    int auto_split;
    int is_causal;
};

struct opus_mla_decode_reduce_kargs {
    const void* __restrict__ o_accum;
    const void* __restrict__ lse_accum;
    void* __restrict__ out_ptr;
    void* __restrict__ lse_ptr;

    const int* __restrict__ reduce_indptr;
    const int* __restrict__ reduce_final_map;
    const int* __restrict__ reduce_partial_map;

    int H;
    int stride_o_b;
    int stride_o_h;
};

// Frozen: the prebuilt code objects are compiled against these layouts.
static_assert(sizeof(opus_mla_decode_work_info) == 32);
static_assert(sizeof(opus_mla_decode_kargs) == 120);
static_assert(offsetof(opus_mla_decode_kargs, work_info_set) == 80);
static_assert(offsetof(opus_mla_decode_kargs, softmax_scale) == 116);
static_assert(sizeof(opus_mla_decode_mxfp8_kargs) == 176);
static_assert(offsetof(opus_mla_decode_mxfp8_kargs, work_info_set) == 112);
static_assert(offsetof(opus_mla_decode_mxfp8_kargs, softmax_scale) == 172);

template<typename T>
__host__ __device__ inline T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

__host__ __device__ inline int mla_decode_num_qo_tiles(int seqlen_qo, int H) {
    if (seqlen_qo * H <= MLA_DECODE_PACKED_QO_LEN_PER_WG) return 1;
    if (H * 2 > MLA_DECODE_PACKED_QO_LEN_PER_WG) return seqlen_qo;
    return ceil_div(seqlen_qo * H, MLA_DECODE_PACKED_QO_LEN_PER_WG);
}

#if defined(MLA_DECODE_ARCH_GFX950)
#include "gfx950/traits.h"
#else
#error "no MLA_DECODE_ARCH_* defined; build through the Makefile"
#endif
