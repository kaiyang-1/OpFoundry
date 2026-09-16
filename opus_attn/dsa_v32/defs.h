#pragma once

#include <algorithm>

using bf16_t = __bf16;
using fp16_t = __fp16;
using fp8_t  = _BitInt(8);
using bf8_t  = unsigned _BitInt(8);

static constexpr int DSA_V32_NUM_CU = 256;
static constexpr int DSA_V32_FIXED_OVERHEAD = 5;
static constexpr float DSA_V32_LN_2 = 0.69314718055994531f;

struct alignas(16) DsaSchedMeta {
    int begin_req_idx;
    int end_req_idx;
    int begin_tile_idx;
    int end_tile_idx;
    int begin_split_idx;
    int _pad[3];
};

struct dsa_v32_a8w8_kargs {
    const void* __restrict__ q_nope_ptr;
    const void* __restrict__ q_scale_ptr;
    const void* __restrict__ q_rope_ptr;
    const void* __restrict__ kv_nope_ptr;
    const void* __restrict__ kv_scale_ptr;
    const void* __restrict__ kv_rope_ptr;
    void* __restrict__ out_ptr;
    void* __restrict__ lse_ptr;
    const int* __restrict__ kv_indptr;
    const int* __restrict__ kv_indices;

    const DsaSchedMeta* __restrict__ sched_meta;
    const int* __restrict__ num_splits;
    void* __restrict__ o_accum;
    void* __restrict__ lse_accum;
    int num_parts;

    int B;
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
    int stride_lse_b;
    int stride_kv_nope_page;
    int stride_kv_scale_page;
    int stride_kv_rope_page;
    float softmax_scale;
};

struct dsa_v32_a16w16_kargs {
    const void* __restrict__ q_ptr;
    const void* __restrict__ kv_ptr;
    void* __restrict__ out_ptr;
    void* __restrict__ lse_ptr;
    const int* __restrict__ kv_indptr;
    const int* __restrict__ kv_indices;

    const DsaSchedMeta* __restrict__ sched_meta;
    const int* __restrict__ num_splits;
    void* __restrict__ o_accum;
    void* __restrict__ lse_accum;
    int num_parts;

    int B;
    int H;
    int total_tokens;
    int stride_q_b;
    int stride_q_h;
    int stride_o_b;
    int stride_o_h;
    int stride_lse_b;
    int stride_kv_page;
    float softmax_scale;
};

template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 32,
         int NUM_WARPS_ = 8,
         typename D_NOPE_ = fp8_t,
         typename D_ROPE_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct dsa_v32_decode_a8w8_16mx8_32nx1_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_NOPE_SIZE = 512;
    static constexpr int D_ROPE_SIZE = 64;
    static constexpr int D_HEAD_SIZE = D_NOPE_SIZE + D_ROPE_SIZE;
    static constexpr int D_SCALE_SIZE = D_NOPE_SIZE / 32;
    static constexpr int D_SCALE_PADDED_SIZE = 32;

    static constexpr int D_QK_SIZE = D_HEAD_SIZE;
    static constexpr int D_VO_SIZE  = D_NOPE_SIZE;

    using D_NOPE = D_NOPE_;
    using D_ROPE = D_ROPE_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K_NOPE = 128;
    static constexpr int W_K_ROPE = 32;

    static constexpr int SLICE_D = 32;
    static constexpr int NUM_D_SLICES = D_NOPE_SIZE / SLICE_D;

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / W_N;
    static constexpr int GEMM0_NOPE_E_K = D_NOPE_SIZE / W_K_NOPE;
    static constexpr int GEMM0_ROPE_E_K = D_ROPE_SIZE / W_K_ROPE;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = SLICE_D / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K_ROPE;

    static constexpr int VEC_Q_NOPE  = 16;
    static constexpr int VEC_Q_ROPE  = 8;
    static constexpr int VEC_KV_NOPE = 16;
    static constexpr int VEC_KV_ROPE = 8;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O    = 4;

    static constexpr int D_128B_NOPE_SIZE = 128 / sizeof(D_NOPE);
    static constexpr int dwordx4_size = 16;
    static constexpr int smem_linear_wave_nope = WARP_SIZE * dwordx4_size / sizeof(D_NOPE);
    static constexpr int smem_n_per_wave = 8;
    static constexpr int smem_n_rpt = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt_nope = D_NOPE_SIZE / D_128B_NOPE_SIZE;
    static constexpr int smem_padding_32B_nope = 32 / sizeof(D_NOPE);
    static constexpr size_t smem_k_nope_bytes = smem_n_rpt * smem_d_rpt_nope * (smem_linear_wave_nope + smem_padding_32B_nope) * sizeof(D_NOPE);

    static constexpr int D_128B_ROPE_SIZE = 128 / sizeof(D_ROPE);
    static constexpr int smem_linear_wave_rope = WARP_SIZE * dwordx4_size / sizeof(D_ROPE);
    static constexpr int smem_d_rpt_rope = D_ROPE_SIZE / D_128B_ROPE_SIZE;
    static constexpr int smem_padding_32B_rope = 32 / sizeof(D_ROPE);
    static constexpr size_t smem_k_rope_bytes = smem_n_rpt * smem_d_rpt_rope * (smem_linear_wave_rope + smem_padding_32B_rope) * sizeof(D_ROPE);

    static constexpr int smem_v_padding = 32 / sizeof(D_ROPE);
    static constexpr size_t smem_v_bytes = KV_TILE_SIZE * (D_NOPE_SIZE + smem_v_padding) * sizeof(D_ROPE);

    static constexpr int smem_mxscl_padding = 4 / sizeof(D_NOPE);
    static constexpr size_t smem_mxscl_bytes = smem_n_rpt * (D_SCALE_PADDED_SIZE * smem_n_per_wave + smem_mxscl_padding) * sizeof(D_NOPE);

    static constexpr size_t smem_kv_bytes() {
        return std::max(smem_k_nope_bytes + smem_k_rope_bytes, smem_v_bytes);
    }

    static constexpr int kv_buffer_load_insts = (KV_TILE_SIZE * D_NOPE_SIZE) / (BLOCK_SIZE * VEC_KV_NOPE)  // nope = 2
                                                + 1; // rope = 1 for warp_id < 4 or mxscl = 1 for warp_id >= 4
    static constexpr int k_nope_ds_read_insts = (GEMM0_E_N * W_N * W_K_NOPE) / (WARP_SIZE * VEC_KV_NOPE);
    static constexpr int k_rope_ds_read_insts = (GEMM0_E_N * W_N * W_K_ROPE) / (WARP_SIZE * VEC_KV_ROPE);
    static constexpr int v_ds_read_insts = (GEMM1_E_N * GEMM1_E_K * W_N * W_K_ROPE) / (WARP_SIZE * VEC_TR_V);
};

template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 64,
         int NUM_WARPS_ = 4,
         typename D_ATTN_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct dsa_v32_decode_a16w16_16mx4_64nx1_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_QK_SIZE = 576;
    static constexpr int D_VO_SIZE  = 512;

    using D_ATTN = D_ATTN_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM0_E_K = D_QK_SIZE / W_K;

    static constexpr int SLICE_D = 32;
    static constexpr int NUM_D_SLICES = D_VO_SIZE / SLICE_D;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = SLICE_D / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;

    static constexpr int VEC_Q = 8;
    static constexpr int VEC_KV = 8;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O = 4;

    static constexpr int dwordx4_size = 16;
    static constexpr int D_128B_SIZE = 128 / sizeof(D_ATTN);
    static constexpr int smem_linear_wave = WARP_SIZE * dwordx4_size / sizeof(D_ATTN);
    static constexpr int smem_n_per_wave = smem_linear_wave / D_128B_SIZE;
    static constexpr int smem_n_rpt = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt = D_QK_SIZE / D_128B_SIZE;
    static constexpr int smem_d_rpt_v = D_VO_SIZE / D_128B_SIZE;
    static constexpr int smem_padding_32B = 32 / sizeof(D_ATTN);
    static constexpr int smem_brick = smem_linear_wave + smem_padding_32B;

    static constexpr int smem_n_sub_tile = smem_n_per_wave * NUM_WARPS;
    static constexpr int smem_n_sub_tile_rpt = KV_TILE_SIZE / smem_n_sub_tile;
    static constexpr int kv_async_load_insts = smem_n_sub_tile_rpt * smem_d_rpt;

    static constexpr size_t smem_kv_bytes = (size_t)smem_n_rpt * smem_d_rpt * smem_brick * sizeof(D_ATTN);

    static constexpr int NUM_KV_BUFS = 2;
    static constexpr int smem_slot_elems = (int)(smem_kv_bytes / sizeof(D_ATTN));
    static constexpr size_t smem_bytes() { return NUM_KV_BUFS * smem_kv_bytes; }
};

template<int Q_TILE_SIZE_ = 32,
         int KV_TILE_SIZE_ = 64,
         int NUM_WARPS_ = 4,
         typename D_ATTN_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct dsa_v32_decode_a16w16_32mx1_16nx4_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_QK_SIZE = 576;
    static constexpr int D_VO_SIZE  = 512;

    using D_ATTN = D_ATTN_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM0_E_K = D_QK_SIZE / W_K;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = D_VO_SIZE / (W_N * T_N);
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;

    static constexpr int VEC_Q = 8;
    static constexpr int VEC_KV = 8;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O = 4;

    static constexpr int dwordx4_size = 16;
    static constexpr int D_128B_SIZE = 128 / sizeof(D_ATTN);
    static constexpr int smem_linear_wave = WARP_SIZE * dwordx4_size / sizeof(D_ATTN);
    static constexpr int smem_n_per_wave = smem_linear_wave / D_128B_SIZE;
    static constexpr int smem_n_rpt = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt = D_QK_SIZE / D_128B_SIZE;
    static constexpr int smem_d_rpt_v = D_VO_SIZE / D_128B_SIZE;
    static constexpr int smem_padding_32B = 32 / sizeof(D_ATTN);
    static constexpr int smem_brick = smem_linear_wave + smem_padding_32B;

    static constexpr int smem_n_sub_tile = smem_n_per_wave * NUM_WARPS;
    static constexpr int smem_n_sub_tile_rpt = KV_TILE_SIZE / smem_n_sub_tile;
    static constexpr int kv_async_load_insts = smem_n_sub_tile_rpt * smem_d_rpt;

    static constexpr size_t smem_kv_bytes = (size_t)smem_n_rpt * smem_d_rpt * smem_brick * sizeof(D_ATTN);

    static constexpr int NUM_KV_BUFS = 2;
    static constexpr int smem_slot_elems = (int)(smem_kv_bytes / sizeof(D_ATTN));

    static constexpr int ML_ELEMS = GEMM0_E_M;
    static constexpr int ML_SLOT_ELEMS = GEMM0_E_M * W_M * T_N;
    static constexpr int P_SLOT_ELEMS = Q_TILE_SIZE * KV_TILE_SIZE;
    static constexpr size_t smem_ml_bytes = 2 * (size_t)ML_SLOT_ELEMS * sizeof(D_ACC);
    static constexpr size_t smem_p_bytes  = 2 * (size_t)P_SLOT_ELEMS * sizeof(D_ATTN);

    static constexpr size_t smem_bytes() {
        return NUM_KV_BUFS * smem_kv_bytes + smem_ml_bytes + smem_p_bytes;
    }
};

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
