// Configuration traits for the gfx950 (wave64 / MFMA) GQA forward kernels.
#pragma once

#include "common/gqa_kargs.h"

using bf16_t = __bf16;

// D=128: MFMA 32x32x16, whole D per tile. D=512: MFMA 16x16x32, D in slices of 32.
template<int Q_TILE_SIZE_ = 16,
        int KV_TILE_SIZE_ = 32,
        int D_TILE_SIZE_ = 512,
        int NUM_WARPS_ = 8,
        bool CAUSAL_ = false>
struct opus_gqa_traits {
    static_assert(D_TILE_SIZE_ == 128 || D_TILE_SIZE_ == 512,
                  "opus_gqa_traits supports D_TILE_SIZE 128 or 512");

    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE = D_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr bool CAUSAL = CAUSAL_;

    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr int WG_Q_ROWS = Q_TILE_SIZE * NUM_WARPS;

    using D_ATTN = bf16_t;
    using D_ACC  = float;

    // waves along M, N, K
    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static constexpr int W_M = (D_TILE_SIZE == 128) ? 32 : 16;
    static constexpr int W_N = (D_TILE_SIZE == 128) ? 32 : 16;
    static constexpr int W_K = (D_TILE_SIZE == 128) ? 16 : 32;

    static constexpr int SLICE_D = (D_TILE_SIZE == 512) ? 32 : D_TILE_SIZE;
    static constexpr int NUM_D_SLICES = D_TILE_SIZE / SLICE_D;
    static_assert(D_TILE_SIZE % SLICE_D == 0);

    // GEMM0: S[Q_TILE x KV_TILE] = Q[Q_TILE x SLICE_D] @ K^T[SLICE_D x KV_TILE]
    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / W_N;
    static constexpr int GEMM0_E_K = SLICE_D / W_K;

    // GEMM1: O[Q_TILE x SLICE_D] = P[Q_TILE x KV_TILE] @ V[KV_TILE x SLICE_D]
    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = SLICE_D / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;

    static constexpr int VEC_Q    = 8;
    static constexpr int VEC_KV   = 8;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O    = 4;

    // LDS: one wave's async copy is 16 bytes per lane
    static constexpr int D_128B_SIZE = 128 / sizeof(D_ATTN);
    static_assert(VEC_KV == 16 / sizeof(D_ATTN));
    static constexpr int smem_linear_wave = WARP_SIZE * 16 / sizeof(D_ATTN);
    static constexpr int smem_n_per_wave = smem_linear_wave / D_128B_SIZE;
    static constexpr int smem_n_rpt = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt = D_TILE_SIZE / D_128B_SIZE;

    static constexpr int smem_padding_16B = 16 / sizeof(D_ATTN);
    static constexpr int smem_padding_32B = 32 / sizeof(D_ATTN);
    static constexpr int smem_padding_64B = 64 / sizeof(D_ATTN);

    // K/V row padding: D=128 16 / 64 bytes, D=512 32 bytes
    static constexpr int smem_k_padding = (D_TILE_SIZE == 128) ? smem_padding_16B : smem_padding_32B;
    static constexpr int smem_v_padding = (D_TILE_SIZE == 128) ? smem_padding_64B : smem_padding_32B;

    static constexpr int smem_k_tile_elems = smem_n_rpt * smem_d_rpt * (smem_linear_wave + smem_k_padding);
    static constexpr int smem_v_tile_elems = smem_n_rpt * smem_d_rpt * (smem_linear_wave + smem_v_padding);
    static constexpr int smem_buffer_elems = smem_k_tile_elems + smem_v_tile_elems;

    static constexpr int k_buffer_load_insts = (KV_TILE_SIZE * D_TILE_SIZE) / (BLOCK_SIZE * VEC_KV);
    static constexpr int v_buffer_load_insts = (KV_TILE_SIZE * D_TILE_SIZE) / (BLOCK_SIZE * VEC_KV);
    static constexpr int k_ds_read_insts = (GEMM0_E_N * GEMM0_E_K * W_N * W_K) / (WARP_SIZE * VEC_KV);
    static constexpr int v_ds_read_insts = (GEMM1_E_N * GEMM1_E_K * W_N * W_K) / (WARP_SIZE * VEC_TR_V);
};

// The instantiated kernels
template <bool CAUSAL> using gqa_d128_traits = opus_gqa_traits<32, 64, 128, 8, CAUSAL>;
template <bool CAUSAL> using gqa_d512_traits = opus_gqa_traits<16, 32, 512, 8, CAUSAL>;

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
