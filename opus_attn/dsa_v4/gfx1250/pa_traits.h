// Configuration traits for gfx1250 (wave32 / WMMA) PA prefill kernels.
#pragma once

#include "common/pa_kargs.h"

template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 64,
         int D_TILE_SIZE_ = 512,
         int NUM_WARPS_ = 4,
         typename D_ATTN_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct pa_16mx4_64nx1_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE = D_TILE_SIZE_;
    static constexpr int D_HEAD_SIZE = D_TILE_SIZE;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 32;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    using D_ATTN = D_ATTN_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    // Wave grid
    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    // WMMA base tile
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    // GEMM0: S = Q @ K^T
    static constexpr int GEMM0_STAGE_N = 4;
    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = (KV_TILE_SIZE / GEMM0_STAGE_N) / W_N;
    static constexpr int GEMM0_E_K = D_TILE_SIZE / W_K;

    // GEMM1: O = P @ V
    static constexpr int GEMM1_STAGE_N = 2;
    static constexpr int GEMM1_STAGE_K = 2;
    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = (D_TILE_SIZE / GEMM1_STAGE_N) / W_N;
    static constexpr int GEMM1_E_K = (KV_TILE_SIZE / GEMM1_STAGE_K) / W_K;

    static constexpr int VEC_Q  = 8;
    static constexpr int VEC_KV = 8;
    static constexpr int VEC_O  = 8;

    // ds_load instruction count per GEMM0 / GEMM1 stage
    static constexpr int k_ds_load_insts = (GEMM0_E_N * GEMM0_E_K * W_N * W_K) / (WARP_SIZE * VEC_KV);
    static constexpr int v_ds_load_insts = (GEMM1_E_N * GEMM1_E_K * W_N * W_K) / (WARP_SIZE * VEC_KV);

    // TDM gather KV load
    static constexpr int ROWS_PER_WAVE      = KV_TILE_SIZE / NUM_WARPS;
    static constexpr int INDICES_PER_TDM    = 8;                                     // 32-bit gather cap
    static constexpr int TDM_LOADS_PER_WAVE = ROWS_PER_WAVE / INDICES_PER_TDM;
    static constexpr int KV_ROW_LDS_BYTES   = D_TILE_SIZE * sizeof(D_ATTN) + 16;
    static constexpr int KV_ROW_PAD_SIZE    = 16 / sizeof(D_ATTN);
    static constexpr int KV_ROW_LDS_ELEMS   = D_TILE_SIZE + KV_ROW_PAD_SIZE;

    // LDS: a tile's two row halves sit SEG_BYTES apart, slots pack back to back inside each half
    static constexpr int WAVES_PER_SEG      = 2;
    static constexpr int SEGS_PER_BUF       = NUM_WARPS / WAVES_PER_SEG;
    static constexpr int ROWS_PER_SEG       = WAVES_PER_SEG * ROWS_PER_WAVE;
    static constexpr int WAVE_LDS_BYTES     = ROWS_PER_WAVE * KV_ROW_LDS_BYTES;
    static constexpr int SEG_BYTES          = 128 * 1024;
    static constexpr int KV_BUF_BYTES       = ROWS_PER_SEG * KV_ROW_LDS_BYTES;       // slot stride
    static constexpr int NUM_KV_BUFS        = 3;                                     // QK(t+2) runs before PV(t+1)

    static_assert(KV_TILE_SIZE % NUM_WARPS == 0 && ROWS_PER_WAVE % INDICES_PER_TDM == 0);
    static_assert((size_t)NUM_KV_BUFS * KV_BUF_BYTES <= (size_t)SEG_BYTES, "slots of one half must not reach into the other");
    static_assert(D_TILE_SIZE * sizeof(D_ATTN) == 1024, "TDM pad_interval=7 assumes a 1024B / 256-DWORD row");

    static constexpr size_t smem_size_bytes() { return (size_t)SEG_BYTES + NUM_KV_BUFS * KV_BUF_BYTES; }
};