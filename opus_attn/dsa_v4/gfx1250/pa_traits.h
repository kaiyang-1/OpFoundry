// Configuration traits for gfx1250 (wave32 / WMMA) PA prefill kernels.
#pragma once

#include "common/pa_kargs.h"

template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 64,
         int D_TILE_SIZE_ = 512,
         int NUM_WARPS_ = 4,
         int CLUSTER_Y_ = 1,
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
    static constexpr int CLUSTER_Y = CLUSTER_Y_;

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
    static constexpr int SEG_BYTES          = 160 * 1024;
    static constexpr int KV_BUF_BYTES       = ROWS_PER_SEG * KV_ROW_LDS_BYTES;       // slot stride
    static constexpr int NUM_KV_BUFS        = 4;                                     // QK(t) gathers t+2, so t-1, t, t+1, t+2 coexist
    static constexpr int KV_STAGE_BYTES     = SEG_BYTES + NUM_KV_BUFS * KV_BUF_BYTES;

    static constexpr int Q_ROW_LDS_ELEMS    = D_TILE_SIZE + 16 / sizeof(D_ATTN);
    static constexpr int O_ROW_LDS_ELEMS    = D_TILE_SIZE + 16 / sizeof(D_OUT);
    static constexpr int QO_SEG_BYTES       = 64 * 1024;
    static constexpr int QO_STAGE_BYTES     = NUM_WARPS * QO_SEG_BYTES;

    static constexpr size_t smem_size_bytes() { return (size_t)(KV_STAGE_BYTES > QO_STAGE_BYTES ? KV_STAGE_BYTES : QO_STAGE_BYTES); }
};

template<int Q_TILE_SIZE_ = 32,
         int KV_TILE_SIZE_ = 64,
         int D_TILE_SIZE_ = 512,
         int NUM_WARPS_ = 4,
         typename D_ATTN_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct pa_32mx1_16nx4_traits {
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
    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    // WMMA base tile
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    // GEMM0: S = Q @ K^T
    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM0_E_K = D_TILE_SIZE / W_K;

    // GEMM1: O = P @ V
    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = D_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;

    static constexpr int VEC_Q  = 8;
    static constexpr int VEC_KV = 8;
    static constexpr int VEC_P  = 8;
    static constexpr int VEC_O  = 8;

    // ds_load instruction count per GEMM0 / GEMM1 tile
    static constexpr int k_ds_load_insts = (GEMM0_E_N * GEMM0_E_K * W_N * W_K) / (WARP_SIZE * VEC_KV);
    static constexpr int v_ds_load_insts = (GEMM1_E_N * GEMM1_E_K * W_N * W_K) / (WARP_SIZE * VEC_KV);

    // Q/O staging tile, shared by the whole workgroup
    static constexpr int QO_ROWS_PER_WAVE = Q_TILE_SIZE / NUM_WARPS;
    static constexpr int Q_ROW_LDS_ELEMS  = D_TILE_SIZE + 16 / sizeof(D_ATTN);
    static constexpr int O_ROW_LDS_ELEMS  = D_TILE_SIZE + 16 / sizeof(D_OUT);
    static constexpr int Q_TILE_LDS_BYTES = Q_TILE_SIZE * Q_ROW_LDS_ELEMS * sizeof(D_ATTN);
    static constexpr int O_TILE_LDS_BYTES = Q_TILE_SIZE * O_ROW_LDS_ELEMS * sizeof(D_OUT);
    static constexpr int QO_LDS_BYTES     = Q_TILE_LDS_BYTES > O_TILE_LDS_BYTES ? Q_TILE_LDS_BYTES : O_TILE_LDS_BYTES;

    // TDM gather KV load
    static constexpr int ROWS_PER_WAVE      = KV_TILE_SIZE / NUM_WARPS;
    static constexpr int INDICES_PER_TDM    = 8;                                     // 32-bit gather cap
    static constexpr int TDM_LOADS_PER_WAVE = ROWS_PER_WAVE / INDICES_PER_TDM;
    static constexpr int KV_ROW_PAD_SIZE    = 16 / sizeof(D_ATTN);
    static constexpr int KV_ROW_LDS_BYTES   = D_TILE_SIZE * sizeof(D_ATTN) + 16;
    static constexpr int KV_ROW_LDS_ELEMS   = D_TILE_SIZE + KV_ROW_PAD_SIZE;

    // KV ring
    static constexpr int WAVE_LDS_BYTES = ROWS_PER_WAVE * KV_ROW_LDS_BYTES;
    static constexpr int KV_BUF_BYTES   = KV_TILE_SIZE * KV_ROW_LDS_BYTES;      // slot stride
    static constexpr int KV_BUF_ELEMS   = KV_BUF_BYTES / (int)sizeof(D_ATTN);
    static constexpr int NUM_KV_BUFS    = 4;                                    // a slot is K one round and V the next, and reuse trails it by two barriers
    static constexpr int GATHER_AHEAD   = 2;
    static constexpr int KV_LDS_BYTES   = NUM_KV_BUFS * KV_BUF_BYTES;

    // Cross-wave row-max exchange, [m-block][head row][wave]: one b128 read per row
    static constexpr int ML_SLOT_ELEMS = GEMM0_E_M * W_M * T_N;
    static constexpr int ML_SLOT_BYTES = ML_SLOT_ELEMS * (int)sizeof(D_ACC);
    static constexpr int M_LDS_OFF     = KV_LDS_BYTES;
    static constexpr int L_LDS_OFF     = M_LDS_OFF + ML_SLOT_BYTES;

    // Cross-wave P exchange, [m-block][wave][chunk][lane][VEC_P]: 16B lane stride, and
    // the block order is the mma1 A-operand order so neither side shuffles lanes
    static constexpr int P_CHUNKS      = GEMM0_E_N * W_M * W_N / (WARP_SIZE * VEC_P);
    static constexpr int P_BLOCK_ELEMS = WARP_SIZE * VEC_P;
    static constexpr int P_NUM_BLOCKS  = GEMM0_E_M * T_N * P_CHUNKS;
    static constexpr int P_LDS_OFF     = L_LDS_OFF + ML_SLOT_BYTES;
    static constexpr int P_LDS_BYTES   = P_NUM_BLOCKS * P_BLOCK_ELEMS * (int)sizeof(D_ATTN);

    static constexpr int ACCUM_LDS_BYTES = P_LDS_OFF + P_LDS_BYTES;

    static constexpr size_t smem_size_bytes() {
        return (size_t)(ACCUM_LDS_BYTES > QO_LDS_BYTES ? ACCUM_LDS_BYTES : QO_LDS_BYTES);
    }
};

template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 64,
         int D_TILE_SIZE_ = 512,
         int NUM_WARPS_ = 4,
         typename D_ATTN_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct pa_16mx1_16nx4_traits {
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
    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    // WMMA base tile
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    // GEMM0: S = Q @ K^T
    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM0_E_K = D_TILE_SIZE / W_K;

    // GEMM1: O = P @ V
    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = D_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;

    static constexpr int VEC_Q  = 8;
    static constexpr int VEC_KV = 8;
    static constexpr int VEC_P  = 8;
    static constexpr int VEC_O  = 8;

    static constexpr int QO_ROWS_PER_WAVE = Q_TILE_SIZE / NUM_WARPS;
    static constexpr int Q_ROW_LDS_ELEMS  = D_TILE_SIZE + 16 / sizeof(D_ATTN);
    static constexpr int O_ROW_LDS_ELEMS  = D_TILE_SIZE + 16 / sizeof(D_OUT);
    static constexpr int Q_TILE_LDS_BYTES = Q_TILE_SIZE * Q_ROW_LDS_ELEMS * sizeof(D_ATTN);
    static constexpr int O_TILE_LDS_BYTES = Q_TILE_SIZE * O_ROW_LDS_ELEMS * sizeof(D_OUT);
    static constexpr int QO_LDS_BYTES     = Q_TILE_LDS_BYTES > O_TILE_LDS_BYTES ? Q_TILE_LDS_BYTES : O_TILE_LDS_BYTES;

    static constexpr int ROWS_PER_WAVE      = KV_TILE_SIZE / NUM_WARPS;
    static constexpr int INDICES_PER_TDM    = 8;
    static constexpr int TDM_LOADS_PER_WAVE = ROWS_PER_WAVE / INDICES_PER_TDM;
    static constexpr int KV_ROW_PAD_SIZE    = 16 / sizeof(D_ATTN);
    static constexpr int KV_ROW_LDS_BYTES   = D_TILE_SIZE * sizeof(D_ATTN) + 16;
    static constexpr int KV_ROW_LDS_ELEMS   = D_TILE_SIZE + KV_ROW_PAD_SIZE;

    static constexpr int NUM_KV_BUFS    = 2;
    static constexpr int WAVE_LDS_BYTES = ROWS_PER_WAVE * KV_ROW_LDS_BYTES;
    static constexpr int KV_BUF_BYTES   = KV_TILE_SIZE * KV_ROW_LDS_BYTES;
    static constexpr int KV_LDS_BYTES   = NUM_KV_BUFS * KV_BUF_BYTES;

    static constexpr int ML_LDS_OFF   = KV_LDS_BYTES;
    static constexpr int ML_LDS_BYTES = 2 * T_N * W_M * sizeof(D_ACC);

    static constexpr int P_LDS_OFF   = ML_LDS_OFF + ML_LDS_BYTES;
    static constexpr int P_BLOCK_ELEMS = W_M * W_N;
    static constexpr int P_LDS_BYTES = NUM_WARPS * P_BLOCK_ELEMS * sizeof(D_ATTN);

    static constexpr int ACCUM_LDS_BYTES = P_LDS_OFF + P_LDS_BYTES;

    static constexpr size_t smem_size_bytes() {
        return (size_t)(ACCUM_LDS_BYTES > QO_LDS_BYTES ? ACCUM_LDS_BYTES : QO_LDS_BYTES);
    }
};

template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 64,
         int NUM_WARPS_ = 4,
         int CLUSTER_Y_ = 1,
         typename D_NOPE_ = fp8_t,
         typename D_ROPE_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct pa_16mx4_64nx1_fp8_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 32;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr int CLUSTER_Y = CLUSTER_Y_;
    static constexpr int MXSCL_BLOCK_SIZE = 32;

    // Packed DSA hdim split
    static constexpr int D_NOPE_SIZE = 448;
    static constexpr int D_NOPE_PADDED_SIZE = 512;
    static constexpr int D_ROPE_SIZE = 64;
    static constexpr int D_HEAD_SIZE = D_NOPE_SIZE + D_ROPE_SIZE;

    using D_NOPE = D_NOPE_;
    using D_ROPE = D_ROPE_;
    using D_ATTN = D_NOPE_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    // Wave grid
    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    // WMMA base tile: NoPE QK^T runs on scaled f8f6f4 16x16x128, RoPE QK^T and PV on bf16 16x16x32.
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K_NOPE = 128;
    static constexpr int W_K_ROPE = 32;

    // GEMM0: S = Q @ K^T
    static constexpr int GEMM0_STAGE_N = 4;
    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = (KV_TILE_SIZE / GEMM0_STAGE_N) / W_N;
    static constexpr int GEMM0_NOPE_E_K = D_NOPE_PADDED_SIZE / W_K_NOPE;
    static constexpr int GEMM0_ROPE_E_K = D_ROPE_SIZE / W_K_ROPE;

    // GEMM1: O = P @ V, with V dequantized to bf16 ahead of the PV WMMA.
    static constexpr int GEMM1_STAGE_N = 2;
    static constexpr int GEMM1_STAGE_K = 2;
    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = (D_HEAD_SIZE / GEMM1_STAGE_N) / W_N;
    static constexpr int GEMM1_E_K = (KV_TILE_SIZE / GEMM1_STAGE_K) / W_K_ROPE;

    static constexpr int VEC_NOPE  = 16;
    static constexpr int VEC_ROPE  = 8;
    static constexpr int VEC_MXSCL = 4;
    static constexpr int VEC_O     = 8;

    // ds_load instruction count per GEMM0 / GEMM1 stage
    static constexpr int k_nope_ds_load_insts = (GEMM0_E_N * GEMM0_NOPE_E_K * W_N * W_K_NOPE) / (WARP_SIZE * VEC_NOPE);
    static constexpr int k_rope_ds_load_insts = (GEMM0_E_N * GEMM0_ROPE_E_K * W_N * W_K_ROPE) / (WARP_SIZE * VEC_ROPE);
    static constexpr int v_ds_load_insts      = (GEMM1_E_N * GEMM1_E_K * W_N * W_K_ROPE) / (WARP_SIZE * VEC_ROPE);

    // TDM gather KV load.
    static constexpr int ROWS_PER_WAVE      = KV_TILE_SIZE / NUM_WARPS;
    static constexpr int INDICES_PER_TDM    = 8;                                     // 32-bit gather cap
    static constexpr int TDM_LOADS_PER_WAVE = ROWS_PER_WAVE / INDICES_PER_TDM;

    // Per-row LDS footprint: the TDM pad settings add one 16B pad at the end of every row.
    static constexpr int K_NOPE_ROW_LDS_BYTES = D_NOPE_PADDED_SIZE * sizeof(D_NOPE) + 16;
    static constexpr int K_ROPE_ROW_LDS_BYTES = D_ROPE_SIZE * sizeof(D_ROPE) + 16;
    static constexpr int K_NOPE_ROW_LDS_ELEMS = K_NOPE_ROW_LDS_BYTES / sizeof(D_NOPE);
    static constexpr int K_ROPE_ROW_LDS_ELEMS = K_ROPE_ROW_LDS_BYTES / sizeof(D_ROPE);
    static constexpr int V_ROW_LDS_BYTES = D_HEAD_SIZE * sizeof(D_ROPE) + 16;
    static constexpr int V_ROW_LDS_ELEMS = V_ROW_LDS_BYTES / sizeof(D_ROPE);

    // LDS: one 64KB segment per (row half, buffer), holding that half's K and V together.
    static constexpr int SEGS_PER_TILE    = 2;
    static constexpr int ROWS_PER_SEG     = KV_TILE_SIZE / SEGS_PER_TILE;
    static constexpr int SEG_BYTES        = 64 * 1024;
    static constexpr int K_NOPE_SEG_BYTES = ROWS_PER_SEG * K_NOPE_ROW_LDS_BYTES;
    static constexpr int K_ROPE_SEG_BYTES = ROWS_PER_SEG * K_ROPE_ROW_LDS_BYTES;
    static constexpr int K_SLOT_BYTES     = K_NOPE_SEG_BYTES + K_ROPE_SEG_BYTES;
    static constexpr int V_SLOT_BYTES     = ROWS_PER_SEG * V_ROW_LDS_BYTES;

    static constexpr int NUM_KV_BUFS    = 2;
    static constexpr int KV_BUF_BYTES   = SEGS_PER_TILE * SEG_BYTES;

    static constexpr int SEG_USED_BYTES = K_SLOT_BYTES + V_SLOT_BYTES;
    static_assert(SEG_USED_BYTES <= SEG_BYTES, "a segment's K and V must not reach into the next one");
    static constexpr int KV_STAGE_BYTES = NUM_KV_BUFS * KV_BUF_BYTES;

    static constexpr int O_ROW_LDS_ELEMS = D_HEAD_SIZE + 16 / sizeof(D_OUT);
    static constexpr int Q_ROPE_SEG_OFF  = 32 * 1024;
    static constexpr int QO_SEG_BYTES    = 64 * 1024;
    static constexpr int QO_STAGE_BYTES  = NUM_WARPS * QO_SEG_BYTES;

    static constexpr size_t smem_size_bytes() { return (size_t)(KV_STAGE_BYTES > QO_STAGE_BYTES ? KV_STAGE_BYTES : QO_STAGE_BYTES); }
};

template<int Q_TILE_SIZE_ = 32,
         int KV_TILE_SIZE_ = 64,
         int NUM_WARPS_ = 4,
         typename D_NOPE_ = fp8_t,
         typename D_ROPE_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct pa_32mx1_16nx4_fp8_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 32;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr int MXSCL_BLOCK_SIZE = 32;

    static constexpr int D_NOPE_SIZE = 448;
    static constexpr int D_NOPE_PADDED_SIZE = 512;
    static constexpr int D_ROPE_SIZE = 64;
    static constexpr int D_HEAD_SIZE = D_NOPE_SIZE + D_ROPE_SIZE;

    using D_NOPE = D_NOPE_;
    using D_ROPE = D_ROPE_;
    using D_ATTN = D_NOPE_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K_NOPE = 128;
    static constexpr int W_K_ROPE = 32;

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM0_NOPE_E_K = D_NOPE_PADDED_SIZE / W_K_NOPE;
    static constexpr int GEMM0_ROPE_E_K = D_ROPE_SIZE / W_K_ROPE;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = D_HEAD_SIZE / (W_N * T_N);
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K_ROPE;

    static constexpr int VEC_NOPE  = 16;
    static constexpr int VEC_ROPE  = 8;
    static constexpr int VEC_MXSCL = 4;
    static constexpr int VEC_O     = 8;

    static constexpr int K_NOPE_ROW_LDS_BYTES = D_NOPE_PADDED_SIZE * sizeof(D_NOPE) + 16;
    static constexpr int K_ROPE_ROW_LDS_BYTES = D_ROPE_SIZE * sizeof(D_ROPE) + 16;
    static constexpr int V_ROW_LDS_BYTES      = D_HEAD_SIZE * sizeof(D_ROPE) + 16;
    static constexpr int K_NOPE_ROW_LDS_ELEMS = K_NOPE_ROW_LDS_BYTES / sizeof(D_NOPE);
    static constexpr int K_ROPE_ROW_LDS_ELEMS = K_ROPE_ROW_LDS_BYTES / sizeof(D_ROPE);
    static constexpr int V_ROW_LDS_ELEMS      = V_ROW_LDS_BYTES / sizeof(D_ROPE);

    static constexpr int ROWS_PER_WAVE      = KV_TILE_SIZE / NUM_WARPS;
    static constexpr int INDICES_PER_TDM    = 8;
    static constexpr int TDM_LOADS_PER_WAVE = ROWS_PER_WAVE / INDICES_PER_TDM;
    static constexpr int TDM_OPS_PER_TILE   = 2 * TDM_LOADS_PER_WAVE;

    static constexpr int K_NOPE_TILE_BYTES = KV_TILE_SIZE * K_NOPE_ROW_LDS_BYTES;
    static constexpr int K_ROPE_TILE_OFF   = K_NOPE_TILE_BYTES;
    static constexpr int K_ROPE_TILE_BYTES = KV_TILE_SIZE * K_ROPE_ROW_LDS_BYTES;
    static constexpr int K_BUF_BYTES       = K_NOPE_TILE_BYTES + K_ROPE_TILE_BYTES;
    static constexpr int NUM_KV_BUFS       = 2;
    static constexpr int K_LDS_BYTES       = NUM_KV_BUFS * K_BUF_BYTES;

    static constexpr int NUM_V_BUFS    = 2;
    static constexpr int V_BUF_BYTES   = KV_TILE_SIZE * V_ROW_LDS_BYTES;
    static constexpr int V_LDS_OFF     = K_LDS_BYTES;
    static constexpr int V_LDS_BYTES   = NUM_V_BUFS * V_BUF_BYTES;
    static constexpr int V_NOPE_BLOCKS = D_NOPE_SIZE / MXSCL_BLOCK_SIZE;
    static constexpr int v_ds_load_insts = (GEMM1_E_N * GEMM1_E_K * W_N * W_K_ROPE) / (WARP_SIZE * VEC_ROPE);

    static constexpr int QO_ROWS_PER_WAVE     = Q_TILE_SIZE / NUM_WARPS;
    static constexpr int Q_NOPE_ROW_LDS_ELEMS = K_NOPE_ROW_LDS_ELEMS;
    static constexpr int Q_ROPE_ROW_LDS_ELEMS = K_ROPE_ROW_LDS_ELEMS;
    static constexpr int O_ROW_LDS_ELEMS      = D_HEAD_SIZE + 16 / sizeof(D_OUT);
    static constexpr int Q_ROPE_TILE_OFF  = (Q_TILE_SIZE * K_NOPE_ROW_LDS_BYTES + 255) / 256 * 256;
    static constexpr int Q_TILE_LDS_BYTES = Q_ROPE_TILE_OFF + Q_TILE_SIZE * K_ROPE_ROW_LDS_BYTES;
    static constexpr int O_TILE_LDS_BYTES = Q_TILE_SIZE * O_ROW_LDS_ELEMS * sizeof(D_OUT);
    static constexpr int QO_LDS_BYTES     = Q_TILE_LDS_BYTES > O_TILE_LDS_BYTES ? Q_TILE_LDS_BYTES : O_TILE_LDS_BYTES;

    static constexpr int ML_SLOT_ELEMS = GEMM0_E_M * W_M * T_N;
    static constexpr int ML_SLOT_BYTES = ML_SLOT_ELEMS * (int)sizeof(D_ACC);
    static constexpr int M_LDS_OFF     = V_LDS_OFF + V_LDS_BYTES;
    static constexpr int L_LDS_OFF     = M_LDS_OFF + ML_SLOT_BYTES;

    static constexpr int VEC_P         = 8;
    static constexpr int P_CHUNKS      = GEMM0_E_N * W_M * W_N / (WARP_SIZE * VEC_P);
    static constexpr int P_BLOCK_ELEMS = WARP_SIZE * VEC_P;
    static constexpr int P_NUM_BLOCKS  = GEMM0_E_M * T_N * P_CHUNKS;
    static constexpr int P_LDS_OFF     = L_LDS_OFF + ML_SLOT_BYTES;
    static constexpr int P_LDS_BYTES   = P_NUM_BLOCKS * P_BLOCK_ELEMS * (int)sizeof(D_ROPE);

    static constexpr int ACCUM_LDS_BYTES = P_LDS_OFF + P_LDS_BYTES;

    static constexpr size_t smem_size_bytes() {
        return (size_t)(ACCUM_LDS_BYTES > QO_LDS_BYTES ? ACCUM_LDS_BYTES : QO_LDS_BYTES);
    }
};

template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 64,
         int NUM_WARPS_ = 4,
         typename D_NOPE_ = fp8_t,
         typename D_ROPE_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct pa_16mx1_16nx4_fp8_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 32;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr int MXSCL_BLOCK_SIZE = 32;

    static constexpr int D_NOPE_SIZE = 448;
    static constexpr int D_NOPE_PADDED_SIZE = 512;
    static constexpr int D_ROPE_SIZE = 64;
    static constexpr int D_HEAD_SIZE = D_NOPE_SIZE + D_ROPE_SIZE;

    using D_NOPE = D_NOPE_;
    using D_ROPE = D_ROPE_;
    using D_ATTN = D_NOPE_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K_NOPE = 128;
    static constexpr int W_K_ROPE = 32;

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM0_NOPE_E_K = D_NOPE_PADDED_SIZE / W_K_NOPE;
    static constexpr int GEMM0_ROPE_E_K = D_ROPE_SIZE / W_K_ROPE;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = D_HEAD_SIZE / (W_N * T_N);
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K_ROPE;

    static constexpr int VEC_NOPE  = 16;
    static constexpr int VEC_ROPE  = 8;
    static constexpr int VEC_MXSCL = 4;
    static constexpr int VEC_O     = 8;

    static constexpr int K_NOPE_ROW_LDS_BYTES = D_NOPE_PADDED_SIZE * sizeof(D_NOPE) + 16;
    static constexpr int K_ROPE_ROW_LDS_BYTES = D_ROPE_SIZE * sizeof(D_ROPE) + 16;
    static constexpr int V_ROW_LDS_BYTES      = D_HEAD_SIZE * sizeof(D_ROPE) + 16;
    static constexpr int K_NOPE_ROW_LDS_ELEMS = K_NOPE_ROW_LDS_BYTES / sizeof(D_NOPE);
    static constexpr int K_ROPE_ROW_LDS_ELEMS = K_ROPE_ROW_LDS_BYTES / sizeof(D_ROPE);
    static constexpr int V_ROW_LDS_ELEMS      = V_ROW_LDS_BYTES / sizeof(D_ROPE);

    static constexpr int ROWS_PER_WAVE      = KV_TILE_SIZE / NUM_WARPS;
    static constexpr int INDICES_PER_TDM    = 8;
    static constexpr int TDM_LOADS_PER_WAVE = ROWS_PER_WAVE / INDICES_PER_TDM;
    static constexpr int TDM_OPS_PER_TILE   = 2 * TDM_LOADS_PER_WAVE;

    static constexpr int K_NOPE_TILE_BYTES = KV_TILE_SIZE * K_NOPE_ROW_LDS_BYTES;
    static constexpr int K_ROPE_TILE_OFF   = K_NOPE_TILE_BYTES;
    static constexpr int K_ROPE_TILE_BYTES = KV_TILE_SIZE * K_ROPE_ROW_LDS_BYTES;
    static constexpr int K_BUF_BYTES       = K_NOPE_TILE_BYTES + K_ROPE_TILE_BYTES;
    static constexpr int NUM_KV_BUFS       = 2;
    static constexpr int K_LDS_BYTES       = NUM_KV_BUFS * K_BUF_BYTES;

    static constexpr int NUM_V_BUFS    = 1;
    static constexpr int V_BUF_BYTES   = KV_TILE_SIZE * V_ROW_LDS_BYTES;
    static constexpr int V_LDS_OFF     = K_LDS_BYTES;
    static constexpr int V_LDS_BYTES   = NUM_V_BUFS * V_BUF_BYTES;
    static constexpr int V_NOPE_BLOCKS = D_NOPE_SIZE / MXSCL_BLOCK_SIZE;
    static constexpr int v_ds_load_insts = (GEMM1_E_N * GEMM1_E_K * W_N * W_K_ROPE) / (WARP_SIZE * VEC_ROPE);

    static constexpr int QO_ROWS_PER_WAVE     = Q_TILE_SIZE / NUM_WARPS;
    static constexpr int Q_NOPE_ROW_LDS_ELEMS = K_NOPE_ROW_LDS_ELEMS;
    static constexpr int Q_ROPE_ROW_LDS_ELEMS = K_ROPE_ROW_LDS_ELEMS;
    static constexpr int O_ROW_LDS_ELEMS      = D_HEAD_SIZE + 16 / sizeof(D_OUT);
    static constexpr int Q_ROPE_TILE_OFF  = (Q_TILE_SIZE * K_NOPE_ROW_LDS_BYTES + 255) / 256 * 256;
    static constexpr int Q_TILE_LDS_BYTES = Q_ROPE_TILE_OFF + Q_TILE_SIZE * K_ROPE_ROW_LDS_BYTES;
    static constexpr int O_TILE_LDS_BYTES = Q_TILE_SIZE * O_ROW_LDS_ELEMS * sizeof(D_OUT);
    static constexpr int QO_LDS_BYTES     = Q_TILE_LDS_BYTES > O_TILE_LDS_BYTES ? Q_TILE_LDS_BYTES : O_TILE_LDS_BYTES;

    static constexpr int ML_LDS_OFF   = V_LDS_OFF + V_LDS_BYTES;
    static constexpr int ML_LDS_BYTES = 2 * T_N * W_M * sizeof(D_ACC);

    static constexpr int P_LDS_OFF     = ML_LDS_OFF + ML_LDS_BYTES;
    static constexpr int P_BLOCK_ELEMS = W_M * W_N;
    static constexpr int P_LDS_BYTES   = NUM_WARPS * P_BLOCK_ELEMS * sizeof(D_ROPE);

    static constexpr int ACCUM_LDS_BYTES = P_LDS_OFF + P_LDS_BYTES;

    static constexpr size_t smem_size_bytes() {
        return (size_t)(ACCUM_LDS_BYTES > QO_LDS_BYTES ? ACCUM_LDS_BYTES : QO_LDS_BYTES);
    }
};
