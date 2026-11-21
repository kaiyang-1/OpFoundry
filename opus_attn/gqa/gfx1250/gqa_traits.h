// Configuration traits for the gfx1250 (wave32 / WMMA) GQA forward kernel.
#pragma once

#include "common/gqa_kargs.h"

// One tiling: 8 waves of 32 Q rows, KV tiles of 64 rows, D = 128.
template<int Q_TILE_SIZE_ = 32,
         int KV_TILE_SIZE_ = 64,
         int D_TILE_SIZE_ = 128,
         int NUM_WARPS_ = 8,
         bool CAUSAL_ = false>
struct opus_gqa_traits {
    static_assert(Q_TILE_SIZE_ == 32 && KV_TILE_SIZE_ == 64 && D_TILE_SIZE_ == 128 && NUM_WARPS_ == 8,
                  "the gfx1250 kernel has one tiling");

    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;  // Q rows per wave
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE = D_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr bool CAUSAL = CAUSAL_;

    static constexpr int WARP_SIZE = 32;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr int WG_Q_ROWS = Q_TILE_SIZE * NUM_WARPS;

    // What the host may run: any lengths, sq != sk, LSE and sink.
    using kargs_type = opus_gqa_varlen_kargs;
    static constexpr bool VARLEN = true;
    static constexpr bool HAS_LSE = true;
    static constexpr bool HAS_SINK = true;
    static constexpr int SEQ_ALIGN = 1;
    static constexpr int MIN_KV_TILES = 0;
};
