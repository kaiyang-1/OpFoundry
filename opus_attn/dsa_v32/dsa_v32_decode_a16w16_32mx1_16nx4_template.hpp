#pragma once

#include <opus/opus.hpp>
#include "defs.h"

namespace dsa_v32_decode_a16w16_32mx1_16nx4 {

template<class T> using v_o_t  = opus::vector_t<typename T::D_ACC, T::O_ACC_ELEMS>;
template<class T> using v_ml_t = opus::vector_t<typename T::D_ACC, T::ML_ELEMS>;

template<class Traits> __device__ auto make_layout_q(int warp_id, int lane_id);
template<class Traits> __device__ auto make_layout_kv_indices(int warp_id, int lane_id);
template<class Traits> __device__ auto make_layout_gkv(int warp_id, int lane_id);
template<class Traits> __device__ auto make_layout_skv(int warp_id);
template<class Traits> __device__ auto make_layout_rk(int n_idx, int lane_id);
template<class Traits> __device__ auto make_layout_rv(int n_idx, int lane_id);
template<class Traits> __device__ auto make_layout_o(int n_idx, int lane_id, int stride_o_h);

template<class Traits, class VS>
__device__ v_ml_t<Traits> attn_row_max(const VS& v_s);
template<class Traits, class VS>
__device__ v_ml_t<Traits> attn_row_sum(const VS& v_s);
template<class Traits, class VS>
__device__ void attn_mask_oob_kv_tile(VS& v_s, int valid_kv_len, int kv_tile_idx,
                                      int n_idx, opus::u32_t neg_inf_v);

template<class Traits>
__device__ void reduce_ml_across_waves(char* smem_ml, int n_idx, int lane_id,
                                       v_ml_t<Traits>& m_row, v_ml_t<Traits>& l_row,
                                       v_ml_t<Traits>& rescale);

template<class Traits>
__device__ void attention_tiles(const dsa_v32_a16w16_kargs& kargs,
                                int page_idx_begin, int valid_kv_len,
                                int tile_begin, int tile_end,
                                char* smem_kv,
                                v_o_t<Traits>& v_o,
                                v_ml_t<Traits>& m_row,
                                v_ml_t<Traits>& l_row,
                                typename Traits::D_ACC temperature_scale);

template<class Traits>
__device__ void decode_one_req(const dsa_v32_a16w16_kargs& kargs, int batch_idx, int h_block_idx,
                               int page_idx_begin, int valid_kv_len,
                               int tile_begin, int tile_end, int slot,
                               char* smem_kv, char* smem_ml,
                               typename Traits::D_ACC temperature_scale);

}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1)
void dsa_v32_decode_a16w16_32mx1_16nx4_kernel(dsa_v32_a16w16_kargs kargs) {
    using namespace opus;
    using namespace dsa_v32_decode_a16w16_32mx1_16nx4;
    using T = opus::remove_cvref_t<Traits>;

    const int part = block_id_x();
    const int h_block_idx = block_id_y();
    if (part >= kargs.num_parts) return;

    const DsaSchedMeta meta = kargs.sched_meta[part];
    if (meta.begin_req_idx >= kargs.B) return;

    __shared__ char smem[T::smem_bytes()];
    char* smem_kv = smem;
    char* smem_ml = smem + T::smem_kv_bytes;

    constexpr float LOG2_E = 1.44269504089f;
    const float temperature_scale = kargs.softmax_scale * LOG2_E;

    for (int req = meta.begin_req_idx; req <= meta.end_req_idx; ++req) {
        const int page_idx_begin = __builtin_amdgcn_readfirstlane(kargs.kv_indptr[req]);
        const int valid_kv_len   = __builtin_amdgcn_readfirstlane(kargs.kv_indptr[req + 1]) - page_idx_begin;
        const int num_tiles      = ceil_div(valid_kv_len, T::KV_TILE_SIZE);

        const int tile_begin = (req == meta.begin_req_idx) ? meta.begin_tile_idx : 0;
        const int tile_end   = (req == meta.end_req_idx)   ? meta.end_tile_idx   : num_tiles;

        const int nsplit = kargs.num_splits[req + 1] - kargs.num_splits[req];
        const bool is_no_split = (nsplit <= 1);
        const int n_split_idx = (req == meta.begin_req_idx) ? meta.begin_split_idx : 0;
        const int slot = is_no_split ? -1 : (kargs.num_splits[req] + n_split_idx);

        decode_one_req<T>(kargs, req, h_block_idx, page_idx_begin, valid_kv_len,
                          tile_begin, tile_end, slot, smem_kv, smem_ml, temperature_scale);
    }
}
