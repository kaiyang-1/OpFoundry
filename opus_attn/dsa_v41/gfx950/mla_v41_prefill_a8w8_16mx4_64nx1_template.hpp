#pragma once

#include <opus/opus.hpp>
#include "common/mla_v41_defs.h"
#include "mla_v41_global_load.hpp"
#include <bit>

using opus::operator""_I;

namespace opus_mla_v41_prefill_a8w8_16mx4_64nx1 {
// ------------------------------------------------------------------------------ shape factors

template<class T> constexpr int LANE_GRPS   = T::WARP_SIZE / T::W_N;
template<class T> constexpr int GRP_PER_BLK = T::D_SCALE_BLOCK / T::VEC_KV;
template<class T> constexpr int K_PIECES    = T::W_N * T::W_K_QK / T::WARP_SIZE / T::VEC_KV;
template<class T> constexpr int N_TILES_SUB = T::GEMM0_E_N / T::smem_n_sub_tile_rpt;
template<class T> constexpr int V_N_TILES   = T::D_128B_V_SIZE / T::W_N;
template<class T> constexpr int VEC_O2      = 2 * T::VEC_O;

// ------------------------------------------------------------------------------------ lds map

template<class T, int SLOT> constexpr int k_slot_base     = SLOT * T::smem_kv_bytes;
template<class T, int SLOT> constexpr int scale_slot_base = 2 * T::smem_kv_bytes + SLOT * T::smem_scale_bytes;
template<class T>           constexpr int v_base          = 2 * T::smem_kv_bytes + 2 * T::smem_scale_bytes;

template<class T, int I>          constexpr int kv_sub_tile_off = I * T::NUM_WARPS * T::smem_d_rpt * T::smem_brick;
template<class T, int EK>         constexpr int k_step_off      = EK * T::D_SCALE_BLOCK;
template<class T, int KS, int DV> constexpr int v_step_off      = KS * T::W_K_PV * T::smem_v_row + DV * T::D_128B_V_SIZE;

// ------------------------------------------------------------------------- global -> lds (async)

template<class T>
__device__ inline auto make_layout_kv_indices(int warp_id, int lane_id) {
    constexpr int threads_d = T::D_128B_SIZE / T::VEC_KV;

    constexpr auto kv_indices_shape = opus::make_tuple(
        opus::number<T::smem_n_sub_tile_rpt>{},
        opus::number<T::smem_n_per_wave>{},
        opus::number<T::NUM_WARPS>{},
        1_I);

    constexpr auto kv_indices_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        kv_indices_shape,
        opus::unfold_x_stride(kv_indices_dim, kv_indices_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(kv_indices_dim, opus::tuple{lane_id / threads_d, warp_id}));
}

template<class T>
__device__ inline auto make_layout_gkv(int lane_id) {
    constexpr int threads_d = T::D_128B_SIZE / T::VEC_KV;

    constexpr auto gkv_shape = opus::make_tuple(
        opus::number<T::smem_d_rpt>{},
        opus::number<threads_d>{},
        opus::number<T::VEC_KV>{});

    constexpr auto gkv_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        gkv_shape,
        opus::unfold_x_stride(gkv_dim, gkv_shape, opus::tuple{opus::number<T::D_128B_SIZE>{}, 1_I}),
        opus::unfold_p_coord(gkv_dim, opus::tuple{lane_id % threads_d}));
}

template<class T>
__device__ inline auto make_layout_skv(int warp_id) {
    constexpr auto skv_shape = opus::make_tuple(
        opus::number<T::smem_d_rpt>{},
        opus::number<T::NUM_WARPS>{},
        opus::number<T::VEC_KV>{});

    constexpr auto skv_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        skv_shape,
        opus::unfold_x_stride(skv_dim, skv_shape, opus::tuple{opus::number<T::smem_brick>{}, 1_I}),
        opus::unfold_p_coord(skv_dim, opus::tuple{warp_id}));
}

template<class T>
__device__ inline auto make_layout_gk_scale(int lane_id) {
    constexpr int threads_d = T::D_128B_SIZE / T::VEC_KV;
    constexpr int threads_s = T::D_SCALE_SIZE / T::VEC_SCALE;

    constexpr auto gs_shape = opus::make_tuple(
        opus::number<threads_d / threads_s>{},
        opus::number<threads_s>{},
        opus::number<T::VEC_SCALE>{});

    constexpr auto gs_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        gs_shape,
        opus::unfold_x_stride(gs_dim, gs_shape, opus::tuple{0_I, 1_I}),
        opus::unfold_p_coord(gs_dim, opus::tuple{(lane_id % threads_d) / threads_s, lane_id % threads_s}));
}

template<class T>
__device__ inline auto make_layout_sk_scale(int warp_id) {
    constexpr auto ss_shape = opus::make_tuple(
        opus::number<T::smem_n_sub_tile_rpt>{},
        opus::number<T::NUM_WARPS>{},
        opus::number<T::VEC_SCALE>{});

    constexpr auto ss_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        ss_shape,
        opus::unfold_x_stride(ss_dim, ss_shape, opus::tuple{opus::number<T::smem_scale_brick>{}, 1_I}),
        opus::unfold_p_coord(ss_dim, opus::tuple{warp_id}));
}

// ------------------------------------------------------------------------- global -> registers

template<class T>
__device__ inline auto make_layout_q(int warp_id, int lane_id, int stride_q_h) {
    constexpr auto q_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::GEMM0_E_K>{},
        opus::number<K_PIECES<T>>{},
        opus::number<LANE_GRPS<T> / GRP_PER_BLK<T>>{},
        opus::number<GRP_PER_BLK<T>>{},
        opus::number<T::VEC_Q>{});

    constexpr auto q_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int lane_g = lane_id / T::W_M;

    return opus::make_layout(
        q_shape,
        opus::unfold_x_stride(q_dim, q_shape, opus::tuple{stride_q_h,
                                                          opus::number<T::D_SCALE_BLOCK>{},
                                                          opus::number<T::D_128B_SIZE>{},
                                                          1_I}),
        opus::unfold_p_coord(q_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_g / GRP_PER_BLK<T>, lane_g % GRP_PER_BLK<T>}));
}

template<class T>
__device__ inline auto make_layout_q_scale(int warp_id, int lane_id, int stride_q_scale_h) {
    constexpr auto qs_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<LANE_GRPS<T>>{},
        opus::number<T::VEC_SCALE>{});

    constexpr auto qs_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        qs_shape,
        opus::unfold_x_stride(qs_dim, qs_shape, opus::tuple{stride_q_scale_h, 1_I}),
        opus::unfold_p_coord(qs_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

// --------------------------------------------------------------------------- lds -> registers

template<class T>
__device__ inline auto make_layout_rk(int lane_id) {
    constexpr auto rk_shape = opus::make_tuple(
        opus::number<N_TILES_SUB<T>>{},
        opus::number<T::W_N / T::NUM_WARPS>{},
        opus::number<K_PIECES<T>>{},
        opus::number<LANE_GRPS<T> / GRP_PER_BLK<T>>{},
        opus::number<T::NUM_WARPS>{},
        opus::number<GRP_PER_BLK<T>>{},
        opus::number<T::VEC_KV>{});

    constexpr auto rk_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int lane_n = lane_id % T::W_N;
    const int lane_g = lane_id / T::W_N;

    return opus::make_layout(
        rk_shape,
        opus::unfold_x_stride(rk_dim, rk_shape, opus::tuple{opus::number<T::D_128B_SIZE>{},
                                                            opus::number<T::NUM_WARPS * T::smem_brick>{},
                                                            opus::number<T::smem_brick>{},
                                                            1_I}),
        opus::unfold_p_coord(rk_dim, opus::tuple{lane_n / T::NUM_WARPS, lane_g / GRP_PER_BLK<T>,
                                                 lane_n % T::NUM_WARPS, lane_g % GRP_PER_BLK<T>}));
}

template<class T>
__device__ inline auto make_layout_rk_scale(int lane_id) {
    constexpr auto rs_shape = opus::make_tuple(
        opus::number<T::smem_n_sub_tile_rpt>{},
        opus::number<T::NUM_WARPS>{},
        opus::number<N_TILES_SUB<T>>{},
        opus::number<T::W_N / T::NUM_WARPS>{},
        opus::number<LANE_GRPS<T>>{},
        opus::number<T::VEC_SCALE>{});

    constexpr auto rs_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int lane_n = lane_id % T::W_N;

    return opus::make_layout(
        rs_shape,
        opus::unfold_x_stride(rs_dim, rs_shape, opus::tuple{opus::number<T::smem_scale_brick>{},
                                                            opus::number<T::smem_scale_row>{},
                                                            1_I}),
        opus::unfold_p_coord(rs_dim, opus::tuple{lane_n % T::NUM_WARPS, lane_n / T::NUM_WARPS, lane_id / T::W_N}));
}

template<class T>
__device__ inline auto make_layout_rv(int lane_id) {
    constexpr int lane_per_grp = 16;
    constexpr int lane_lo = 4;
    constexpr int lane_hi = lane_per_grp / lane_lo;

    constexpr int num_grps = T::WARP_SIZE / lane_per_grp;
    constexpr int grp_n = T::W_N / (lane_lo * T::VEC_TR_V);
    constexpr int grp_k = num_grps / grp_n;

    constexpr auto rv_shape = opus::make_tuple(
        opus::number<V_N_TILES<T>>{},
        opus::number<T::W_K_PV / (lane_hi * grp_k)>{},
        opus::number<grp_k>{},
        opus::number<lane_hi>{},
        opus::number<grp_n>{},
        opus::number<lane_lo>{},
        opus::number<T::VEC_TR_V>{});

    constexpr auto rv_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int grp_id = lane_id / lane_per_grp;
    const int lane_in_grp = lane_id % lane_per_grp;

    return opus::make_layout(
        rv_shape,
        opus::unfold_x_stride(rv_dim, rv_shape, opus::tuple{opus::number<grp_n * lane_lo * T::VEC_TR_V>{},
                                                            opus::number<T::smem_v_row>{},
                                                            1_I}),
        opus::unfold_p_coord(rv_dim, opus::tuple{grp_id / grp_n, lane_in_grp / lane_lo, grp_id % grp_n, lane_in_grp % lane_lo}));
}

// ------------------------------------------------------------------------------ v dequant

template<class T>
__device__ inline auto make_layout_rv_dequant(int warp_id, int lane_id) {
    constexpr int threads_d = T::D_128B_SIZE / T::VEC_DEQ;

    constexpr auto rd_shape = opus::make_tuple(
        opus::number<T::smem_n_sub_tile_rpt>{},
        opus::number<T::smem_d_rpt>{},
        opus::number<T::NUM_WARPS>{},
        opus::number<T::smem_n_per_wave / T::deq_rows_per_inst>{},
        opus::number<T::deq_rows_per_inst>{},
        opus::number<threads_d>{},
        opus::number<T::VEC_DEQ>{});

    constexpr auto rd_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        rd_shape,
        opus::unfold_x_stride(rd_dim, rd_shape, opus::tuple{opus::number<T::smem_brick>{}, opus::number<T::D_128B_SIZE>{}, 1_I}),
        opus::unfold_p_coord(rd_dim, opus::tuple{warp_id, lane_id / threads_d, lane_id % threads_d}));
}

template<class T>
__device__ inline auto make_layout_rs_dequant(int warp_id, int lane_id) {
    constexpr int threads_d = T::D_128B_SIZE / T::VEC_DEQ;
    constexpr int blks_d = T::D_128B_SIZE / T::D_SCALE_BLOCK;

    constexpr auto rsd_shape = opus::make_tuple(
        opus::number<T::smem_n_sub_tile_rpt>{},
        opus::number<T::NUM_WARPS>{},
        opus::number<T::smem_d_rpt>{},
        opus::number<T::smem_n_per_wave / T::deq_rows_per_inst>{},
        opus::number<T::deq_rows_per_inst>{},
        opus::number<blks_d>{},
        1_I);

    constexpr auto rsd_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        rsd_shape,
        opus::unfold_x_stride(rsd_dim, rsd_shape, opus::tuple{opus::number<T::smem_scale_brick>{},
                                                              opus::number<blks_d>{},
                                                              opus::number<T::smem_scale_row>{},
                                                              1_I}),
        opus::unfold_p_coord(rsd_dim, opus::tuple{warp_id, lane_id / threads_d, (lane_id % threads_d) / (T::D_SCALE_BLOCK / T::VEC_DEQ)}));
}

template<class T>
__device__ inline auto make_layout_sv_dequant(int warp_id, int lane_id) {
    constexpr int threads_d = T::D_128B_SIZE / T::VEC_DEQ;

    constexpr auto sd_shape = opus::make_tuple(
        opus::number<T::smem_n_sub_tile_rpt>{},
        opus::number<T::smem_d_rpt>{},
        opus::number<T::smem_n_per_wave / T::deq_rows_per_inst>{},
        opus::number<T::deq_rows_per_inst>{},
        opus::number<T::NUM_WARPS>{},
        opus::number<threads_d>{},
        opus::number<T::VEC_DEQ>{});

    constexpr auto sd_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        sd_shape,
        opus::unfold_x_stride(sd_dim, sd_shape, opus::tuple{opus::number<T::smem_n_sub_tile * T::smem_v_row>{},
                                                            opus::number<T::D_128B_SIZE>{},
                                                            opus::number<T::smem_v_row>{},
                                                            1_I}),
        opus::unfold_p_coord(sd_dim, opus::tuple{lane_id / threads_d, warp_id, lane_id % threads_d}));
}

// ------------------------------------------------------------------------- registers -> global

template<class T>
__device__ inline auto make_layout_o_packed(int warp_id, int lane_id, int stride_o_h) {
    constexpr auto o_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::D_HEAD_SIZE / (2 * T::W_N)>{},
        opus::number<2>{},
        opus::number<2>{},
        opus::number<VEC_O2<T>>{});

    constexpr auto o_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_g = lane_id / T::W_M;

    return opus::make_layout(
        o_block_shape,
        opus::unfold_x_stride(o_block_dim, o_block_shape, opus::tuple{stride_o_h, 1_I}),
        opus::unfold_p_coord(o_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_g % 2, lane_g / 2}));
}

} // namespace opus_mla_v41_prefill_a8w8_16mx4_64nx1

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1)
void opus_mla_v41_prefill_a8w8_16mx4_64nx1_kernel(opus_mla_v41_prefill_kargs kargs) {
}
