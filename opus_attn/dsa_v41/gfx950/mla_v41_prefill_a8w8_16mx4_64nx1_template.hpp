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

// ----------------------------------------------------------------- register budget and pipeline

constexpr int Q_AGPR_BASE = 0;
template<class T> constexpr int Q_LOAD_INSTS     = T::Q_TILE_SIZE * T::D_HEAD_SIZE / T::WARP_SIZE / T::VEC_Q;
template<class T> constexpr int Q_AGPR_PER_LOAD  = T::VEC_Q * sizeof(typename T::D_ATTN) / 4;
template<class T> constexpr int K_AGPR_BASE      = Q_AGPR_BASE + Q_LOAD_INSTS<T> * Q_AGPR_PER_LOAD<T>;
template<class T> constexpr int K_CHUNKS         = N_TILES_SUB<T> * K_PIECES<T>;
template<class T> constexpr int K_AGPR_PER_CHUNK = T::VEC_KV * sizeof(typename T::D_ATTN) / 4;
template<class T> constexpr int K_AGPR_PER_BUF   = K_CHUNKS<T> * K_AGPR_PER_CHUNK<T>;

constexpr int O_VGPR_BASE = 0;
constexpr int S_VGPR_BASE = 128;
constexpr int P_VGPR_BASE = S_VGPR_BASE;

constexpr int K_DEPTH = 4;
constexpr int V_DEPTH = 4;

template<class T> constexpr int ACC_PER_TILE  = T::W_M * T::W_N / T::WARP_SIZE;
template<class T> constexpr int S_ELEMS       = T::GEMM0_E_N * ACC_PER_TILE<T>;
template<class T> constexpr int O_ELEMS       = T::Q_TILE_SIZE * T::D_HEAD_SIZE / T::WARP_SIZE;
template<class T> constexpr int QK_STEPS      = T::smem_n_sub_tile_rpt * T::GEMM0_E_K;
template<class T> constexpr int DV_STEPS      = T::D_HEAD_SIZE / T::D_128B_V_SIZE;
template<class T> constexpr int PV_STEPS      = T::GEMM1_E_K * DV_STEPS<T>;
template<class T> constexpr int V_TR_PER_TILE = T::W_N * T::W_K_PV / T::WARP_SIZE / T::VEC_TR_V;
template<class T> constexpr int V_TR_PER_STEP = V_N_TILES<T> * V_TR_PER_TILE<T>;
template<class T> constexpr int DEQ_PER_STEP  = T::deq_insts / QK_STEPS<T>;
template<class T> constexpr int DEQ_RING      = 2 * DEQ_PER_STEP<T>;

template<class T> using v_q_t = opus::array<opus::vector_t<typename T::D_ATTN, T::VEC_Q>, Q_LOAD_INSTS<T>>;
template<class T> using v_k_t = opus::vector_t<typename T::D_ATTN, T::VEC_KV>;
template<class T> using v_v_t = opus::vector_t<typename T::D_PV, T::W_N * T::W_K_PV / T::WARP_SIZE>;
template<class T> using v_s_t = opus::vector_t<typename T::D_ACC, S_ELEMS<T>>;
template<class T> using v_p_t = opus::vector_t<typename T::D_PV, S_ELEMS<T>>;
template<class T> using v_o_t = opus::vector_t<typename T::D_ACC, O_ELEMS<T>>;

// lgkmcnt is a 4-bit field; a larger count saturates into a stricter wait.
template<int N> __device__ inline void wait_lgkmcnt() { opus::s_waitcnt_lgkmcnt(opus::number<(N < 15 ? N : 15)>{}); }

// ------------------------------------------------------------------------------------ lds map

constexpr int Q_SLOT = 1;

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

// ------------------------------------------------------------------------------- transfers

template<class T, int SLOT, class US>
__device__ inline auto kv_lds_bases(char* smem, const US& u_skv) {
    auto s_off = opus::layout_to_offsets<T::VEC_KV>(u_skv);
    opus::array<typename T::D_ATTN*, T::smem_n_sub_tile_rpt> bases;
    opus::static_for<T::smem_n_sub_tile_rpt>([&](auto ig) {
        auto* base = reinterpret_cast<typename T::D_ATTN*>(smem + k_slot_base<T, SLOT>)
                   + kv_sub_tile_off<T, ig.value> + s_off[0];
        asm volatile("" : "+s"(base));
        bases[ig.value] = base;
    });
    return bases;
}

template<class T, class UG, class VP>
__device__ inline auto kv_gmem_bases(const typename T::D_ATTN* p_kv, const UG& u_gkv,
                                     const VP& kv_pages, int stride_kv_page) {
    opus::array<const typename T::D_ATTN*, T::smem_n_sub_tile_rpt> bases;
    auto g_off = opus::layout_to_offsets<T::VEC_KV>(u_gkv);
    opus::static_for<T::smem_n_sub_tile_rpt>([&](auto ig) {
        bases[ig.value] = p_kv + static_cast<int64_t>(kv_pages[ig.value]) * stride_kv_page + g_off[0];
    });
    return bases;
}

template<class T, int IDX, class GB, class LB, class US>
__device__ inline void async_load_kv_inst(const GB& gmem_bases, const LB& lds_bases, const US&) {
    constexpr int ig = IDX / T::smem_d_rpt;
    constexpr int id = IDX % T::smem_d_rpt;
    constexpr int ELEM = (int)sizeof(typename T::D_ATTN);
    constexpr auto s_imm = opus::layout_imm_offsets_v<opus::remove_cvref_t<US>, T::VEC_KV>;
    constexpr int g_imm  = id * T::D_128B_SIZE * ELEM;
    constexpr int m0_imm = (s_imm[id] - s_imm[0]) * ELEM - g_imm;
    global_load_lds<T::VEC_KV, m0_imm, g_imm>(gmem_bases[ig], lds_bases[ig]);
}

template<class T, class GB, class LB, class US>
__device__ inline void async_load_kv_tile(const GB& gmem_bases, const LB& lds_bases, const US& u_skv) {
    opus::static_for<T::kv_async_load_insts>([&](auto i) {
        async_load_kv_inst<T, i.value>(gmem_bases, lds_bases, u_skv);
    });
}

template<class T, int SLOT, class US>
__device__ inline auto scale_lds_bases(char* smem, const US& u_ss) {
    auto s_off = opus::layout_to_offsets<T::VEC_SCALE>(u_ss);
    opus::array<typename T::D_ATTN*, T::smem_n_sub_tile_rpt> bases;
    opus::static_for<T::smem_n_sub_tile_rpt>([&](auto ig) {
        auto* base = reinterpret_cast<typename T::D_ATTN*>(smem + scale_slot_base<T, SLOT>) + s_off[ig.value];
        asm volatile("" : "+s"(base));
        bases[ig.value] = base;
    });
    return bases;
}

template<class T, class UG, class VP>
__device__ inline auto scale_gmem_bases(const typename T::D_ATTN* p_scale, const UG& u_gs,
                                        const VP& kv_pages, int stride_kv_scale_page) {
    opus::array<const typename T::D_ATTN*, T::smem_n_sub_tile_rpt> bases;
    auto g_off = opus::layout_to_offsets<T::VEC_SCALE>(u_gs);
    opus::static_for<T::smem_n_sub_tile_rpt>([&](auto ig) {
        bases[ig.value] = p_scale + static_cast<int64_t>(kv_pages[ig.value]) * stride_kv_scale_page + g_off[0];
    });
    return bases;
}

template<class T, class GB, class LB>
__device__ inline void async_load_scale_tile(const GB& gmem_bases, const LB& lds_bases) {
    opus::static_for<T::scale_async_load_insts>([&](auto ig) {
        global_load_lds<T::VEC_SCALE, 0, 0>(gmem_bases[ig.value], lds_bases[ig.value]);
    });
}

// The WG's Q block has the shape of a K tile, so it rides the K DMA with head indices as pages.
template<class T, class UI, class UG, class US>
__device__ inline void async_load_q_tile(char* smem, const typename T::D_ATTN* p_q, int stride_q_h, int num_heads,
                                         const UI& u_kv_indices, const UG& u_gkv, const US& u_skv) {
    static_assert(T::T_M * T::Q_TILE_SIZE == T::KV_TILE_SIZE);
    auto heads = opus::layout_to_offsets<1>(u_kv_indices);
    opus::static_for<T::smem_n_sub_tile_rpt>([&](auto ig) { heads[ig.value] = min(heads[ig.value], num_heads - 1); });
    async_load_kv_tile<T>(kv_gmem_bases<T>(p_q, u_gkv, heads, stride_q_h), kv_lds_bases<T, Q_SLOT>(smem, u_skv), u_skv);
}

// ------------------------------------------------------------------------------------ q / k / v

template<class T, class UK>
__device__ inline auto load_q(char* smem, const UK& u_rk, int warp_id) {
    using namespace opus;
    constexpr auto imm = layout_imm_offsets_v<remove_cvref_t<UK>, T::VEC_KV>;
    const int warp_off = (warp_id / N_TILES_SUB<T>) * kv_sub_tile_off<T, 1>
                       + (warp_id % N_TILES_SUB<T>) * (imm[K_PIECES<T>] - imm[0]);

    auto s_q = make_smem(reinterpret_cast<typename T::D_ATTN*>(smem + k_slot_base<T, Q_SLOT>) + warp_off);
    auto q_offsets = layout_to_offsets<T::VEC_KV>(u_rk);
    v_q_t<T> v_q;
    static_for<Q_LOAD_INSTS<T>>([&](auto i) {
        constexpr int ek = i.value / K_PIECES<T>;
        constexpr int h  = i.value % K_PIECES<T>;
        [[clang::amdgpu_pin_agpr(Q_AGPR_BASE + i.value * Q_AGPR_PER_LOAD<T>)]]
        v_q[i.value] = load<T::VEC_Q>(s_q, q_offsets[h] + k_step_off<T, ek>);
    });
    return v_q;
}

template<class T, int SLOT, int STEP, int BUF, class UK, class VK>
__device__ inline void load_k_step(char* smem, const UK& u_rk, VK& v_k) {
    using namespace opus;
    constexpr int ns = STEP / T::GEMM0_E_K;
    constexpr int ek = STEP % T::GEMM0_E_K;

    auto s_k = make_smem(reinterpret_cast<typename T::D_ATTN*>(smem + k_slot_base<T, SLOT>));
    auto k_offsets = layout_to_offsets<T::VEC_KV>(u_rk);

    static_for<K_CHUNKS<T>>([&](auto i) {
        [[clang::amdgpu_pin_agpr(K_AGPR_BASE<T> + BUF * K_AGPR_PER_BUF<T> + i.value * K_AGPR_PER_CHUNK<T>)]]
        v_k[BUF][i.value] = load<T::VEC_KV>(s_k, k_offsets[i.value] + kv_sub_tile_off<T, ns> + k_step_off<T, ek>);
    });
}

template<class T, int SLOT, class UKS>
__device__ inline auto load_k_scale(char* smem, const UKS& u_rk_scale) {
    using namespace opus;
    auto s_scale = make_smem(reinterpret_cast<typename T::D_ATTN*>(smem + scale_slot_base<T, SLOT>));
    auto v = load<T::VEC_SCALE>(s_scale, u_rk_scale);
    return __builtin_bit_cast(vector_t<int, T::GEMM0_E_N>, v);
}

template<class T, int IMM, class SM>
__device__ inline auto tr_load_v(SM& s_v, int lane_off) {
    using D_PV = typename T::D_PV;
    static_assert(IMM >= 0 && IMM < (1 << 16));
    opus::vector_t<opus::i32_t, 2> raw;
    const opus::u32_t addr = static_cast<opus::u32_t>(
        reinterpret_cast<__UINTPTR_TYPE__>(s_v.ptr + lane_off * static_cast<int>(sizeof(D_PV))));
    asm volatile("ds_read_b64_tr_b16 %0, %1 offset:%2\n" : "=a"(raw) : "v"(addr), "i"(IMM) : "memory");
    return __builtin_bit_cast(opus::vector_t<D_PV, T::VEC_TR_V>, raw);
}

template<class T, int STEP, int BUF, class UV, class VV>
__device__ inline void load_v_step(char* smem, const UV&, int v_lane_off, VV& v_v) {
    using namespace opus;
    using D_PV = typename T::D_PV;
    constexpr int ks = STEP / DV_STEPS<T>;
    constexpr int dv = STEP % DV_STEPS<T>;
    constexpr auto imm = layout_imm_offsets_v<remove_cvref_t<UV>, T::VEC_TR_V>;

    auto s_v = make_smem(reinterpret_cast<D_PV*>(smem + v_base<T>));
    static_for<V_N_TILES<T>>([&](auto en) {
        static_for<V_TR_PER_TILE<T>>([&](auto h) {
            constexpr int i = en.value * V_TR_PER_TILE<T> + h.value;
            auto chunk = tr_load_v<T, (v_step_off<T, ks, dv> + imm[i]) * (int)sizeof(D_PV)>(s_v, v_lane_off);
            set_slice(v_v[BUF][en.value], chunk, number<h.value * T::VEC_TR_V>{}, number<(h.value + 1) * T::VEC_TR_V>{});
        });
    });
}

// ----------------------------------------------------------------------------------- dequant

template<class T, int SLOT, int I, class UD, class UDS, class DS, class DE>
__device__ inline void dequant_read(char* smem, const UD& u_rv_dequant, const UDS& u_rs_dequant, DS& deq_src, DE& deq_e8m0) {
    using namespace opus;
    using D_ATTN = typename T::D_ATTN;
    constexpr int r = I % DEQ_RING<T>;
    auto s_k     = make_smem(reinterpret_cast<D_ATTN*>(smem + k_slot_base<T, SLOT>));
    auto s_scale = make_smem(reinterpret_cast<D_ATTN*>(smem + scale_slot_base<T, SLOT>));
    deq_src[r]  = load<T::VEC_DEQ>(s_k, layout_to_offsets<T::VEC_DEQ>(u_rv_dequant)[I]);
    deq_e8m0[r] = load<1>(s_scale, layout_to_offsets<1>(u_rs_dequant)[I]);
}

template<class T, int I, class USD, class DS, class DE>
__device__ inline void dequant_write(char* smem, const USD& u_sv_dequant, const DS& deq_src, const DE& deq_e8m0,
                                     int valid_rows, int warp_id, int lane_id) {
    using namespace opus;
    using D_PV = typename T::D_PV;
    constexpr int r = I % DEQ_RING<T>;
    constexpr int threads_d = T::D_128B_SIZE / T::VEC_DEQ;
    constexpr int ig = I / (T::smem_d_rpt * T::smem_n_per_wave / T::deq_rows_per_inst);
    constexpr int rh = I % (T::smem_n_per_wave / T::deq_rows_per_inst);

    const u32_t e8m0 = __builtin_bit_cast(unsigned char, deq_e8m0[r]);
    const float scale = std::bit_cast<float>(e8m0 << 23);
    const auto src_w = __builtin_bit_cast(vector_t<u32_t, T::VEC_DEQ / 4>, deq_src[r]);

    vector_t<D_PV, T::VEC_DEQ> v_dst;
    auto* dst_pk = reinterpret_cast<vector_t<D_PV, 2>*>(&v_dst);
    static_for<T::VEC_DEQ / 4>([&](auto d) {
        dst_pk[d.value * 2 + 0] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(src_w[d.value], scale, false);
        dst_pk[d.value * 2 + 1] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(src_w[d.value], scale, true);
    });
    if (valid_rows < T::KV_TILE_SIZE) {
        const int pos = ig * T::smem_n_sub_tile + (rh * T::deq_rows_per_inst + lane_id / threads_d) * T::NUM_WARPS + warp_id;
        if (pos >= valid_rows) v_dst = vector_t<D_PV, T::VEC_DEQ>{0};
    }

    auto s_v = make_smem(reinterpret_cast<D_PV*>(smem + v_base<T>));
    store<T::VEC_DEQ>(s_v, v_dst, layout_to_offsets<T::VEC_DEQ>(u_sv_dequant)[I]);
}

// -------------------------------------------------------------------------------------- gemm

template<class T, int SLOT, class UK, class VQ, class SK, class VS, class VK, class DR, class DW, class EM>
__device__ inline void compute_qk(char* smem, const UK& u_rk, const VQ& v_q, int scale_q, const SK& scale_k,
                                  VS& v_s, VK& v_k, const DR& deq_read, const DW& deq_write, const EM& issue_dma) {
    using namespace opus;
    using D_ATTN = typename T::D_ATTN;

    auto mfma_qk = make_mfma<D_ATTN, D_ATTN, typename T::D_ACC>(
        number<T::W_M>{}, number<T::W_N>{}, number<T::W_K_QK>{});
    using op_t     = typename decltype(mfma_qk)::vtype_a;
    using s_tile_t = typename decltype(mfma_qk)::vtype_c;

    const auto* q_ops = reinterpret_cast<const op_t*>(&v_q);
    auto* s_tiles = reinterpret_cast<s_tile_t*>(&v_s);

    static_for<DEQ_PER_STEP<T>>([&](auto j) { deq_read(j); });

    static_for<QK_STEPS<T>>([&](auto st) {
        constexpr int step = st.value;
        constexpr int b    = step % K_DEPTH;
        constexpr int ns   = step / T::GEMM0_E_K;
        constexpr int ek   = step % T::GEMM0_E_K;

        if constexpr (step + K_DEPTH - 1 < QK_STEPS<T>) {
            load_k_step<T, SLOT, step + K_DEPTH - 1, (step + K_DEPTH - 1) % K_DEPTH>(smem, u_rk, v_k);
        }
        issue_dma(number<step>{});
        if constexpr (step + 1 < QK_STEPS<T>) {
            static_for<DEQ_PER_STEP<T>>([&](auto j) { deq_read(number<(step + 1) * DEQ_PER_STEP<T> + j.value>{}); });
        }

        const auto* k_ops = reinterpret_cast<const op_t*>(&v_k[b]);
        static_for<N_TILES_SUB<T>>([&](auto e2) {
            constexpr int e = ns * N_TILES_SUB<T> + e2.value;
            [[clang::amdgpu_pin_vgpr(S_VGPR_BASE + e * ACC_PER_TILE<T>)]]
            s_tiles[e] = mfma_qk(k_ops[e2.value], q_ops[ek], s_tiles[e], scale_k[e], scale_q, number<ek>{}, number<ek>{});
        });

        static_for<DEQ_PER_STEP<T>>([&](auto j) { deq_write(number<step * DEQ_PER_STEP<T> + j.value>{}); });
    });
}

template<class T, int NXT, class UK, class UKS, class UV, class VP, class VO, class VK, class VV, class SK>
__device__ inline void compute_pv(char* smem, const UK& u_rk, const UKS& u_rk_scale, const UV& u_rv, int v_lane_off,
                                  const VP& v_p, VO& v_o, VK& v_k, VV& v_v, SK& scale_k) {
    using namespace opus;
    using D_PV = typename T::D_PV;

    auto mfma_pv = make_mfma<D_PV, D_PV, typename T::D_ACC>(
        number<T::W_M>{}, number<T::W_N>{}, number<T::W_K_PV>{}, mfma_adaptor_swap_ab{});
    using o_tile_t  = typename decltype(mfma_pv)::vtype_c;
    using p_chunk_t = typename decltype(mfma_pv)::vtype_a;

    const auto* p_chunks = reinterpret_cast<const p_chunk_t*>(&v_p);
    auto* o_tiles = reinterpret_cast<o_tile_t*>(&v_o);

    constexpr int K_PRIME_STEPS  = K_DEPTH - 1;
    constexpr int SLOT_SWAP_STEP = PV_STEPS<T> - 1 - K_PRIME_STEPS;

    wait_lgkmcnt<(V_DEPTH - 1) * V_TR_PER_STEP<T>>();
    __builtin_amdgcn_sched_barrier(0);

    static_for<PV_STEPS<T>>([&](auto st) {
        constexpr int step = st.value;
        constexpr int b    = step % V_DEPTH;
        constexpr int ks   = step / DV_STEPS<T>;
        constexpr int dv   = step % DV_STEPS<T>;

        if constexpr (step > SLOT_SWAP_STEP) {
            constexpr int kb = step - SLOT_SWAP_STEP - 1;
            load_k_step<T, NXT, kb, kb>(smem, u_rk, v_k);
        }

        static_for<V_N_TILES<T>>([&](auto en) {
            constexpr int ot = dv * V_N_TILES<T> + en.value;
            [[clang::amdgpu_pin_vgpr(O_VGPR_BASE + ot * ACC_PER_TILE<T>)]]
            o_tiles[ot] = mfma_pv(p_chunks[ks], v_v[b][en.value], o_tiles[ot]);
        });

        if constexpr (step < SLOT_SWAP_STEP) {
            constexpr int v_last = step + V_DEPTH < PV_STEPS<T> ? step + V_DEPTH : PV_STEPS<T> - 1;
            if constexpr (step + V_DEPTH < PV_STEPS<T>) {
                load_v_step<T, step + V_DEPTH, b>(smem, u_rv, v_lane_off, v_v);
            }
            wait_lgkmcnt<(v_last - step - 1) * V_TR_PER_STEP<T>>();
        } else if constexpr (step == SLOT_SWAP_STEP) {
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_sched_barrier(0);
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            scale_k = load_k_scale<T, NXT>(smem, u_rk_scale);
        }
        __builtin_amdgcn_sched_barrier(0);
    });
}

// ----------------------------------------------------------------------------------- softmax

template<int THR_X, int THR_Y>
__device__ inline void attn_mask_vec2_imm(opus::u32_t rel, opus::u32_t neg_inf_v,
                                          opus::u32_t& x_ref, opus::u32_t& y_ref) {
    opus::u64_t x_mask, y_mask;
    asm volatile(
        "v_cmp_lt_i32_e64 %0, %6, %7\n\t"
        "v_cmp_lt_i32_e64 %1, %6, %9\n\t"
        "v_cndmask_b32_e64 %2, %4, %8, %0\n\t"
        "v_cndmask_b32_e64 %3, %5, %8, %1\n\t"
        : "=s"(x_mask), "=s"(y_mask), "=v"(x_ref), "=v"(y_ref)
        : "v"(x_ref), "v"(y_ref), "v"(rel),
          "n"(THR_X), "v"(neg_inf_v), "n"(THR_Y)
        : "vcc"
    );
}

template<class T, class V>
__device__ inline void attn_mask_oob_kv_tile(V& v_s, int valid_kv_len, int kv_tile_idx, opus::u32_t neg_inf_v) {
    using D_ACC = typename T::D_ACC;
    using D_ACC_X2 = opus::vector_t<D_ACC, 2>;
    using U32_X2 = opus::vector_t<opus::u32_t, 2>;

    constexpr int c_pack = T::W_M * T::W_N / T::WARP_SIZE;

    const int last_valid_kv_pos = valid_kv_len - 1;
    const int k_start_pos = kv_tile_idx * T::KV_TILE_SIZE;
    int lane_id = opus::thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int lane_group = lane_id / T::W_N;

    opus::static_for<T::GEMM0_E_N>([&](auto i_n) {
        const int k_pos = k_start_pos + i_n.value * T::W_N + lane_group * c_pack;
        const opus::u32_t rel = static_cast<opus::u32_t>(last_valid_kv_pos - k_pos);

        opus::static_for<c_pack / 2>([&](auto i_pair) {
            constexpr int idx = i_n.value * c_pack + i_pair.value * 2;
            auto pair_bits = __builtin_bit_cast(U32_X2, opus::slice(v_s, opus::number<idx>{}, opus::number<idx + 2>{}));
            opus::u32_t x_ref = pair_bits[0];
            opus::u32_t y_ref = pair_bits[1];
            attn_mask_vec2_imm<i_pair.value * 2, i_pair.value * 2 + 1>(rel, neg_inf_v, x_ref, y_ref);
            pair_bits[0] = x_ref;
            pair_bits[1] = y_ref;
            opus::set_slice(v_s, __builtin_bit_cast(D_ACC_X2, pair_bits), opus::number<idx>{}, opus::number<idx + 2>{});
        });
    });
}

template<class T, class V>
__device__ inline typename T::D_ACC attn_row_max(const V& v_s) {
    using D_ACC = typename T::D_ACC;
    D_ACC row_max = opus::numeric_limits<D_ACC>::lowest();
    opus::static_for<S_ELEMS<T>>([&](auto i) { row_max = max(row_max, v_s[i.value]); });

    opus::vector_t<opus::u32_t, 2> res32 = __builtin_amdgcn_permlane32_swap(std::bit_cast<opus::u32_t>(row_max), std::bit_cast<opus::u32_t>(row_max), false, true);
    row_max = max(std::bit_cast<D_ACC>(res32.x), std::bit_cast<D_ACC>(res32.y));
    opus::vector_t<opus::u32_t, 2> res16 = __builtin_amdgcn_permlane16_swap(std::bit_cast<opus::u32_t>(row_max), std::bit_cast<opus::u32_t>(row_max), false, true);
    return max(std::bit_cast<D_ACC>(res16.x), std::bit_cast<D_ACC>(res16.y));
}

template<class T, class V>
__device__ inline typename T::D_ACC attn_row_sum(const V& v_s) {
    using D_ACC = typename T::D_ACC;
    D_ACC row_sum = D_ACC(0.0f);
    opus::static_for<S_ELEMS<T>>([&](auto i) { row_sum += v_s[i.value]; });

    opus::vector_t<opus::u32_t, 2> res32 = __builtin_amdgcn_permlane32_swap(std::bit_cast<opus::u32_t>(row_sum), std::bit_cast<opus::u32_t>(row_sum), false, true);
    row_sum = std::bit_cast<D_ACC>(res32.x) + std::bit_cast<D_ACC>(res32.y);
    opus::vector_t<opus::u32_t, 2> res16 = __builtin_amdgcn_permlane16_swap(std::bit_cast<opus::u32_t>(row_sum), std::bit_cast<opus::u32_t>(row_sum), false, true);
    return std::bit_cast<D_ACC>(res16.x) + std::bit_cast<D_ACC>(res16.y);
}

template<class T, class VS, class VP>
__device__ inline void attn_cast_p(const VS& v_s, VP& v_p) {
    using pv2_t = opus::vector_t<typename T::D_PV, 2>;
    auto p_pairs = reinterpret_cast<pv2_t*>(&v_p);
    opus::static_for<S_ELEMS<T> / 2>([&](auto i) {
        auto pair = opus::slice(v_s, opus::number<i.value * 2>{}, opus::number<i.value * 2 + 2>{});
        [[clang::amdgpu_pin_vgpr(P_VGPR_BASE + i.value)]]
        p_pairs[i.value] = opus::cast<typename T::D_PV>(pair);
    });
}

template<class T, class VO>
__device__ inline void scale_o_tile(VO& v_o, typename T::D_ACC scale) {
    using namespace opus;
    using acc_tile_t = vector_t<typename T::D_ACC, ACC_PER_TILE<T>>;
    auto o_tiles = reinterpret_cast<acc_tile_t*>(&v_o);
    static_for<O_ELEMS<T> / ACC_PER_TILE<T>>([&](auto i) {
        [[clang::amdgpu_pin_vgpr(O_VGPR_BASE + i.value * ACC_PER_TILE<T>)]]
        o_tiles[i.value] = o_tiles[i.value] * scale;
    });
}

template<class T, class VS, class VP, class VO>
__device__ inline void softmax_tile(VS& v_s, VP& v_p, VO& v_o,
                                    typename T::D_ACC& m_row, typename T::D_ACC& l_row,
                                    typename T::D_ACC temperature_scale) {
    using D_ACC = typename T::D_ACC;
    constexpr D_ACC RESCALE_THRESHOLD = D_ACC(8.0f);

    D_ACC row_max = attn_row_max<T>(v_s) * temperature_scale;
    const bool all_below = __builtin_amdgcn_ballot_w64((row_max - m_row) <= RESCALE_THRESHOLD)
                        == __builtin_amdgcn_read_exec();
    row_max = all_below ? m_row : max(m_row, row_max);

    opus::static_for<S_ELEMS<T>>([&](auto i) {
        v_s[i.value] = __builtin_amdgcn_exp2f(__builtin_fmaf(v_s[i.value], temperature_scale, -row_max));
    });
    const D_ACC row_sum = attn_row_sum<T>(v_s);
    attn_cast_p<T>(v_s, v_p);
    asm volatile("" : "+v"(v_p));

    if (!all_below) {
        const D_ACC rescale = __builtin_amdgcn_exp2f(m_row - row_max);
        m_row = row_max;
        l_row *= rescale;
        scale_o_tile<T>(v_o, rescale);
    }
    l_row += row_sum;
}

// --------------------------------------------------------------------------------- epilogue

template<class T, class G, class VO, class UO>
__device__ inline void store_o_packed(G& g_o, const VO& v_o, const UO& u_o) {
    using namespace opus;
    using D_OUT = typename T::D_OUT;
    constexpr int VEC = VEC_O2<T>;
    constexpr int NC  = O_ELEMS<T> / VEC;

    auto o_offsets = layout_to_offsets<VEC>(u_o);
    static_for<NC>([&](auto c) {
        auto buf = cast<D_OUT>(slice(v_o, number<c.value * VEC>{}, number<(c.value + 1) * VEC>{}));
        auto* p = reinterpret_cast<u32_t*>(&buf);
        auto r0 = __builtin_amdgcn_permlane16_swap(p[0], p[2], false, true);
        auto r1 = __builtin_amdgcn_permlane16_swap(p[1], p[3], false, true);
        p[0] = r0[0]; p[2] = r0[1];
        p[1] = r1[0]; p[3] = r1[1];
        store<VEC>(g_o, buf, o_offsets[c.value]);
    });
}

// ------------------------------------------------------------------------------------ driver

template<class T, class QI, class QR>
__device__ void attention_segment(const opus_mla_v41_prefill_kargs& kargs,
                                  const void* kv_ptr, const void* kv_scale_ptr,
                                  const int* kv_indptr, const int* kv_indices, int q_token_idx, char* smem,
                                  const QI& issue_q, const QR& read_q, const v_q_t<T>& v_q, int scale_q, v_o_t<T>& v_o,
                                  typename T::D_ACC& m_row, typename T::D_ACC& l_row,
                                  typename T::D_ACC temperature_scale) {
    using namespace opus;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    const int page_idx_begin = kv_indptr[q_token_idx];
    const int valid_kv_len   = kv_indptr[q_token_idx + 1] - page_idx_begin;
    const int num_kv_tiles   = ceil_div(valid_kv_len, T::KV_TILE_SIZE);
    if (num_kv_tiles <= 0) return;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    const auto* p_kv    = reinterpret_cast<const D_ATTN*>(kv_ptr);
    const auto* p_scale = reinterpret_cast<const D_ATTN*>(kv_scale_ptr);
    auto g_kv_indices = make_gmem(kv_indices + page_idx_begin, valid_kv_len * sizeof(int));

    auto u_kv_indices = make_layout_kv_indices<T>(warp_id, lane_id);
    auto u_gkv        = make_layout_gkv<T>(lane_id);
    auto u_skv        = make_layout_skv<T>(warp_id);
    auto u_gk_scale   = make_layout_gk_scale<T>(lane_id);
    auto u_sk_scale   = make_layout_sk_scale<T>(warp_id);
    auto u_rk         = make_layout_rk<T>(lane_id);
    auto u_rk_scale   = make_layout_rk_scale<T>(lane_id);
    auto u_rv         = make_layout_rv<T>(lane_id);
    auto u_rv_dequant = make_layout_rv_dequant<T>(warp_id, lane_id);
    auto u_rs_dequant = make_layout_rs_dequant<T>(warp_id, lane_id);
    auto u_sv_dequant = make_layout_sv_dequant<T>(warp_id, lane_id);
    const int v_lane_off = layout_to_offsets<T::VEC_TR_V>(u_rv)[0];

    const u32_t neg_inf_v = std::bit_cast<u32_t>(-numeric_limits<D_ACC>::infinity());

    v_s_t<T> v_s;
    auto& v_p = reinterpret_cast<v_p_t<T>&>(v_s);
    v_k_t<T> v_k[K_DEPTH][K_CHUNKS<T>];
    v_v_t<T> v_v[V_DEPTH][V_N_TILES<T>];
    vector_t<int, T::GEMM0_E_N> scale_k;
    vector_t<D_ATTN, T::VEC_DEQ> deq_src[DEQ_RING<T>];
    vector_t<D_ATTN, 1> deq_e8m0[DEQ_RING<T>];

    auto load_pages = [&](int tile_idx) { return load<1>(g_kv_indices, u_kv_indices, tile_idx * T::KV_TILE_SIZE); };
    decltype(load_pages(0)) kv_pages;

    auto run_tile = [&](auto cur, int tile_idx) {
        constexpr int CUR = decltype(cur)::value;
        constexpr int NXT = 1 - CUR;

        auto kv_lds   = kv_lds_bases<T, NXT>(smem, u_skv);
        auto kv_gmem  = kv_gmem_bases<T>(p_kv, u_gkv, kv_pages, kargs.stride_kv_page);
        auto scl_lds  = scale_lds_bases<T, NXT>(smem, u_sk_scale);
        auto scl_gmem = scale_gmem_bases<T>(p_scale, u_gk_scale, kv_pages, kargs.stride_kv_scale_page);
        auto issue_dma = [&](auto idx) {
            constexpr int i = decltype(idx)::value;
            if constexpr (i < T::kv_async_load_insts) async_load_kv_inst<T, i>(kv_gmem, kv_lds, u_skv);
            if constexpr (i < T::scale_async_load_insts) global_load_lds<T::VEC_SCALE, 0, 0>(scl_gmem[i], scl_lds[i]);
        };

        const int valid_rows = valid_kv_len - tile_idx * T::KV_TILE_SIZE;
        auto deq_read = [&](auto i) {
            dequant_read<T, CUR, decltype(i)::value>(smem, u_rv_dequant, u_rs_dequant, deq_src, deq_e8m0);
        };
        auto deq_write = [&](auto i) {
            dequant_write<T, decltype(i)::value>(smem, u_sv_dequant, deq_src, deq_e8m0, valid_rows, warp_id, lane_id);
        };

        clear(v_s);
        compute_qk<T, CUR>(smem, u_rk, v_q, scale_q, scale_k, v_s, v_k, deq_read, deq_write, issue_dma);
        kv_pages = load_pages(tile_idx + 2);

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        static_for<V_DEPTH>([&](auto i) { load_v_step<T, i.value, i.value>(smem, u_rv, v_lane_off, v_v); });

        if (valid_rows < T::KV_TILE_SIZE) {
            attn_mask_oob_kv_tile<T>(v_s, valid_kv_len, tile_idx, neg_inf_v);
        }
        softmax_tile<T>(v_s, v_p, v_o, m_row, l_row, temperature_scale);
        compute_pv<T, NXT>(smem, u_rk, u_rk_scale, u_rv, v_lane_off, v_p, v_o, v_k, v_v, scale_k);
    };

    kv_pages = load_pages(0);
    __builtin_amdgcn_sched_barrier(0);
    issue_q();
    __builtin_amdgcn_sched_barrier(0);
    async_load_kv_tile<T>(kv_gmem_bases<T>(p_kv, u_gkv, kv_pages, kargs.stride_kv_page),
                          kv_lds_bases<T, 0>(smem, u_skv), u_skv);
    async_load_scale_tile<T>(scale_gmem_bases<T>(p_scale, u_gk_scale, kv_pages, kargs.stride_kv_scale_page),
                             scale_lds_bases<T, 0>(smem, u_sk_scale));
    kv_pages = load_pages(1);
    s_waitcnt_vmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    scale_k = load_k_scale<T, 0>(smem, u_rk_scale);
    read_q();
    static_for<K_DEPTH - 1>([&](auto i) { load_k_step<T, 0, i.value, i.value>(smem, u_rk, v_k); });

    int tile_idx = 0;
    #pragma clang loop unroll(disable)
    for (; tile_idx + 1 < num_kv_tiles; tile_idx += 2) {
        run_tile(0_I, tile_idx);
        run_tile(1_I, tile_idx + 1);
    }
    if (tile_idx < num_kv_tiles) {
        run_tile(0_I, tile_idx);
    }
}

} // namespace opus_mla_v41_prefill_a8w8_16mx4_64nx1

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1)
void opus_mla_v41_prefill_a8w8_16mx4_64nx1_kernel(opus_mla_v41_prefill_kargs kargs) {
    using namespace opus;
    using namespace opus_mla_v41_prefill_a8w8_16mx4_64nx1;
    using T = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;
    using D_OUT  = typename T::D_OUT;

    __shared__ char smem[T::smem_bytes()];

    const int q_token_idx   = block_id_x();
    const int h_block_start = block_id_y() * T::T_M * T::Q_TILE_SIZE;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    const int64_t q_gmem_offset = static_cast<int64_t>(q_token_idx) * kargs.stride_q_n
                                + static_cast<int64_t>(h_block_start) * kargs.stride_q_h;
    const auto* p_q = reinterpret_cast<const D_ATTN*>(kargs.q_ptr) + q_gmem_offset;
    const int num_heads = min(kargs.H - h_block_start, T::T_M * T::Q_TILE_SIZE);

    const int64_t q_scale_gmem_offset = static_cast<int64_t>(q_token_idx) * kargs.stride_q_scale_n
                                      + static_cast<int64_t>(h_block_start) * kargs.stride_q_scale_h;
    auto g_q_scale = make_gmem(reinterpret_cast<const D_ATTN*>(kargs.q_scale_ptr) + q_scale_gmem_offset,
                               (kargs.H - h_block_start) * kargs.stride_q_scale_h * sizeof(D_ATTN));
    auto v_q_scale = load<T::VEC_SCALE>(g_q_scale, make_layout_q_scale<T>(warp_id, lane_id, kargs.stride_q_scale_h));
    const int scale_q = __builtin_bit_cast(int, v_q_scale);

    // Q is loaded by the first non-empty segment, behind its first page-index load: vmcnt
    // retires in order, so an earlier Q would stall that wait.
    v_q_t<T> v_q;
    bool q_pending = true;
    auto issue_q = [&] {
        if (!q_pending) return;
        async_load_q_tile<T>(smem, p_q, kargs.stride_q_h, num_heads, make_layout_kv_indices<T>(warp_id, lane_id),
                             make_layout_gkv<T>(lane_id), make_layout_skv<T>(warp_id));
    };
    // Q must leave its slot before the first QK DMA refills it.
    auto read_q = [&] {
        if (!q_pending) return;
        v_q = load_q<T>(smem, make_layout_rk<T>(lane_id), warp_id);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        q_pending = false;
    };

    constexpr D_ACC LOG2_E = 1.44269504089f;
    const D_ACC temperature_scale = kargs.softmax_scale * LOG2_E;

    v_o_t<T> v_o;
    clear(v_o);
    D_ACC m_row = numeric_limits<D_ACC>::lowest();
    D_ACC l_row = D_ACC(0.0f);

    attention_segment<T>(kargs, kargs.unified_kv_ptr, kargs.unified_kv_scale_ptr,
                         kargs.kv_indptr_prefix, kargs.kv_indices_prefix, q_token_idx, smem,
                         issue_q, read_q, v_q, scale_q, v_o, m_row, l_row, temperature_scale);
    __builtin_amdgcn_s_barrier();
    attention_segment<T>(kargs, kargs.kv_ptr, kargs.kv_scale_ptr,
                         kargs.kv_indptr_extend, kargs.kv_indices_extend, q_token_idx, smem,
                         issue_q, read_q, v_q, scale_q, v_o, m_row, l_row, temperature_scale);

    const int sink_head_idx = h_block_start + warp_id * T::Q_TILE_SIZE + lane_id % T::W_M;
    auto g_attn_sink = make_gmem(reinterpret_cast<const D_ACC*>(kargs.attn_sink_ptr), kargs.H * sizeof(D_ACC));
    const D_ACC sink_log2 = load(g_attn_sink, sink_head_idx)[0] * LOG2_E;
    const D_ACC m_final = max(m_row, sink_log2);
    const D_ACC alpha   = __builtin_amdgcn_exp2f(m_row - m_final);
    const D_ACC l_final = l_row * alpha + __builtin_amdgcn_exp2f(sink_log2 - m_final);
    scale_o_tile<T>(v_o, (l_final > D_ACC(0.0f)) ? (alpha / l_final) : D_ACC(0.0f));

    const int64_t o_gmem_offset = static_cast<int64_t>(q_token_idx) * kargs.stride_o_n
                                + static_cast<int64_t>(h_block_start) * kargs.stride_o_h;
    auto g_o = make_gmem(reinterpret_cast<D_OUT*>(kargs.out_ptr) + o_gmem_offset,
                         (kargs.H - h_block_start) * kargs.stride_o_h * sizeof(D_OUT));
    store_o_packed<T>(g_o, v_o, make_layout_o_packed<T>(warp_id, lane_id, kargs.stride_o_h));
}
