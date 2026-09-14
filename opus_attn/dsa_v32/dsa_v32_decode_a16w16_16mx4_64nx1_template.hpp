#pragma once

#include <opus/opus.hpp>
#include "defs.h"

using opus::operator""_I;

namespace dsa_v32_decode_a16w16_16mx4_64nx1 {

constexpr int Q_AGPR_BASE = 0;
constexpr int S_VGPR_BASE = 0;

template<class T> constexpr int Q_ELEMS          = T::Q_TILE_SIZE * T::D_QK_SIZE / T::WARP_SIZE;
template<class T> constexpr int Q_LOAD_INSTS     = Q_ELEMS<T> / T::VEC_Q;
template<class T> constexpr int Q_AGPR_PER_LOAD  = T::VEC_Q * sizeof(typename T::D_ATTN) / 4;

template<class T> constexpr int K_CHUNKS         = (T::GEMM0_E_N / T::smem_n_sub_tile_rpt) * (T::D_128B_SIZE / T::W_K);
template<class T> constexpr int K_AGPR_PER_CHUNK = T::VEC_KV * sizeof(typename T::D_ATTN) / 4;
template<class T> constexpr int K_AGPR_BASE      = Q_AGPR_BASE + Q_LOAD_INSTS<T> * Q_AGPR_PER_LOAD<T>;

template<class T> constexpr int S_ELEMS          = T::GEMM0_E_N * T::W_M * T::W_N / T::WARP_SIZE;
template<class T> constexpr int S_VGPR_PER_TILE  = T::W_M * T::W_N / T::WARP_SIZE;

template<class T> constexpr int O_ELEMS          = T::Q_TILE_SIZE * T::D_VO_SIZE / T::WARP_SIZE;

template<class T> using v_q_t = opus::array<opus::vector_t<typename T::D_ATTN, T::VEC_Q>, Q_LOAD_INSTS<T>>;
template<class T> using v_s_t = opus::vector_t<typename T::D_ACC, S_ELEMS<T>>;
template<class T> using v_o_t = opus::vector_t<typename T::D_ACC, O_ELEMS<T>>;

template<class T, int I> constexpr int kv_sub_tile_off = I * T::NUM_WARPS * T::smem_d_rpt * T::smem_brick;
template<class T, int J> constexpr int kv_d_brick_off  = J * T::NUM_WARPS * T::smem_brick;

template<class T>
__device__ inline auto make_layout_q(int warp_id, int lane_id, int stride_q_h) {
    constexpr auto q_block_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::GEMM0_E_K>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_Q>{});

    constexpr auto q_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        q_block_shape,
        opus::unfold_x_stride(q_block_dim, q_block_shape, opus::tuple{stride_q_h, 1_I}),
        opus::unfold_p_coord(q_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

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
__device__ inline auto make_layout_rk(int lane_id) {
    constexpr auto rk_shape = opus::make_tuple(
        opus::number<T::NUM_WARPS>{},
        opus::number<T::GEMM0_E_N / T::smem_n_sub_tile_rpt>{},
        opus::number<T::W_N / T::NUM_WARPS>{},
        opus::number<T::D_128B_SIZE / T::W_K>{},
        opus::number<T::WARP_SIZE / T::W_N>{},
        opus::number<T::VEC_KV>{});

    constexpr auto rk_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_n = lane_id % T::W_N;

    return opus::make_layout(
        rk_shape,
        opus::unfold_x_stride(rk_dim, rk_shape, opus::tuple{opus::number<T::smem_brick>{}, opus::number<T::D_128B_SIZE>{}, 1_I}),
        opus::unfold_p_coord(rk_dim, opus::tuple{lane_n % T::NUM_WARPS, lane_n / T::NUM_WARPS, lane_id / T::W_N}));
}

template<class Traits> __device__ auto make_layout_rv(int lane_id);

template<class T>
__device__ inline auto make_layout_o(int warp_id, int lane_id, int stride_o_h) {
    constexpr auto o_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::D_VO_SIZE / T::W_N>{},
        opus::number<T::W_M * T::W_N / T::WARP_SIZE / T::VEC_O>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_O>{});

    constexpr auto o_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        o_block_shape,
        opus::unfold_x_stride(o_block_dim, o_block_shape, opus::tuple{stride_o_h, 1_I}),
        opus::unfold_p_coord(o_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

template<class T, class G, class U>
__device__ inline auto load_q(G& g_q, const U& u_q) {
    using namespace opus;
    auto offsets = layout_to_offsets<T::VEC_Q>(u_q);
    v_q_t<T> v_q;
    static_for<Q_LOAD_INSTS<T>>([&](auto i) {
        [[clang::amdgpu_pin_agpr(Q_AGPR_BASE + i.value * Q_AGPR_PER_LOAD<T>)]]
        v_q[i.value] = load<T::VEC_Q>(g_q, offsets[i.value]);
    });
    return v_q;
}

template<class T, class G, class S, class UG, class US, class VP>
__device__ inline void async_load_kv_tile(G& g_kv, S& s_kv, const UG& u_gkv, const US& u_skv,
                                          const VP& kv_pages, int stride_kv_page) {
    opus::static_for<T::smem_n_sub_tile_rpt>([&](auto ig) {
        opus::async_load<T::VEC_KV>(g_kv, s_kv.ptr,
                                    u_gkv + kv_pages[ig.value] * stride_kv_page,
                                    u_skv + opus::number<kv_sub_tile_off<T, ig.value>>{});
    });
}

template<class T, class U, class VQ, class VS>
__device__ inline void compute_qk(char* smem_kv, const U& u_rk, const VQ& v_q, VS& v_s) {
    using namespace opus;
    using D_ATTN = typename T::D_ATTN;
    constexpr int N_TILES = T::GEMM0_E_N / T::smem_n_sub_tile_rpt;
    constexpr int K_STEPS = T::D_128B_SIZE / T::W_K;

    auto mfma_qk = make_mfma<D_ATTN, D_ATTN, typename T::D_ACC>(
        number<T::W_M>{}, number<T::W_N>{}, number<T::W_K>{}, mfma_adaptor_swap_ab{});
    using s_tile_t = typename decltype(mfma_qk)::vtype_c;

    auto offsets = layout_to_offsets<T::VEC_KV>(u_rk);
    auto s_tiles = reinterpret_cast<s_tile_t*>(&v_s);

    static_for<T::smem_n_sub_tile_rpt>([&](auto ns) {
        auto s_sub = make_smem(reinterpret_cast<D_ATTN*>(smem_kv) + kv_sub_tile_off<T, ns.value>);
        static_for<T::smem_d_rpt>([&](auto ds) {
            vector_t<D_ATTN, T::VEC_KV> k[K_CHUNKS<T>];
            static_for<K_CHUNKS<T>>([&](auto i) {
                [[clang::amdgpu_pin_agpr(K_AGPR_BASE<T> + i.value * K_AGPR_PER_CHUNK<T>)]]
                k[i.value] = load<T::VEC_KV>(s_sub, offsets[i.value] + kv_d_brick_off<T, ds.value>);
            });
            s_waitcnt_lgkmcnt(0_I);
            static_for<N_TILES>([&](auto e2) {
                static_for<K_STEPS>([&](auto kk) {
                    constexpr int e  = ns.value * N_TILES + e2.value;
                    constexpr int ik = e2.value * K_STEPS + kk.value;
                    constexpr int ek = ds.value * K_STEPS + kk.value;
                    [[clang::amdgpu_pin_vgpr(S_VGPR_BASE + e * S_VGPR_PER_TILE<T>)]]
                    s_tiles[e] = mfma_qk(v_q[ek], k[ik], s_tiles[e]);
                });
            });
        });
    });
}

template<class Traits, class VS>
__device__ typename Traits::D_ACC attn_row_max(const VS& v_s);
template<class Traits, class VS>
__device__ typename Traits::D_ACC attn_row_sum(const VS& v_s);
template<class Traits, class VS>
__device__ void attn_mask_oob_kv_tile(VS& v_s, int valid_kv_len, int kv_tile_idx, opus::u32_t neg_inf_v);

template<class Traits>
__device__ void attention_tiles(const dsa_v32_a16w16_kargs& kargs,
                                int page_idx_begin, int valid_kv_len,
                                int tile_begin, int tile_end,
                                char* smem_kv, const v_q_t<Traits>& v_q,
                                v_o_t<Traits>& v_o,
                                typename Traits::D_ACC& m_row,
                                typename Traits::D_ACC& l_row,
                                typename Traits::D_ACC temperature_scale) {
    using namespace opus;
    using T = remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    auto g_kv = make_gmem(reinterpret_cast<const D_ATTN*>(kargs.kv_ptr),
                          kargs.total_tokens * kargs.stride_kv_page * sizeof(D_ATTN));
    auto g_kv_indices = make_gmem(kargs.kv_indices + page_idx_begin, valid_kv_len * sizeof(int));
    auto s_kv = make_smem(reinterpret_cast<D_ATTN*>(smem_kv));

    auto u_kv_indices = make_layout_kv_indices<T>(warp_id, lane_id);
    auto u_gkv        = make_layout_gkv<T>(lane_id);
    auto u_skv        = make_layout_skv<T>(warp_id);
    auto u_rk         = make_layout_rk<T>(lane_id);

    for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
        auto kv_pages = load<1>(g_kv_indices, u_kv_indices, tile_idx * T::KV_TILE_SIZE);

        __builtin_amdgcn_s_barrier();
        async_load_kv_tile<T>(g_kv, s_kv, u_gkv, u_skv, kv_pages, kargs.stride_kv_page);
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        v_s_t<T> v_s;
        clear(v_s);
        compute_qk<T>(smem_kv, u_rk, v_q, v_s);

    }
}

template<class Traits>
__device__ void decode_one_req(const dsa_v32_a16w16_kargs& kargs, int batch_idx, int h_block_idx,
                               int page_idx_begin, int valid_kv_len,
                               int tile_begin, int tile_end, int slot,
                               char* smem_kv, typename Traits::D_ACC temperature_scale) {
    using namespace opus;
    using T = remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;
    using D_OUT  = typename T::D_OUT;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    const int h_block_start = h_block_idx * T::T_M * T::Q_TILE_SIZE;
    const int q_gmem_offset = batch_idx * kargs.stride_q_b + h_block_start * kargs.stride_q_h;

    auto g_q = make_gmem(reinterpret_cast<const D_ATTN*>(kargs.q_ptr) + q_gmem_offset,
                         (kargs.H - h_block_start) * kargs.stride_q_h * sizeof(D_ATTN));
    auto u_q = make_layout_q<T>(warp_id, lane_id, kargs.stride_q_h);
    auto v_q = load_q<T>(g_q, u_q);

    v_o_t<T> v_o;
    clear(v_o);
    D_ACC m_row = numeric_limits<D_ACC>::lowest();
    D_ACC l_row = D_ACC(0.0f);

    attention_tiles<T>(kargs, page_idx_begin, valid_kv_len, tile_begin, tile_end,
                       smem_kv, v_q, v_o, m_row, l_row, temperature_scale);

    const D_ACC o_scale = (l_row > D_ACC(0.0f)) ? (D_ACC(1.0f) / l_row) : D_ACC(0.0f);
    static_for<O_ELEMS<T>>([&](auto i) { v_o[i.value] *= o_scale; });

    int lane_id_o = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id_o));
    const int warp_id_o = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    if (slot < 0) {
        const int o_gmem_offset = batch_idx * kargs.stride_o_b + h_block_start * kargs.stride_o_h;
        auto g_o = make_gmem(reinterpret_cast<D_OUT*>(kargs.out_ptr) + o_gmem_offset,
                             (kargs.H - h_block_start) * kargs.stride_o_h * sizeof(D_OUT));
        auto u_o = make_layout_o<T>(warp_id_o, lane_id_o, kargs.stride_o_h);
        auto v_o_out = cast<D_OUT>(v_o);
        store<T::VEC_O>(g_o, v_o_out, u_o);

        if (lane_id_o < T::W_M) {
            const int lse_offset = batch_idx * kargs.stride_lse_b + h_block_start;
            auto g_lse = make_gmem(reinterpret_cast<D_ACC*>(kargs.lse_ptr) + lse_offset,
                                   (kargs.H - h_block_start) * sizeof(D_ACC));
            const D_ACC lse = (l_row > D_ACC(0.0f)) ? (m_row + log2f(l_row)) * D_ACC(DSA_V32_LN_2)
                                                    : numeric_limits<D_ACC>::infinity();
            g_lse.store(lse, warp_id_o * T::Q_TILE_SIZE + lane_id_o);
        }
    } else {
        const int oa_offset = (slot * kargs.H + h_block_start) * T::D_VO_SIZE;
        auto g_oa = make_gmem(reinterpret_cast<D_ACC*>(kargs.o_accum) + oa_offset,
                              (kargs.H - h_block_start) * T::D_VO_SIZE * sizeof(D_ACC));
        auto u_oa = make_layout_o<T>(warp_id_o, lane_id_o, T::D_VO_SIZE);
        store<T::VEC_O>(g_oa, v_o, u_oa);

        if (lane_id_o < T::W_M) {
            const int lse_offset = slot * kargs.H + h_block_start;
            auto g_lse = make_gmem(reinterpret_cast<D_ACC*>(kargs.lse_accum) + lse_offset,
                                   (kargs.H - h_block_start) * sizeof(D_ACC));
            const D_ACC lse = (l_row > D_ACC(0.0f)) ? (m_row + log2f(l_row))
                                                    : numeric_limits<D_ACC>::lowest();
            g_lse.store(lse, warp_id_o * T::Q_TILE_SIZE + lane_id_o);
        }
    }
}

}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1)
void dsa_v32_decode_a16w16_16mx4_64nx1_kernel(dsa_v32_a16w16_kargs kargs) {
    using namespace opus;
    using namespace dsa_v32_decode_a16w16_16mx4_64nx1;
    using T = opus::remove_cvref_t<Traits>;

    const int part = block_id_x();
    const int h_block_idx = block_id_y();
    if (part >= kargs.num_parts) return;

    const DsaSchedMeta meta = kargs.sched_meta[part];
    if (meta.begin_req_idx >= kargs.B) return;

    __shared__ char smem_kv[T::smem_bytes()];

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
                          tile_begin, tile_end, slot, smem_kv, temperature_scale);
    }
}
