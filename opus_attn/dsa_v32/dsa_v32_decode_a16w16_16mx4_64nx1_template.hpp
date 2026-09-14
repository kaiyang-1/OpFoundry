#pragma once

#include <opus/opus.hpp>
#include "defs.h"
#include <bit>

using opus::operator""_I;

namespace dsa_v32_decode_a16w16_16mx4_64nx1 {
// ----------------------------------------------------------------- register budget and tiles

constexpr int Q_AGPR_BASE = 0;
template<class T> constexpr int Q_ELEMS          = T::Q_TILE_SIZE * T::D_QK_SIZE / T::WARP_SIZE;
template<class T> constexpr int Q_LOAD_INSTS     = Q_ELEMS<T> / T::VEC_Q;
template<class T> constexpr int Q_AGPR_PER_LOAD  = T::VEC_Q * sizeof(typename T::D_ATTN) / 4;

template<class T> constexpr int K_AGPR_BASE      = Q_AGPR_BASE + Q_LOAD_INSTS<T> * Q_AGPR_PER_LOAD<T>;
template<class T> constexpr int K_CHUNKS         = (T::GEMM0_E_N / T::smem_n_sub_tile_rpt) * (T::D_128B_SIZE / T::W_K);
template<class T> constexpr int K_AGPR_PER_CHUNK = T::VEC_KV * sizeof(typename T::D_ATTN) / 4;

template<class T> constexpr int V_N_TILES        = T::D_128B_SIZE / T::W_N;
template<class T> constexpr int V_TR_PER_TILE    = T::W_N * T::W_K / T::WARP_SIZE / T::VEC_TR_V;
template<class T> constexpr int V_TR_GRP_K       = (T::WARP_SIZE / 16) / (T::W_N / (4 * T::VEC_TR_V));
template<class T, int I> constexpr int v_tr_issue_off = (I / V_TR_PER_TILE<T>) * T::W_N + (I % V_TR_PER_TILE<T>) * V_TR_GRP_K<T> * T::D_128B_SIZE;

constexpr int O_VGPR_BASE = 0;
constexpr int S_VGPR_BASE = 128;
template<class T> constexpr int P_VGPR_BASE      = S_VGPR_BASE;
template<class T> constexpr int S_ELEMS          = T::GEMM0_E_N * T::W_M * T::W_N / T::WARP_SIZE;
template<class T> constexpr int S_VGPR_PER_TILE  = T::W_M * T::W_N / T::WARP_SIZE;
template<class T> constexpr int O_ELEMS          = T::Q_TILE_SIZE * T::D_VO_SIZE / T::WARP_SIZE;

template<class T> using v_q_t = opus::array<opus::vector_t<typename T::D_ATTN, T::VEC_Q>, Q_LOAD_INSTS<T>>;
template<class T> using v_s_t = opus::vector_t<typename T::D_ACC, S_ELEMS<T>>;
template<class T> using v_p_t = opus::vector_t<typename T::D_ATTN, S_ELEMS<T>>;
template<class T> using v_o_t = opus::vector_t<typename T::D_ACC, O_ELEMS<T>>;

template<class T, int I> constexpr int kv_sub_tile_off = I * T::NUM_WARPS * T::smem_d_rpt * T::smem_brick;
template<class T, int J> constexpr int kv_d_brick_off  = J * T::NUM_WARPS * T::smem_brick;

// ----------------------------------------------------------------------------------- layouts

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
        opus::number<T::smem_n_sub_tile / T::W_K>{},
        opus::number<lane_hi>{},
        opus::number<T::W_K / (lane_hi * grp_k)>{},
        opus::number<grp_k>{},
        opus::number<grp_n>{},
        opus::number<lane_lo>{},
        opus::number<T::VEC_TR_V>{});

    constexpr auto rv_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int grp_id = lane_id / lane_per_grp;
    const int lane_in_grp = lane_id % lane_per_grp;

    return opus::make_layout(
        rv_shape,
        opus::unfold_x_stride(rv_dim, rv_shape, opus::tuple{opus::number<grp_n * lane_lo * T::VEC_TR_V>{},
                                                            opus::number<T::smem_brick>{},
                                                            opus::number<T::D_128B_SIZE>{},
                                                            1_I}),
        opus::unfold_p_coord(rv_dim, opus::tuple{lane_in_grp / lane_lo, grp_id / grp_n, grp_id % grp_n, lane_in_grp % lane_lo}));
}

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

// ------------------------------------------------------------------------------------ memory

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

template<class T, int IMM, class SM>
__device__ inline auto tr_load_v(SM& s_kv, int off) {
    using D_ATTN = typename T::D_ATTN;
    static_assert(IMM >= 0 && IMM < (1 << 16));
    opus::vector_t<opus::i32_t, 2> raw;
    const opus::u32_t addr = static_cast<opus::u32_t>(
        reinterpret_cast<__UINTPTR_TYPE__>(s_kv.ptr + off * static_cast<int>(sizeof(D_ATTN))));
    asm volatile("ds_read_b64_tr_b16 %0, %1 offset:%2\n" : "=a"(raw) : "v"(addr), "i"(IMM) : "memory");
    return __builtin_bit_cast(opus::vector_t<D_ATTN, T::VEC_TR_V>, raw);
}

// -------------------------------------------------------------------------------------- gemm

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

template<class T, class U, class VP, class VO>
__device__ inline void compute_pv(char* smem_kv, const U& u_rv, const VP& v_p, VO& v_o) {
    using namespace opus;
    using D_ATTN = typename T::D_ATTN;

    auto mfma_pv = make_mfma<D_ATTN, D_ATTN, typename T::D_ACC>(
        number<T::W_M>{}, number<T::W_N>{}, number<T::W_K>{}, mfma_adaptor_swap_ab{});
    using o_tile_t = typename decltype(mfma_pv)::vtype_c;
    using v_tile_t = typename decltype(mfma_pv)::vtype_b;
    using p_tile_t = typename decltype(mfma_pv)::vtype_a;

    auto offsets  = layout_to_offsets<T::VEC_TR_V>(u_rv);
    auto p_chunks = reinterpret_cast<const p_tile_t*>(&v_p);
    auto o_tiles  = reinterpret_cast<o_tile_t*>(&v_o);

    static_for<T::smem_n_sub_tile_rpt>([&](auto ns) {
        auto s_sub = make_smem(reinterpret_cast<D_ATTN*>(smem_kv) + kv_sub_tile_off<T, ns.value>);
        static_for<T::smem_d_rpt_v>([&](auto ds) {
            v_tile_t v[V_N_TILES<T>];
            static_for<V_N_TILES<T>>([&](auto en) {
                static_for<V_TR_PER_TILE<T>>([&](auto h) {
                    constexpr int i = en.value * V_TR_PER_TILE<T> + h.value;
                    vector_t<D_ATTN, T::VEC_TR_V> chunk;
                    chunk = tr_load_v<T, (kv_d_brick_off<T, ds.value> + v_tr_issue_off<T, i>) * (int)sizeof(D_ATTN)>(s_sub, offsets[0]);
                    s_waitcnt_lgkmcnt(0_I);
                    __builtin_amdgcn_sched_barrier(0);
                    set_slice(v[en.value], chunk, number<h.value * T::VEC_TR_V>{},
                                                  number<(h.value + 1) * T::VEC_TR_V>{});
                });
            });
            s_waitcnt_lgkmcnt(0_I);
            static_for<V_N_TILES<T>>([&](auto en) {
                constexpr int ot = ds.value * V_N_TILES<T> + en.value;
                [[clang::amdgpu_pin_vgpr(O_VGPR_BASE + ot * S_VGPR_PER_TILE<T>)]]
                o_tiles[ot] = mfma_pv(p_chunks[ns.value], v[en.value], o_tiles[ot]);
            });
            __builtin_amdgcn_sched_barrier(0);
        });
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
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    D_ACC row_max = opus::numeric_limits<D_ACC>::lowest();
    opus::static_for<s_len>([&](auto i) { row_max = max(row_max, v_s[i.value]); });

    opus::vector_t<opus::u32_t, 2> res32 = __builtin_amdgcn_permlane32_swap(std::bit_cast<opus::u32_t>(row_max), std::bit_cast<opus::u32_t>(row_max), false, true);
    row_max = max(std::bit_cast<float>(res32.x), std::bit_cast<float>(res32.y));
    opus::vector_t<opus::u32_t, 2> res16 = __builtin_amdgcn_permlane16_swap(std::bit_cast<opus::u32_t>(row_max), std::bit_cast<opus::u32_t>(row_max), false, true);
    return max(std::bit_cast<float>(res16.x), std::bit_cast<float>(res16.y));
}

template<class T, class V>
__device__ inline typename T::D_ACC attn_row_sum(const V& v_s) {
    using D_ACC = typename T::D_ACC;
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    D_ACC row_sum = 0.0f;
    opus::static_for<s_len>([&](auto i) { row_sum += v_s[i.value]; });

    opus::vector_t<opus::u32_t, 2> res32 = __builtin_amdgcn_permlane32_swap(std::bit_cast<opus::u32_t>(row_sum), std::bit_cast<opus::u32_t>(row_sum), false, true);
    row_sum = std::bit_cast<float>(res32.x) + std::bit_cast<float>(res32.y);
    opus::vector_t<opus::u32_t, 2> res16 = __builtin_amdgcn_permlane16_swap(std::bit_cast<opus::u32_t>(row_sum), std::bit_cast<opus::u32_t>(row_sum), false, true);
    return std::bit_cast<float>(res16.x) + std::bit_cast<float>(res16.y);
}

template<class T, class VO>
__device__ inline void scale_o_tile(VO& v_o, typename T::D_ACC scale) {
    using namespace opus;
    using acc_tile_t = vector_t<typename T::D_ACC, S_VGPR_PER_TILE<T>>;
    auto o_tiles = reinterpret_cast<acc_tile_t*>(&v_o);
    static_for<O_ELEMS<T> / S_VGPR_PER_TILE<T>>([&](auto i) {
        [[clang::amdgpu_pin_vgpr(O_VGPR_BASE + i.value * S_VGPR_PER_TILE<T>)]]
        o_tiles[i.value] = o_tiles[i.value] * scale;
    });
}

template<class T, class VS, class VP, class VO>
__device__ inline void softmax_tile(VS& v_s, VP& v_p, VO& v_o,
                                    typename T::D_ACC& m_row, typename T::D_ACC& l_row,
                                    typename T::D_ACC temperature_scale) {
    using namespace opus;
    using D_ACC = typename T::D_ACC;
    using attn2_t = vector_t<typename T::D_ATTN, 2>;

    const D_ACC row_max = max(m_row, attn_row_max<T>(v_s) * temperature_scale);
    const D_ACC rescale = __builtin_amdgcn_exp2f(m_row - row_max);

    m_row = row_max;

    static_for<S_ELEMS<T>>([&](auto i) {
        v_s[i.value] = __builtin_amdgcn_exp2f(__builtin_fmaf(v_s[i.value], temperature_scale, -row_max));
    });

    l_row = __builtin_fmaf(l_row, rescale, attn_row_sum<T>(v_s));

    auto p_pairs = reinterpret_cast<attn2_t*>(&v_p);
    static_for<S_ELEMS<T> / 2>([&](auto i) {
        auto pair = slice(v_s, number<i.value * 2>{}, number<i.value * 2 + 2>{});
        [[clang::amdgpu_pin_vgpr(P_VGPR_BASE<T> + i.value)]]
        p_pairs[i.value] = cast<typename T::D_ATTN>(pair);
    });

    scale_o_tile<T>(v_o, rescale);
}

// ------------------------------------------------------------------------------------ driver

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
    using D_ACC  = typename T::D_ACC;

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
    auto u_rv         = make_layout_rv<T>(lane_id);

    const u32_t neg_inf_v = std::bit_cast<u32_t>(-numeric_limits<D_ACC>::infinity());

    for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
        auto kv_pages = load<1>(g_kv_indices, u_kv_indices, tile_idx * T::KV_TILE_SIZE);

        __builtin_amdgcn_s_barrier();
        async_load_kv_tile<T>(g_kv, s_kv, u_gkv, u_skv, kv_pages, kargs.stride_kv_page);
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        v_s_t<T> v_s;
        clear(v_s);
        compute_qk<T>(smem_kv, u_rk, v_q, v_s);

        if ((tile_idx + 1) * T::KV_TILE_SIZE > valid_kv_len) {
            attn_mask_oob_kv_tile<T>(v_s, valid_kv_len, tile_idx, neg_inf_v);
        }

        v_p_t<T> v_p;
        softmax_tile<T>(v_s, v_p, v_o, m_row, l_row, temperature_scale);
        compute_pv<T>(smem_kv, u_rv, v_p, v_o);

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
    scale_o_tile<T>(v_o, o_scale);

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
