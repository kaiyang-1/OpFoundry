#pragma once

#include <opus/opus.hpp>
#include "mla_v4_traits.h"
#include <cstdint>
#include <bit>

using opus::operator""_I;

namespace opus_mla_v4_prefill_a8w8_32mx1_16nx4 {

OPUS_D opus::u32x16_t s_buffer_load_b512(opus::u32x4_t rsrc, int soffset) {
    opus::u32x16_t ids;
    asm volatile("s_buffer_load_b512 %0, %1, %2 offset:0x0 nv"
                 : "=&s"(ids)
                 : "s"(rsrc), "s"(soffset));
    return ids;
}

OPUS_D void s_wait_kmcnt_for(opus::u32x16_t& ids) {
    asm volatile("s_wait_kmcnt 0x0" : "+s"(ids));
}

OPUS_D opus::u32x4_t make_buffer_rsrc_raw(const void* ptr, opus::u32_t num_bytes,
                                          opus::u32_t config = opus::buffer_default_config()) {
    __amdgpu_buffer_rsrc_t rsrc = __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(ptr), /*stride=*/0, num_bytes, config);
    opus::u32x4_t raw;
    __builtin_memcpy(&raw, &rsrc, sizeof(raw));
    opus::static_for<4>([&](auto k) { raw[k.value] = __builtin_amdgcn_readfirstlane(raw[k.value]); });
    return raw;
}

template<class T, int E_LEAD, int E_K, int W_K, int VEC, int ROW_ELEMS>
__device__ inline auto make_layout_operand(int lane_id) {
    constexpr auto shape = opus::make_tuple(
        opus::number<E_LEAD>{},
        opus::number<T::W_M>{},
        opus::number<E_K>{},
        opus::number<T::W_M * W_K / (T::WARP_SIZE * VEC)>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<VEC>{});

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<ROW_ELEMS>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{lane_id % T::W_M, lane_id / T::W_M}));
}

template<class T, int E_LEAD, int ROW_ELEMS>
__device__ inline auto make_layout_mxscl(int lane_id) {
    constexpr auto shape = opus::make_tuple(
        opus::number<E_LEAD>{},
        opus::number<T::W_M>{},
        opus::number<T::W_M * T::D_NOPE_PADDED_SIZE / T::MXSCL_BLOCK_SIZE / (T::WARP_SIZE * T::VEC_MXSCL)>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_MXSCL>{});

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<ROW_ELEMS>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{lane_id % T::W_M, lane_id / T::W_M}));
}

template<class T>
__device__ inline auto make_layout_v_nope(int lane_id) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::W_N>{},
        opus::number<T::V_NOPE_BLOCKS>{},
        opus::number<T::WARP_SIZE / T::W_N>{},
        opus::number<T::VEC_NOPE>{});

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<T::V_ROW_LDS_ELEMS>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{lane_id % T::W_N, lane_id / T::W_N}));
}

template<class T>
__device__ inline auto make_layout_v(int lane_id) {
    constexpr int lane_per_grp = 16;
    constexpr int lane_n = 2;
    constexpr int lane_k = lane_per_grp / lane_n;

    constexpr int dwordx32_rpt = 4 * 32 / sizeof(typename T::D_ROPE) / T::VEC_ROPE;

    constexpr auto v_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_N / dwordx32_rpt>{},
        opus::number<lane_n>{},
        opus::number<dwordx32_rpt>{},
        opus::number<T::GEMM1_E_K * T::W_K_ROPE / (T::WARP_SIZE / lane_per_grp) / T::VEC_ROPE>{},
        opus::number<T::WARP_SIZE / lane_per_grp>{},
        opus::number<lane_k>{},
        opus::number<T::VEC_ROPE>{});

    constexpr auto v_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        v_block_shape,
        opus::unfold_x_stride(v_block_dim, v_block_shape, opus::tuple{opus::number<T::VEC_ROPE>{}, opus::number<T::V_ROW_LDS_ELEMS>{}, 1_I}),
        opus::unfold_p_coord(v_block_dim, opus::tuple{(lane_id % lane_per_grp) / lane_k, lane_id / lane_per_grp, (lane_id % lane_per_grp) % lane_k}));
}

template<class T>
__device__ inline auto make_layout_o(int warp_id, int lane_id) {
    constexpr int dwordx32_rpt = 4 * 32 / sizeof(typename T::D_OUT) / T::VEC_O;

    constexpr auto o_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_M>{},
        opus::number<T::W_M>{},
        opus::number<T::T_N>{},
        opus::number<T::GEMM1_E_N / dwordx32_rpt>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<dwordx32_rpt>{},
        opus::number<T::VEC_O>{});

    constexpr auto o_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}));

    return opus::make_layout(
        o_block_shape,
        opus::unfold_x_stride(o_block_dim, o_block_shape, opus::tuple{opus::number<T::O_ROW_LDS_ELEMS>{}, 1_I}),
        opus::unfold_p_coord(o_block_dim, opus::tuple{lane_id % T::W_M, warp_id, lane_id / T::W_M}));
}

template<class T, int NBLK, class V>
__device__ inline void zero_nope_pad(V& v) {
    constexpr opus::index_t per  = opus::vector_traits<V>::size() / NBLK;
    constexpr opus::index_t vals = per * T::D_NOPE_SIZE / T::D_NOPE_PADDED_SIZE;
    opus::static_for<NBLK>([&](auto b) {
        opus::static_for([&](auto i) { v[b.value * per + i.value] = static_cast<typename T::D_NOPE>(0); },
                         opus::number<vals>{}, opus::number<per>{});
    });
}

template<class T, int NBLK, class V>
__device__ inline void zero_mxscl_pad(V& v, bool upper_half) {
    constexpr opus::index_t dwords_per_blk = opus::vector_traits<V>::size() / 4 / NBLK;
    constexpr opus::index_t pad = (T::D_NOPE_PADDED_SIZE - T::D_NOPE_SIZE) / T::MXSCL_BLOCK_SIZE;
    const opus::u32_t keep = upper_half ? (0xFFFFFFFFu >> (8 * pad)) : 0xFFFFFFFFu;
    auto* dw = reinterpret_cast<opus::u32_t*>(&v);
    opus::static_for<NBLK>([&](auto b) { dw[(b.value + 1) * dwords_per_blk - 1] &= keep; });
}

template<int Lo, int Hi, typename V, typename Op>
__device__ inline auto tree_reduce(const V& v, Op op) {
    if constexpr (Hi - Lo == 1) return v[Lo];
    else {
        constexpr int Mid = (Lo + Hi) / 2;
        return op(tree_reduce<Lo, Mid>(v, op), tree_reduce<Mid, Hi>(v, op));
    }
}

template<typename T, opus::index_t Offset, opus::index_t Count, typename V>
__device__ inline typename T::D_ACC attn_row_max(const V& v_s) {
    using D_ACC = typename T::D_ACC;
    D_ACC row_max = max(opus::numeric_limits<D_ACC>::lowest(),
                        tree_reduce<Offset, Offset + Count>(v_s, [](D_ACC x, D_ACC y) { return max(x, y); }));

    int res16 = __builtin_amdgcn_permlane_xor(std::bit_cast<int>(row_max), 16, 32);
    return max(row_max, std::bit_cast<float>(res16));
}

template<typename T, opus::index_t Offset, opus::index_t Count, typename V>
__device__ inline typename T::D_ACC attn_row_sum(const V& v_s) {
    using D_ACC = typename T::D_ACC;
    D_ACC row_sum = tree_reduce<Offset, Offset + Count>(v_s, [](D_ACC x, D_ACC y) { return x + y; });

    int res16 = __builtin_amdgcn_permlane_xor(std::bit_cast<int>(row_sum), 16, 32);
    return row_sum + std::bit_cast<float>(res16);
}

template<typename T, typename V>
__device__ inline auto attn_row_max_blocks(const V& v_s) {
    constexpr opus::index_t per_m = opus::vector_traits<V>::size() / T::GEMM0_E_M;
    opus::vector_t<typename T::D_ACC, T::GEMM0_E_M> out;
    opus::static_for<T::GEMM0_E_M>([&](auto m) {
        out[m.value] = attn_row_max<T, m.value * per_m, per_m>(v_s);
    });
    return out;
}

template<typename T, opus::index_t Offset, opus::index_t Count, typename V>
__device__ inline void attn_row_scale_sub(V& v_s, typename T::D_ACC scale, typename T::D_ACC row_max) {
    opus::static_for<Count>([&](auto i) {
        v_s[Offset + i.value] = __builtin_fmaf(v_s[Offset + i.value], scale, -row_max);
    });
}

template<typename T, typename V>
__device__ inline void attn_exp2(V& v_s) {
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    opus::static_for<s_len>([&](auto i) { v_s[i.value] = __builtin_amdgcn_exp2f(v_s[i.value]); });
}

template<typename T, opus::index_t Offset, opus::index_t Count, typename V>
__device__ inline void scale_slice(V& v, typename T::D_ACC scale) {
    opus::static_for<Count>([&](auto i) { v[Offset + i.value] *= scale; });
}

template<typename T, typename V>
__device__ inline void attn_mask_oob_score(V& v_s, int valid_kv_len, int kv_tile_idx,
                                           int wave_kv_base, int lane_id) {
    using D_ACC = typename T::D_ACC;

    if ((kv_tile_idx + 1) * T::KV_TILE_SIZE <= valid_kv_len) return;

    constexpr int elems_per_mma   = T::W_M * T::W_N / T::WARP_SIZE;
    constexpr int lane_hi_kv_step = T::W_N / (T::WARP_SIZE / T::W_M);
    static_assert(opus::vector_traits<V>::size() == T::GEMM0_E_M * T::GEMM0_E_N * elems_per_mma);

    const D_ACC neg_inf = -opus::numeric_limits<D_ACC>::infinity();
    const int lane_hi = lane_id / T::W_M;
    const int rel_base = (valid_kv_len - 1) - kv_tile_idx * T::KV_TILE_SIZE - wave_kv_base
                       - lane_hi * lane_hi_kv_step;

    opus::static_for<T::GEMM0_E_M * T::GEMM0_E_N>([&](auto i_blk) {
        const int rel = rel_base - (i_blk.value % T::GEMM0_E_N) * T::W_N;
        opus::static_for<elems_per_mma>([&](auto i_reg) {
            constexpr int idx = i_blk.value * elems_per_mma + i_reg.value;
            v_s[idx] = (i_reg.value > rel) ? neg_inf : v_s[idx];
        });
    });
}

template<typename T, typename S, typename V>
__device__ inline void ml_publish(S& s_ml, const V& parts, int warp_id, int lane_id) {
    typename T::D_ACC mine = parts[0];
    opus::static_for<T::GEMM0_E_M - 1>([&](auto i) {
        mine = (lane_id / T::W_M == i.value + 1) ? parts[i.value + 1] : mine;
    });
    opus::store(s_ml, mine,
                (lane_id / T::W_M) * (T::W_M * T::T_N) + (lane_id % T::W_M) * T::T_N + warp_id);
}

template<typename T, typename S, typename Op>
__device__ inline auto ml_reduce(S& s_ml, int lane_id, Op op) {
    opus::vector_t<typename T::D_ACC, T::GEMM0_E_M> out;
    opus::static_for<T::GEMM0_E_M>([&](auto i) {
        auto parts = opus::load<T::T_N>(s_ml, i.value * (T::W_M * T::T_N) + (lane_id % T::W_M) * T::T_N);
        out[i.value] = tree_reduce<0, T::T_N>(parts, op);
    });
    return out;
}

template<typename T, typename S, typename V>
__device__ inline void store_p(S& s_p, const V& v_s, int warp_id, int lane_id) {
    using namespace opus;
    static_for<T::GEMM0_E_M * T::P_CHUNKS>([&](auto i) {
        constexpr int m = i.value / T::P_CHUNKS;
        constexpr int c = i.value % T::P_CHUNKS;
        vector_t<typename T::D_ACC, T::VEC_P> f;
        static_for<T::VEC_P>([&](auto j) {
            f[j.value] = v_s[(m * T::GEMM0_E_N + c) * T::VEC_P + j.value];
        });
        const int blk = (m * T::T_N + warp_id) * T::P_CHUNKS + c;
        store<T::VEC_P>(s_p, cast<typename T::D_ROPE>(f), blk * T::P_BLOCK_ELEMS + lane_id * T::VEC_P);
    });
}

template<typename T, typename S, typename V>
__device__ inline void gather_p(S& s_p, V& v_p, int lane_id) {
    opus::static_for<T::P_NUM_BLOCKS>([&](auto b) {
        auto chunk = opus::load<T::VEC_P>(s_p, b.value * T::P_BLOCK_ELEMS + lane_id * T::VEC_P);
        opus::static_for<T::VEC_P>([&](auto j) {
            v_p[b.value * T::VEC_P + j.value] = chunk[j.value];
        });
    });
}

template<class Traits, class VQN, class VQR, class VQS, class VO>
__device__ __attribute__((always_inline)) void mla_v4_prefill_accum_pipelined(
        opus_mla_v4_prefill_fp8_kargs kargs,
        const void* kv_nope_ptr, const void* kv_rope_ptr, int kv_rows,
        const int* kv_indices, int page_idx_begin, int valid_kv_len, int num_kv_tiles,
        char* smem_buf,
        VQN& v_q_nope, VQR& v_q_rope, VQS& v_q_mxscl, VO& v_o,
        opus::vector_t<typename Traits::D_ACC, Traits::GEMM0_E_M>& m_row,
        opus::vector_t<typename Traits::D_ACC, Traits::GEMM0_E_M>& l_part,
        typename Traits::D_ACC temperature_scale) {
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;
    using D_NOPE = typename T::D_NOPE;
    using D_ROPE = typename T::D_ROPE;
    using D_ACC  = typename T::D_ACC;

    if (num_kv_tiles <= 0) return;

    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    const int wave_kv_base = warp_id * T::ROWS_PER_WAVE;

    auto s_k_nope = make_smem(reinterpret_cast<D_NOPE*>(smem_buf + wave_kv_base * T::K_NOPE_ROW_LDS_BYTES));
    auto s_k_rope = make_smem(reinterpret_cast<D_ROPE*>(smem_buf + T::K_ROPE_TILE_OFF + wave_kv_base * T::K_ROPE_ROW_LDS_BYTES));
    auto s_v_wr   = make_smem(reinterpret_cast<D_ROPE*>(smem_buf + T::V_LDS_OFF + wave_kv_base * T::V_ROW_LDS_BYTES));
    auto s_v_rd   = make_smem(reinterpret_cast<D_ROPE*>(smem_buf + T::V_LDS_OFF) + warp_id * (T::D_HEAD_SIZE / T::T_N));
    auto s_m = make_smem(reinterpret_cast<D_ACC*>(smem_buf + T::M_LDS_OFF));
    auto s_p = make_smem(reinterpret_cast<D_ROPE*>(smem_buf + T::P_LDS_OFF));

    int buf_delta = T::K_BUF_BYTES;
    auto advance_bufs = [&]() {
        s_k_nope.ptr += buf_delta;
        s_k_rope.ptr += buf_delta;
        buf_delta = -buf_delta;
    };

    int v_wr_delta = T::V_BUF_BYTES, v_rd_delta = T::V_BUF_BYTES;
    auto advance_v_wr = [&]() { s_v_wr.ptr += v_wr_delta; v_wr_delta = -v_wr_delta; };
    auto advance_v_rd = [&]() { s_v_rd.ptr += v_rd_delta; v_rd_delta = -v_rd_delta; };
    static_assert(T::NUM_V_BUFS == 2, "the pipelined round alternates between exactly two V slots");

    const u32x4_t kv_indices_rsrc = make_buffer_rsrc_raw(kv_indices + page_idx_begin, (u32_t)(valid_kv_len * sizeof(int)));

    using k_nope_window = tdm<D_NOPE, seq<T::D_NOPE_PADDED_SIZE, T::INDICES_PER_TDM>,
                              tdm_traits::gather<32>,
                              tdm_traits::cache<tdm_traits::make_cache_policy(
                                  tdm_traits::load_temporal_hint::regular, tdm_traits::scope::cu)>,
                              tdm_traits::padding_auto<D_NOPE, T::D_NOPE_PADDED_SIZE>>;
    using k_rope_window = tdm<D_ROPE, seq<T::D_ROPE_SIZE, T::INDICES_PER_TDM>,
                              tdm_traits::gather<32>,
                              tdm_traits::cache<tdm_traits::make_cache_policy(
                                  tdm_traits::load_temporal_hint::regular, tdm_traits::scope::cu)>,
                              tdm_traits::padding_auto<D_ROPE, T::D_ROPE_SIZE>>;

    auto tdm_k_nope = make_tdm<k_nope_window>(
        (u32_t)reinterpret_cast<uintptr_t>(smem_buf + wave_kv_base * T::K_NOPE_ROW_LDS_BYTES),
        reinterpret_cast<const D_NOPE*>(kv_nope_ptr),
        /*shape0=*/ (u32_t)T::D_NOPE_PADDED_SIZE,
        /*shape1=*/ (u32_t)kv_rows,
        /*stride=*/ (u64_t)kargs.stride_kv_nope_page);

    auto tdm_k_rope = make_tdm<k_rope_window>(
        (u32_t)reinterpret_cast<uintptr_t>(smem_buf + T::K_ROPE_TILE_OFF + wave_kv_base * T::K_ROPE_ROW_LDS_BYTES),
        reinterpret_cast<const D_ROPE*>(kv_rope_ptr),
        /*shape0=*/ (u32_t)T::D_ROPE_SIZE,
        /*shape1=*/ (u32_t)kv_rows,
        /*stride=*/ (u64_t)kargs.stride_kv_rope_page);

    auto load_row_ids = [&](int tile_idx) {
        const int idx_byte_off = (tile_idx * T::KV_TILE_SIZE + wave_kv_base) * (int)sizeof(int);
        return s_buffer_load_b512(kv_indices_rsrc, idx_byte_off);
    };

    constexpr int nope_lds_step  = T::INDICES_PER_TDM * T::K_NOPE_ROW_LDS_ELEMS;
    constexpr int rope_lds_step  = T::INDICES_PER_TDM * T::K_ROPE_ROW_LDS_ELEMS;
    constexpr int nope_slot_step = T::K_BUF_BYTES / (int)sizeof(D_NOPE);
    constexpr int rope_slot_step = T::K_BUF_BYTES / (int)sizeof(D_ROPE);

    auto issue_kv_tile = [&](const u32x16_t& ids, int tile_idx, u32_t slot, auto clamp_tail) {
        [[maybe_unused]] const int wave_valid = valid_kv_len - (tile_idx * T::KV_TILE_SIZE + wave_kv_base);

        static_for<T::TDM_LOADS_PER_WAVE>([&](auto d) {
            constexpr int ld = d.value;
            u32_t idx[T::INDICES_PER_TDM];
            static_for<T::INDICES_PER_TDM>([&](auto r) {
                constexpr int slot_r = ld * T::INDICES_PER_TDM + r.value;
                u32_t id = ids[slot_r];
                if constexpr (decltype(clamp_tail)::value) id = slot_r < wave_valid ? id : (u32_t)kv_rows;
                idx[r.value] = __builtin_amdgcn_readfirstlane(id);
            });
            tdm_k_nope.set_indices(idx, T::INDICES_PER_TDM);
            tdm_k_rope.set_indices(idx, T::INDICES_PER_TDM);
            tdm_k_nope.async_load(slot * (u32_t)nope_slot_step + u32_t(ld * nope_lds_step));
            tdm_k_rope.async_load(slot * (u32_t)rope_slot_step + u32_t(ld * rope_lds_step));
        });
    };

    auto mma0_nope = make_tiled_mma<D_NOPE, D_NOPE, D_ACC>(
        seq<1, T::GEMM0_E_N, T::GEMM0_NOPE_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K_NOPE>{},
        wmma_adaptor_swap_ab{});

    auto mma0_rope = make_tiled_mma<D_ROPE, D_ROPE, D_ACC>(
        seq<1, T::GEMM0_E_N, T::GEMM0_ROPE_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K_ROPE>{},
        wmma_adaptor_swap_ab{});

    auto mma1 = make_tiled_mma<D_ROPE, D_ROPE, D_ACC>(
        seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K_ROPE>{},
        wmma_adaptor_swap_ab{});

    auto u_rk_nope  = make_layout_operand<T, T::GEMM0_E_N, T::GEMM0_NOPE_E_K, T::W_K_NOPE, T::VEC_NOPE, T::K_NOPE_ROW_LDS_ELEMS>(lane_id);
    auto u_rk_rope  = make_layout_operand<T, T::GEMM0_E_N, T::GEMM0_ROPE_E_K, T::W_K_ROPE, T::VEC_ROPE, T::K_ROPE_ROW_LDS_ELEMS>(lane_id);
    auto u_rk_mxscl = make_layout_mxscl<T, T::GEMM0_E_N, T::K_NOPE_ROW_LDS_ELEMS>(lane_id);
    auto u_wv_nope  = make_layout_v_nope<T>(lane_id);
    auto u_wv_rope  = make_layout_operand<T, T::GEMM0_E_N, T::GEMM0_ROPE_E_K, T::W_K_ROPE, T::VEC_ROPE, T::V_ROW_LDS_ELEMS>(lane_id);
    auto u_rv       = make_layout_v<T>(lane_id);

    constexpr D_ACC RESCALE_THRESHOLD = D_ACC(8.0f);
    constexpr index_t s_len_per_m = T::Q_TILE_SIZE * T::KV_TILE_SIZE / (T::T_N * T::WARP_SIZE) / T::GEMM0_E_M;
    constexpr index_t o_len_per_m = T::Q_TILE_SIZE * T::D_HEAD_SIZE / (T::T_N * T::WARP_SIZE) / T::GEMM1_E_M;
    constexpr index_t q_scale_dwords = vector_traits<VQS>::size() * sizeof(D_NOPE) / sizeof(int);
    constexpr index_t k_scale_dwords = q_scale_dwords / T::GEMM0_E_M;

    typename decltype(mma0_nope)::vtype_b v_k_nope;
    typename decltype(mma0_rope)::vtype_b v_k_rope;
    vector_t<D_NOPE, T::GEMM0_E_N * T::W_N * (T::D_NOPE_PADDED_SIZE / T::MXSCL_BLOCK_SIZE) / T::WARP_SIZE> v_k_mxscl;
    vector_t<D_ACC, T::GEMM0_E_M * s_len_per_m> v_s, v_s_next;
    typename decltype(mma1)::vtype_b v_v;
    typename decltype(mma1)::vtype_a v_p;

    auto& scale_q  = reinterpret_cast<vector_t<int, q_scale_dwords>&>(v_q_mxscl);
    auto& scale_k  = reinterpret_cast<vector_t<int, k_scale_dwords>&>(v_k_mxscl);
    auto* q_nope_m = reinterpret_cast<typename decltype(mma0_nope)::vtype_a*>(&v_q_nope);
    auto* q_rope_m = reinterpret_cast<typename decltype(mma0_rope)::vtype_a*>(&v_q_rope);

    u32x16_t row_ids = load_row_ids(0);
    u32_t gather_slot = 0;

    auto issue_tile = [&](int tile) {
        s_wait_kmcnt_for(row_ids);
        if (tile + 1 < num_kv_tiles) issue_kv_tile(row_ids, tile, gather_slot, false_type{});
        else                         issue_kv_tile(row_ids, tile, gather_slot, true_type{});
        row_ids = load_row_ids(tile + 1);
        gather_slot ^= 1u;
    };

    auto publish_v_rows = [&]() {
        vector_t<D_ROPE, T::V_NOPE_BLOCKS * T::VEC_NOPE> v_v_nope;
        auto* src = reinterpret_cast<const vector_t<u32_t, 2>*>(&v_k_nope);
        auto* dst = reinterpret_cast<vector_t<D_ROPE, 8>*>(&v_v_nope);
        static_for<T::V_NOPE_BLOCKS>([&](auto b) {
            constexpr int dw   = b.value / 8;
            constexpr int half = (b.value / 4) % 2;
            constexpr int byte = b.value % 4;
            constexpr int sel  = ((byte & 1) << 2) | (byte & 2) | half;
            dst[2 * b.value + 0] = __builtin_amdgcn_cvt_scale_pk8_bf16_fp8(src[2 * b.value + 0], scale_k[dw], sel);
            dst[2 * b.value + 1] = __builtin_amdgcn_cvt_scale_pk8_bf16_fp8(src[2 * b.value + 1], scale_k[dw], sel);
        });
        store<T::VEC_NOPE>(s_v_wr, v_v_nope, u_wv_nope);
        store<T::VEC_ROPE>(s_v_wr, v_k_rope, u_wv_rope + number<T::D_NOPE_SIZE>{});
        advance_v_wr();
    };

    auto load_k = [&]() {
        v_k_nope  = load<T::VEC_NOPE>(s_k_nope, u_rk_nope);
        v_k_mxscl = load<T::VEC_MXSCL>(s_k_nope, u_rk_mxscl + number<T::D_NOPE_SIZE>{});
        v_k_rope  = load<T::VEC_ROPE>(s_k_rope, u_rk_rope);
        zero_nope_pad<T, T::GEMM0_E_N>(v_k_nope);
        zero_mxscl_pad<T, T::GEMM0_E_N>(v_k_mxscl, lane_id >= T::W_N);
        advance_bufs();
    };

    auto compute_qk = [&](auto& dst) __attribute__((always_inline)) {
        auto* blk = reinterpret_cast<typename decltype(mma0_nope)::vtype_c*>(&dst);
        static_for<T::GEMM0_E_M>([&](auto m) {
            clear(blk[m.value]);
            static_for<T::GEMM0_NOPE_E_K>([&](auto ek) {
                constexpr int sel = ek.value % 2;
                constexpr int dwq = m.value * k_scale_dwords + ek.value / 2;
                constexpr int dwk = ek.value / 2;
                blk[m.value] = mma0_nope.step_k(ek, q_nope_m[m.value], v_k_nope, blk[m.value],
                                                scale_q[dwq], scale_k[dwk], number<sel>{}, number<sel>{});
            });
            blk[m.value] = mma0_rope(q_rope_m[m.value], v_k_rope, blk[m.value]);
        });
    };

    issue_tile(0);
    issue_tile(1);
    s_wait_tensorcnt(number<T::TDM_OPS_PER_TILE>{});
    load_k();
    publish_v_rows();
    compute_qk(v_s);
    attn_mask_oob_score<T>(v_s, valid_kv_len, 0, wave_kv_base, lane_id);
    ml_publish<T>(s_m, attn_row_max_blocks<T>(v_s), warp_id, lane_id);
    s_wait_dscnt(0_I);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);

    auto round = [&](int tile, auto mask_next) __attribute__((always_inline)) {
        auto tile_max = ml_reduce<T>(s_m, lane_id, [](D_ACC x, D_ACC y) { return max(x, y); });
        bool below = true;
        static_for<T::GEMM0_E_M>([&](auto i) {
            tile_max[i.value] *= temperature_scale;
            below = below && ((tile_max[i.value] - m_row[i.value]) <= RESCALE_THRESHOLD);
        });
        const bool all_below = __builtin_amdgcn_ballot_w32(below) == __builtin_amdgcn_read_exec_lo();

        vector_t<D_ACC, T::GEMM0_E_M> row_max;
        static_for<T::GEMM0_E_M>([&](auto i) {
            row_max[i.value] = all_below ? m_row[i.value] : max(m_row[i.value], tile_max[i.value]);
        });
        if (!all_below) {
            static_for<T::GEMM0_E_M>([&](auto i) {
                const D_ACC rescale_m = __builtin_amdgcn_exp2f(m_row[i.value] - row_max[i.value]);
                m_row[i.value] = row_max[i.value];
                l_part[i.value] *= rescale_m;
                scale_slice<T, i.value * o_len_per_m, o_len_per_m>(v_o, rescale_m);
            });
        }
        static_for<T::GEMM0_E_M>([&](auto i) {
            attn_row_scale_sub<T, i.value * s_len_per_m, s_len_per_m>(v_s, temperature_scale, row_max[i.value]);
        });
        attn_exp2<T>(v_s);
        static_for<T::GEMM0_E_M>([&](auto i) {
            l_part[i.value] += attn_row_sum<T, i.value * s_len_per_m, s_len_per_m>(v_s);
        });
        store_p<T>(s_p, v_s, warp_id, lane_id);

        v_v = tr_load<T::VEC_ROPE>(s_v_rd, u_rv);
        advance_v_rd();
        s_wait_dscnt(number<T::v_ds_load_insts>{});
        __builtin_amdgcn_s_barrier_signal(-1);

        s_wait_tensorcnt(number<T::TDM_OPS_PER_TILE>{});
        load_k();
        publish_v_rows();
        compute_qk(v_s_next);
        if constexpr (decltype(mask_next)::value) {
            attn_mask_oob_score<T>(v_s_next, valid_kv_len, tile + 1, wave_kv_base, lane_id);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier_wait(-1);

        gather_p<T>(s_p, v_p, lane_id);
        v_s = v_s_next;
        ml_publish<T>(s_m, attn_row_max_blocks<T>(v_s), warp_id, lane_id);
        s_wait_dscnt(0_I);
        __builtin_amdgcn_s_barrier_signal(-1);

        v_o = mma1(v_p, v_v, v_o);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier_wait(-1);
    };

    int tile = 0;
    for (; tile + 2 < num_kv_tiles; ++tile) {
        issue_tile(tile + 2);
        round(tile, false_type{});
    }

    #pragma clang loop unroll(disable)
    for (; tile < num_kv_tiles; ++tile) {
        issue_tile(tile + 2);
        round(tile, true_type{});
    }
}

}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void opus_mla_v4_prefill_a8w8_32mx1_16nx4_kernel(opus_mla_v4_prefill_fp8_kargs kargs) {
    using namespace opus;
    using namespace opus_mla_v4_prefill_a8w8_32mx1_16nx4;
    using T = opus::remove_cvref_t<Traits>;
    using D_NOPE = typename T::D_NOPE;
    using D_ROPE = typename T::D_ROPE;
    using D_ACC  = typename T::D_ACC;
    using D_OUT  = typename T::D_OUT;

    const int q_token_idx = block_id_x();
    const int h_block_idx = block_id_y();
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    const int h_block_start = h_block_idx * T::Q_TILE_SIZE;
    const u32_t qo_head_origin = (u32_t)(h_block_start + warp_id * T::QO_ROWS_PER_WAVE);

    __shared__ char smem_buf[T::smem_size_bytes()];
    char* const qo_tile = smem_buf;

    constexpr index_t o_len_per_m = T::Q_TILE_SIZE * T::D_HEAD_SIZE / (T::T_N * T::WARP_SIZE) / T::GEMM1_E_M;
    constexpr D_ACC LOG2_E = 1.44269504089f;
    const D_ACC temperature_scale = kargs.softmax_scale * LOG2_E;

    auto s_q_nope = make_smem(reinterpret_cast<D_NOPE*>(qo_tile));
    auto s_q_rope = make_smem(reinterpret_cast<D_ROPE*>(qo_tile + T::Q_ROPE_TILE_OFF));

    vector_t<D_NOPE, T::GEMM0_E_M * T::GEMM0_NOPE_E_K * T::W_M * T::W_K_NOPE / T::WARP_SIZE> v_q_nope;
    vector_t<D_ROPE, T::GEMM0_E_M * T::GEMM0_ROPE_E_K * T::W_M * T::W_K_ROPE / T::WARP_SIZE> v_q_rope;
    vector_t<D_NOPE, T::GEMM0_E_M * T::W_M * (T::D_NOPE_PADDED_SIZE / T::MXSCL_BLOCK_SIZE) / T::WARP_SIZE> v_q_mxscl;

    {
        using q_nope_window = tdm<D_NOPE, seq<T::D_NOPE_PADDED_SIZE, T::QO_ROWS_PER_WAVE>,
                                  tdm_traits::padding_auto<D_NOPE, T::D_NOPE_PADDED_SIZE>>;
        using q_rope_window = tdm<D_ROPE, seq<T::D_ROPE_SIZE, T::QO_ROWS_PER_WAVE>,
                                  tdm_traits::padding_auto<D_ROPE, T::D_ROPE_SIZE>>;

        auto tdm_q_nope = make_tdm<q_nope_window>(
            (u32_t)reinterpret_cast<uintptr_t>(qo_tile),
            reinterpret_cast<const D_NOPE*>(kargs.q_nope_ptr) + (int64_t)q_token_idx * kargs.stride_q_nope_n,
            /*shape0=*/ (u32_t)T::D_NOPE_PADDED_SIZE,
            /*shape1=*/ (u32_t)kargs.H,
            /*stride=*/ (u64_t)kargs.stride_q_nope_h,
            /*origin0=*/ 0u,
            /*origin1=*/ qo_head_origin);
        auto tdm_q_rope = make_tdm<q_rope_window>(
            (u32_t)reinterpret_cast<uintptr_t>(qo_tile + T::Q_ROPE_TILE_OFF),
            reinterpret_cast<const D_ROPE*>(kargs.q_rope_ptr) + (int64_t)q_token_idx * kargs.stride_q_rope_n,
            /*shape0=*/ (u32_t)T::D_ROPE_SIZE,
            /*shape1=*/ (u32_t)kargs.H,
            /*stride=*/ (u64_t)kargs.stride_q_rope_h,
            /*origin0=*/ 0u,
            /*origin1=*/ qo_head_origin);
        tdm_q_nope.async_load((u32_t)(warp_id * T::QO_ROWS_PER_WAVE * T::Q_NOPE_ROW_LDS_ELEMS));
        tdm_q_rope.async_load((u32_t)(warp_id * T::QO_ROWS_PER_WAVE * T::Q_ROPE_ROW_LDS_ELEMS));

        s_wait_tensorcnt(0_I);
        __builtin_amdgcn_s_barrier();
        v_q_nope  = load<T::VEC_NOPE>(s_q_nope, make_layout_operand<T, T::GEMM0_E_M, T::GEMM0_NOPE_E_K, T::W_K_NOPE, T::VEC_NOPE, T::Q_NOPE_ROW_LDS_ELEMS>(lane_id));
        v_q_mxscl = load<T::VEC_MXSCL>(s_q_nope, make_layout_mxscl<T, T::GEMM0_E_M, T::Q_NOPE_ROW_LDS_ELEMS>(lane_id) + number<T::D_NOPE_SIZE>{});
        v_q_rope  = load<T::VEC_ROPE>(s_q_rope, make_layout_operand<T, T::GEMM0_E_M, T::GEMM0_ROPE_E_K, T::W_K_ROPE, T::VEC_ROPE, T::Q_ROPE_ROW_LDS_ELEMS>(lane_id));
        s_wait_dscnt(0_I);
        __builtin_amdgcn_s_barrier();

        zero_nope_pad<T, T::GEMM0_E_M>(v_q_nope);
        zero_mxscl_pad<T, T::GEMM0_E_M>(v_q_mxscl, lane_id >= T::W_M);
    }

    vector_t<D_ACC, T::Q_TILE_SIZE * T::D_HEAD_SIZE / (T::T_N * T::WARP_SIZE)> v_o;
    clear(v_o);
    vector_t<D_ACC, T::GEMM0_E_M> m_row, l_part;
    static_for<T::GEMM0_E_M>([&](auto i) {
        m_row[i.value] = opus::numeric_limits<D_ACC>::lowest();
        l_part[i.value] = D_ACC(0.0f);
    });

    {
        const int page_idx_begin = kargs.kv_indptr_prefix[q_token_idx];
        const int valid_kv_len   = kargs.kv_indptr_prefix[q_token_idx + 1] - page_idx_begin;
        const int num_kv_tiles   = ceil_div(valid_kv_len, T::KV_TILE_SIZE);
        mla_v4_prefill_accum_pipelined<Traits>(kargs, kargs.unified_kv_nope_ptr, kargs.unified_kv_rope_ptr, kargs.total_pages,
                                           kargs.kv_indices_prefix, page_idx_begin, valid_kv_len, num_kv_tiles,
                                           smem_buf, v_q_nope, v_q_rope, v_q_mxscl, v_o, m_row, l_part, temperature_scale);
    }

    {
        const int page_idx_begin = kargs.kv_indptr_extend[q_token_idx];
        const int valid_kv_len   = kargs.kv_indptr_extend[q_token_idx + 1] - page_idx_begin;
        const int num_kv_tiles   = ceil_div(valid_kv_len, T::KV_TILE_SIZE);
        mla_v4_prefill_accum_pipelined<Traits>(kargs, kargs.kv_nope_ptr, kargs.kv_rope_ptr, kargs.total_tokens,
                                           kargs.kv_indices_extend, page_idx_begin, valid_kv_len, num_kv_tiles,
                                           smem_buf, v_q_nope, v_q_rope, v_q_mxscl, v_o, m_row, l_part, temperature_scale);
    }

    {
        auto s_l = make_smem(reinterpret_cast<D_ACC*>(smem_buf + T::L_LDS_OFF));
        ml_publish<T>(s_l, l_part, warp_id, lane_id);
        s_wait_dscnt(0_I);
        __builtin_amdgcn_s_barrier();
        l_part = ml_reduce<T>(s_l, lane_id, [](D_ACC x, D_ACC y) { return x + y; });
    }

    auto g_attn_sink = make_gmem(reinterpret_cast<const D_ACC*>(kargs.attn_sink_ptr), kargs.H * sizeof(D_ACC));
    static_for<T::GEMM0_E_M>([&](auto i) {
        const int sink_head_idx = h_block_start + i.value * T::W_M + (lane_id % T::W_M);
        const D_ACC sink_log2 = load(g_attn_sink, sink_head_idx)[0] * LOG2_E;
        const D_ACC m_final = max(m_row[i.value], sink_log2);
        const D_ACC alpha   = __builtin_amdgcn_exp2f(m_row[i.value] - m_final);
        const D_ACC l_final = l_part[i.value] * alpha + __builtin_amdgcn_exp2f(sink_log2 - m_final);
        const D_ACC o_scale = (l_final > D_ACC(0.0f)) ? (alpha / l_final) : D_ACC(0.0f);
        scale_slice<T, i.value * o_len_per_m, o_len_per_m>(v_o, o_scale);
    });

    {
        using o_window = tdm<D_OUT, seq<T::O_ROW_LDS_ELEMS, T::QO_ROWS_PER_WAVE>>;

        s_wait_tensorcnt(0_I);
        s_wait_dscnt(0_I);
        __builtin_amdgcn_s_barrier();

        auto s_o = make_smem(reinterpret_cast<D_OUT*>(qo_tile));
        store<T::VEC_O>(s_o, cast<D_OUT>(v_o), make_layout_o<T>(warp_id, lane_id));

        auto tdm_o = make_tdm<o_window>(
            (u32_t)reinterpret_cast<uintptr_t>(qo_tile),
            reinterpret_cast<D_OUT*>(kargs.out_ptr) + (int64_t)q_token_idx * kargs.stride_o_n,
            /*shape0=*/ (u32_t)T::D_HEAD_SIZE,
            /*shape1=*/ (u32_t)kargs.H,
            /*stride=*/ (u64_t)kargs.stride_o_h,
            /*origin0=*/ 0u,
            /*origin1=*/ qo_head_origin);

        s_wait_dscnt(0_I);
        __builtin_amdgcn_s_barrier();
        tdm_o.async_store((u32_t)(warp_id * T::QO_ROWS_PER_WAVE * T::O_ROW_LDS_ELEMS));
        s_wait_tensorcnt(0_I);
    }
}
