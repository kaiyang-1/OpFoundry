#pragma once

#include <opus/opus.hpp>
#include "pa_traits.h"
#include <cstdint>
#include <bit>

using opus::operator""_I;

namespace pa_16mx4_64nx1_fp8 {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-inline"
OPUS_D opus::u32x16_t llvm_amdgcn_s_buffer_load_v16i32(opus::u32x4_t rsrc, int offset, int aux)
    __asm("llvm.amdgcn.s.buffer.load.v16i32");
#pragma clang diagnostic pop

OPUS_D opus::u32x4_t make_buffer_rsrc_raw(const void* ptr, opus::u32_t num_bytes,
                                          opus::u32_t config = opus::buffer_default_config()) {
    __amdgpu_buffer_rsrc_t rsrc = __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(ptr), /*stride=*/0, num_bytes, config);
    opus::u32x4_t raw;
    __builtin_memcpy(&raw, &rsrc, sizeof(raw));
    opus::static_for<4>([&](auto k) { raw[k.value] = __builtin_amdgcn_readfirstlane(raw[k.value]); });
    return raw;
}

template<class T>
__device__ inline auto make_layout_q_nope(int warp_id, int lane_id, int stride_q_nope_h) {
    constexpr auto q_block_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::D_NOPE_PADDED_SIZE / T::W_K_NOPE>{},
        opus::number<T::W_M * T::W_K_NOPE / (T::WARP_SIZE * T::VEC_NOPE)>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_NOPE>{});

    constexpr auto q_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        q_block_shape,
        opus::unfold_x_stride(q_block_dim, q_block_shape, opus::tuple{stride_q_nope_h, 1_I}),
        opus::unfold_p_coord(q_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

template<class T>
__device__ inline auto make_layout_q_rope(int warp_id, int lane_id, int stride_q_rope_h) {
    constexpr auto q_block_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::D_ROPE_SIZE / T::W_K_ROPE>{},
        opus::number<T::W_M * T::W_K_ROPE / (T::WARP_SIZE * T::VEC_ROPE)>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_ROPE>{});

    constexpr auto q_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        q_block_shape,
        opus::unfold_x_stride(q_block_dim, q_block_shape, opus::tuple{stride_q_rope_h, 1_I}),
        opus::unfold_p_coord(q_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

template<class T>
__device__ inline auto make_layout_q_mxscl(int warp_id, int lane_id, int stride_q_nope_h) {
    constexpr int MXSCL_BLOCK_SIZE = 32;

    constexpr auto q_block_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::W_M * T::D_NOPE_PADDED_SIZE / MXSCL_BLOCK_SIZE / (T::WARP_SIZE * T::VEC_MXSCL)>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_MXSCL>{});

    constexpr auto q_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        q_block_shape,
        opus::unfold_x_stride(q_block_dim, q_block_shape, opus::tuple{stride_q_nope_h, 1_I}),
        opus::unfold_p_coord(q_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

template<class T, int RowLdsElems>
__device__ inline auto make_layout_rkv_nope(int lane_id) {
    constexpr auto k_block_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_N>{},
        opus::number<T::W_N>{},
        opus::number<T::GEMM0_NOPE_E_K>{},
        opus::number<T::W_N * T::W_K_NOPE / (T::WARP_SIZE * T::VEC_NOPE)>{},
        opus::number<T::WARP_SIZE / T::W_N>{},
        opus::number<T::VEC_NOPE>{});

    constexpr auto k_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        k_block_shape,
        opus::unfold_x_stride(k_block_dim, k_block_shape, opus::tuple{opus::number<RowLdsElems>{}, 1_I}),
        opus::unfold_p_coord(k_block_dim, opus::tuple{lane_id % T::W_N, lane_id / T::W_N}));
}

template<class T, int RowLdsElems>
__device__ inline auto make_layout_rkv_rope(int lane_id) {
    constexpr auto k_block_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_N>{},
        opus::number<T::W_N>{},
        opus::number<T::GEMM0_ROPE_E_K>{},
        opus::number<T::W_N * T::W_K_ROPE / (T::WARP_SIZE * T::VEC_ROPE)>{},
        opus::number<T::WARP_SIZE / T::W_N>{},
        opus::number<T::VEC_ROPE>{});

    constexpr auto k_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        k_block_shape,
        opus::unfold_x_stride(k_block_dim, k_block_shape, opus::tuple{opus::number<RowLdsElems>{}, 1_I}),
        opus::unfold_p_coord(k_block_dim, opus::tuple{lane_id % T::W_N, lane_id / T::W_N}));
}

template<class T>
__device__ inline auto make_layout_rk_mxscl(int lane_id) {
    constexpr int MXSCL_BLOCK_SIZE = 32;

    constexpr auto k_block_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_N>{},
        opus::number<T::W_N>{},
        opus::number<T::W_N * T::D_NOPE_PADDED_SIZE / MXSCL_BLOCK_SIZE / (T::WARP_SIZE * T::VEC_MXSCL)>{},
        opus::number<T::WARP_SIZE / T::W_N>{},
        opus::number<T::VEC_MXSCL>{});
    
    constexpr auto k_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        k_block_shape,
        opus::unfold_x_stride(k_block_dim, k_block_shape, opus::tuple{opus::number<T::K_NOPE_ROW_LDS_ELEMS>{}, 1_I}),
        opus::unfold_p_coord(k_block_dim, opus::tuple{lane_id % T::W_N, lane_id / T::W_N}));
}

template<class T>
__device__ inline auto make_layout_rv(int lane_id) {
    constexpr int lane_per_grp = 16;
    constexpr int lane_n = 2;
    constexpr int lane_k = lane_per_grp / lane_n;

    constexpr int dwordx32_rpt = 4 * 32 / sizeof(typename T::D_ROPE) / T::VEC_ROPE;

    constexpr auto v_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_N / dwordx32_rpt>{},
        opus::number<lane_n>{},
        opus::number<dwordx32_rpt>{},
        opus::number<T::W_K_ROPE / (T::WARP_SIZE / lane_per_grp) / T::VEC_ROPE>{},
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
__device__ inline auto make_layout_o(int warp_id, int lane_id, int stride_o_h) {
    constexpr int dwordx32_rpt = 4 * 32 / sizeof(typename T::D_OUT) / T::VEC_O;

    constexpr auto o_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::GEMM1_STAGE_N>{},
        opus::number<T::GEMM1_E_N / dwordx32_rpt>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<dwordx32_rpt>{},
        opus::number<T::VEC_O>{});

    constexpr auto o_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}));

    return opus::make_layout(
        o_block_shape,
        opus::unfold_x_stride(o_block_dim, o_block_shape, opus::tuple{stride_o_h, 1_I}),
        opus::unfold_p_coord(o_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

template<class T, class V>
__device__ inline void zero_nope_pad(V& v_nope) {
    constexpr opus::index_t len  = opus::vector_traits<V>::size();
    constexpr opus::index_t vals = len * T::D_NOPE_SIZE / T::D_NOPE_PADDED_SIZE;
    opus::static_for([&](auto i) { v_nope[i.value] = static_cast<typename T::D_NOPE>(0); },
                     opus::number<vals>{}, opus::number<len>{});
}

template<class T, class V>
__device__ inline void zero_mxscl_pad(V& v_mxscl, bool upper_half) {
    constexpr int MXSCL_BLOCK_SIZE = 32;
    constexpr opus::index_t len = opus::vector_traits<V>::size();
    constexpr opus::index_t pad = (T::D_NOPE_PADDED_SIZE - T::D_NOPE_SIZE) / MXSCL_BLOCK_SIZE;
    opus::static_for<pad>([&](auto i) {
        constexpr opus::index_t idx = len - pad + i.value;
        v_mxscl[idx] = upper_half ? static_cast<typename T::D_NOPE>(0) : v_mxscl[idx];
    });
}

template<int Lo, int Hi, typename V, typename Op>
__device__ inline auto tree_reduce(const V& v, Op op) {
    if constexpr (Hi - Lo == 1) return v[Lo];
    else {
        constexpr int Mid = (Lo + Hi) / 2;
        return op(tree_reduce<Lo, Mid>(v, op), tree_reduce<Mid, Hi>(v, op));
    }
}

template<typename T, typename V>
__device__ inline typename T::D_ACC attn_row_max(const V& v_s) {
    using D_ACC = typename T::D_ACC;
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    D_ACC row_max = max(opus::numeric_limits<D_ACC>::lowest(),
                        tree_reduce<0, s_len>(v_s, [](D_ACC x, D_ACC y) { return max(x, y); }));

    opus::vector_t<opus::u32_t, 2> res16 = __builtin_amdgcn_permlane16_swap(std::bit_cast<opus::u32_t>(row_max), std::bit_cast<opus::u32_t>(row_max), false, true);
    return max(std::bit_cast<float>(res16.x), std::bit_cast<float>(res16.y));
}

template<typename T, typename V>
__device__ inline void attn_row_scale_sub(V& v_s, typename T::D_ACC scale, typename T::D_ACC row_max) {
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    opus::static_for<s_len>([&](auto i) {
        v_s[i.value] = __builtin_fmaf(v_s[i.value], scale, -row_max);
    });
}

template<typename T, opus::index_t Offset, opus::index_t Count, typename V>
__device__ inline void attn_exp2_slice(V& v_s) {
    opus::static_for<Count>([&](auto i) {
        constexpr opus::index_t idx = Offset + i.value;
        v_s[idx] = __builtin_amdgcn_exp2f(v_s[idx]);
    });
}

template<typename T, typename V>
__device__ inline typename T::D_ACC attn_row_sum(const V& v_s) {
    using D_ACC = typename T::D_ACC;
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    D_ACC row_sum = tree_reduce<0, s_len>(v_s, [](D_ACC x, D_ACC y) { return x + y; });

    opus::vector_t<opus::u32_t, 2> res16 = __builtin_amdgcn_permlane16_swap(std::bit_cast<opus::u32_t>(row_sum), std::bit_cast<opus::u32_t>(row_sum), false, true);
    return std::bit_cast<float>(res16.x) + std::bit_cast<float>(res16.y);
}

template<typename T, typename V>
__device__ inline void scale_output_tile(V& v_o, typename T::D_ACC scale) {
    constexpr opus::index_t o_len = opus::vector_traits<V>::size();
    opus::static_for<o_len>([&](auto i) { v_o[i.value] *= scale; });
}

template<typename T, typename V>
__device__ inline void attn_mask_oob_score(V& v_s, int valid_kv_len, int kv_tile_idx, int seg_base_rows) {
    using D_ACC = typename T::D_ACC;

    if ((kv_tile_idx + 1) * T::KV_TILE_SIZE <= valid_kv_len) return;

    constexpr int elems_per_stage = (T::W_M * T::W_N) / T::WARP_SIZE;
    constexpr int lane_hi_kv_step = T::W_N / (T::WARP_SIZE / T::W_M);
    static_assert(opus::vector_traits<V>::size() == T::GEMM0_STAGE_N * elems_per_stage);
    static_assert((T::KV_TILE_SIZE & (T::KV_TILE_SIZE - 1)) == 0);

    const D_ACC neg_inf = -opus::numeric_limits<D_ACC>::infinity();
    const int lane_hi = (opus::thread_id_x() % T::WARP_SIZE) / T::W_M;   // {0, 1} on wave32
    const int rel_base = (valid_kv_len - 1) - kv_tile_idx * T::KV_TILE_SIZE - lane_hi * lane_hi_kv_step;

    opus::static_for<T::GEMM0_STAGE_N>([&](auto i_s) {
        const int rel = rel_base - ((i_s.value * T::W_N) ^ seg_base_rows);
        opus::static_for<elems_per_stage>([&](auto i_reg) {
            constexpr int idx = i_s.value * elems_per_stage + i_reg.value;
            v_s[idx] = (i_reg.value > rel) ? neg_inf : v_s[idx];
        });
    });
}

template<class Traits, class VQN, class VQR, class VQS, class VO>
__device__ __attribute__((always_inline)) void pa_prefill_accum_pipelined(pa_fp8_kargs kargs,
                                           const void* kv_nope_ptr, const void* kv_rope_ptr, int kv_rows,
                                           const int* kv_indices, int page_idx_begin, int valid_kv_len, int num_kv_tiles,
                                           char* smem_kv_buf,
                                           VQN& v_q_nope, VQR& v_q_rope, VQS& v_q_mxscl, VO& v_o,
                                           typename Traits::D_ACC& m_row,
                                           typename Traits::D_ACC& l_row,
                                           typename Traits::D_ACC temperature_scale) {
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;
    using D_NOPE = typename T::D_NOPE;
    using D_ROPE = typename T::D_ROPE;
    using D_ACC  = typename T::D_ACC;

    if (num_kv_tiles <= 0) return;

    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    //   wave 0: 0 1 2 3   wave 1: 2 3 0 1   wave 2: 1 0 3 2   wave 3: 3 2 1 0
    const int wave_rot     = ((warp_id & 1) << 1) | (warp_id >> 1);
    const int seg_rot_rows = wave_rot * T::W_N;
    const int seg_base_rows = (seg_rot_rows / T::ROWS_PER_SEG) * T::ROWS_PER_SEG;
    auto stage_row  = [&](int j) { return (j * T::W_N) ^ seg_rot_rows; };
    auto nope_stage = [&](int j) { const int r = stage_row(j);
        return (r / T::ROWS_PER_SEG) * T::SEG_BYTES + (r % T::ROWS_PER_SEG) * T::K_NOPE_ROW_LDS_BYTES; };
    auto rope_stage = [&](int j) { const int r = stage_row(j);
        return (r / T::ROWS_PER_SEG) * T::SEG_BYTES + T::K_NOPE_SEG_BYTES + (r % T::ROWS_PER_SEG) * T::K_ROPE_ROW_LDS_BYTES; };
    auto v_stage    = [&](int j) { const int r = stage_row(j);
        return (r / T::ROWS_PER_SEG) * T::SEG_BYTES + (r % T::ROWS_PER_SEG) * T::V_ROW_LDS_BYTES; };

    smem<D_NOPE> s_k_nope[T::GEMM0_STAGE_N] = {
        make_smem(reinterpret_cast<D_NOPE*>(smem_kv_buf + nope_stage(0))),
        make_smem(reinterpret_cast<D_NOPE*>(smem_kv_buf + nope_stage(1))),
        make_smem(reinterpret_cast<D_NOPE*>(smem_kv_buf + nope_stage(2))),
        make_smem(reinterpret_cast<D_NOPE*>(smem_kv_buf + nope_stage(3))),
    };
    smem<D_ROPE> s_k_rope[T::GEMM0_STAGE_N] = {
        make_smem(reinterpret_cast<D_ROPE*>(smem_kv_buf + rope_stage(0))),
        make_smem(reinterpret_cast<D_ROPE*>(smem_kv_buf + rope_stage(1))),
        make_smem(reinterpret_cast<D_ROPE*>(smem_kv_buf + rope_stage(2))),
        make_smem(reinterpret_cast<D_ROPE*>(smem_kv_buf + rope_stage(3))),
    };
    smem<D_ROPE> s_v = make_smem(reinterpret_cast<D_ROPE*>(smem_kv_buf + v_stage(T::GEMM0_STAGE_N - 1)));

    const int pv_rot_bytes = (seg_base_rows / T::ROWS_PER_SEG) * T::SEG_BYTES;
    smem<D_ROPE> s_pv[T::GEMM1_STAGE_K] = {
        make_smem(reinterpret_cast<D_ROPE*>(smem_kv_buf + pv_rot_bytes)),
        make_smem(reinterpret_cast<D_ROPE*>(smem_kv_buf + (T::SEG_BYTES - pv_rot_bytes))),
    };

    auto u_rk_nope  = make_layout_rkv_nope<T, T::K_NOPE_ROW_LDS_ELEMS>(lane_id);
    auto u_rk_rope  = make_layout_rkv_rope<T, T::K_ROPE_ROW_LDS_ELEMS>(lane_id);
    auto u_rk_mxscl = make_layout_rk_mxscl<T>(lane_id);
    auto u_rv_nope  = make_layout_rkv_nope<T, T::V_ROW_LDS_ELEMS>(lane_id);
    auto u_rv_rope  = make_layout_rkv_rope<T, T::V_ROW_LDS_ELEMS>(lane_id);
    auto u_rv       = make_layout_rv<T>(lane_id);

    auto mma0_nope = make_tiled_mma<D_NOPE, D_NOPE, D_ACC>(
        seq<T::GEMM0_E_N, T::GEMM0_E_M, T::GEMM0_NOPE_E_K>{},
        seq<T::T_N, T::T_M, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K_NOPE>{});
    auto mma0_rope = make_tiled_mma<D_ROPE, D_ROPE, D_ACC>(
        seq<T::GEMM0_E_N, T::GEMM0_E_M, T::GEMM0_ROPE_E_K>{},
        seq<T::T_N, T::T_M, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K_ROPE>{});
    auto mma1 = make_tiled_mma<D_ROPE, D_ROPE, D_ACC>(
        seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K_ROPE>{},
        wmma_adaptor_swap_ab{});

    vector_t<D_NOPE, T::GEMM0_E_N * T::GEMM0_NOPE_E_K * T::W_N * T::W_K_NOPE / T::WARP_SIZE> v_k_nope;
    vector_t<D_ROPE, T::GEMM0_E_N * T::GEMM0_ROPE_E_K * T::W_N * T::W_K_ROPE / T::WARP_SIZE> v_k_rope;
    vector_t<D_NOPE, T::GEMM0_E_N * T::W_N * (T::D_NOPE_PADDED_SIZE / 32) / T::WARP_SIZE> v_k_mxscl;   // 32 = E8M0 block
    vector_t<D_ACC, T::Q_TILE_SIZE * T::KV_TILE_SIZE / T::WARP_SIZE> v_s;
    vector_t<D_ROPE, T::Q_TILE_SIZE * T::KV_TILE_SIZE / T::WARP_SIZE> v_p;
    vector_t<D_ROPE, T::GEMM0_E_N * T::W_N * T::D_NOPE_PADDED_SIZE / T::WARP_SIZE> v_v_nope;
    vector_t<D_ROPE, T::GEMM1_E_N * T::GEMM1_E_K * T::W_N * T::W_K_ROPE / T::WARP_SIZE> v_v;

    auto v_p_stages = reinterpret_cast<vector_t<D_ROPE, T::W_M * T::W_K_ROPE / T::WARP_SIZE>*>(&v_p);
    auto v_o_stages = reinterpret_cast<vector_t<D_ACC, T::W_M * T::D_HEAD_SIZE / T::GEMM1_STAGE_N / T::WARP_SIZE>*>(&v_o);

    const u32x4_t kv_indices_rsrc = make_buffer_rsrc_raw(kv_indices + page_idx_begin, (u32_t)(valid_kv_len * sizeof(int)));

    constexpr tdm_cfg k_nope_gather_cfg{
        .tile_dim          = { (u32_t)T::D_NOPE_PADDED_SIZE, (u32_t)T::INDICES_PER_TDM },
        .gather            = true,
        .gather_index_size = 1,
        .lds_pad_en        = true,
        .pad_interval      = 6,
        .pad_amount        = 3,
    };
    constexpr tdm_cfg k_rope_gather_cfg{
        .tile_dim          = { (u32_t)T::D_ROPE_SIZE, (u32_t)T::INDICES_PER_TDM },
        .gather            = true,
        .gather_index_size = 1,
        .lds_pad_en        = true,
        .pad_interval      = 4,
        .pad_amount        = 3,
    };

    // Wave w gathers tile rows [w * ROWS_PER_WAVE, +ROWS_PER_WAVE), which land in one segment.
    const int wave_seg = (warp_id * T::ROWS_PER_WAVE / T::ROWS_PER_SEG) * T::SEG_BYTES;
    const int wave_row = (warp_id * T::ROWS_PER_WAVE) % T::ROWS_PER_SEG;

    auto tdm_k_nope = make_tdm<D_NOPE, k_nope_gather_cfg>(
        smem_kv_buf + wave_seg + wave_row * T::K_NOPE_ROW_LDS_BYTES,
        reinterpret_cast<const D_NOPE*>(kv_nope_ptr),
        /*lds_off=*/ 0,
        /*td0=*/ T::D_NOPE_PADDED_SIZE,
        /*td1=*/ kv_rows,
        /*s0=*/ kargs.stride_kv_nope_page);

    auto tdm_k_rope = make_tdm<D_ROPE, k_rope_gather_cfg>(
        smem_kv_buf + wave_seg + T::K_NOPE_SEG_BYTES + wave_row * T::K_ROPE_ROW_LDS_BYTES,
        reinterpret_cast<const D_ROPE*>(kv_rope_ptr),
        /*lds_off=*/ 0,
        /*td0=*/ T::D_ROPE_SIZE,
        /*td1=*/ kv_rows,
        /*s0=*/ kargs.stride_kv_rope_page);

    auto load_row_ids = [&](int tile_idx) {
        const int idx_byte_off = (tile_idx * T::KV_TILE_SIZE + warp_id * T::ROWS_PER_WAVE) * (int)sizeof(int);
        return llvm_amdgcn_s_buffer_load_v16i32(kv_indices_rsrc, idx_byte_off, /*aux=*/0);
    };

    constexpr int nope_lds_step = T::INDICES_PER_TDM * T::K_NOPE_ROW_LDS_BYTES;
    constexpr int rope_lds_step = T::INDICES_PER_TDM * T::K_ROPE_ROW_LDS_BYTES;

    auto issue_kv_tile = [&](const u32x16_t& ids, int tile_idx, auto clamp_tail) {
        [[maybe_unused]] const int wave_valid = valid_kv_len - (tile_idx * T::KV_TILE_SIZE + warp_id * T::ROWS_PER_WAVE);

        static_for<T::TDM_LOADS_PER_WAVE>([&](auto d) {
            constexpr int ld = d.value;
            static_for<T::INDICES_PER_TDM>([&](auto r) {
                constexpr int slot = ld * T::INDICES_PER_TDM + r.value;
                u32_t id = ids[slot];
                if constexpr (decltype(clamp_tail)::value) id = slot < wave_valid ? id : (u32_t)kv_rows;
                id = __builtin_amdgcn_readfirstlane(id);
                tdm_k_nope.set_gather_row_index(r.value, id);
                tdm_k_rope.set_gather_row_index(r.value, id);
            });
            tdm_k_nope.load();
            tdm_k_rope.load();
            if constexpr (ld + 1 < T::TDM_LOADS_PER_WAVE) {
                tdm_k_nope.move(0_I, 0_I, 0_I, 0_I, 0_I, number<nope_lds_step>{});
                tdm_k_rope.move(0_I, 0_I, 0_I, 0_I, 0_I, number<rope_lds_step>{});
            }
        });
        // Rewind the LDS window so the next tile starts at this wave's first row again.
        tdm_k_nope.move(0_I, 0_I, 0_I, 0_I, 0_I, number<-(T::TDM_LOADS_PER_WAVE - 1) * nope_lds_step>{});
        tdm_k_rope.move(0_I, 0_I, 0_I, 0_I, 0_I, number<-(T::TDM_LOADS_PER_WAVE - 1) * rope_lds_step>{});
    };

    auto compute_qk = [&](auto slot_swap) __attribute__((always_inline)) {
        constexpr int SLOT_SWAP = decltype(slot_swap)::value;
        constexpr index_t scale_dwords = vector_traits<VQS>::size() * sizeof(D_NOPE) / sizeof(int);
        using scale_t   = vector_t<int, scale_dwords>;
        using s_stage_t = vector_t<D_ACC, T::W_M * T::W_N / T::WARP_SIZE>;

        auto* v_s_stages = reinterpret_cast<s_stage_t*>(&v_s);
        auto& scale_q    = reinterpret_cast<scale_t&>(v_q_mxscl);

        static_for<T::GEMM0_STAGE_N>([&](auto j) {
            constexpr int slot = j.value ^ SLOT_SWAP;
            v_k_nope  = load<T::VEC_NOPE>(s_k_nope[j.value], u_rk_nope);
            v_k_mxscl = load<T::VEC_MXSCL>(s_k_nope[j.value], u_rk_mxscl + T::D_NOPE_SIZE);
            v_k_rope  = load<T::VEC_ROPE>(s_k_rope[j.value], u_rk_rope);
            s_wait_dscnt(0_I);
            zero_nope_pad<T>(v_k_nope);
            zero_mxscl_pad<T>(v_k_mxscl, lane_id >= T::W_N);

            auto& scale_k = reinterpret_cast<scale_t&>(v_k_mxscl);

            clear(v_s_stages[slot]);
            static_for<T::GEMM0_NOPE_E_K>([&](auto ek) {
                constexpr int sel = ek.value % 2;
                constexpr int dw  = ek.value / 2;
                v_s_stages[slot] = mma0_nope.step_k(ek, v_k_nope, v_q_nope, v_s_stages[slot],
                                                    scale_k[dw], scale_q[dw], number<sel>{}, number<sel>{});
            });
            v_s_stages[slot] = mma0_rope(v_k_rope, v_q_rope, v_s_stages[slot]);

            if constexpr (j.value == T::GEMM0_STAGE_N - 1) {
                constexpr int MXSCL_BLOCK_SIZE  = 32;
                constexpr index_t mxscl_per_row = T::D_NOPE_PADDED_SIZE / MXSCL_BLOCK_SIZE;   // 16 E8M0 bytes

                auto* src = reinterpret_cast<vector_t<u32_t, 2>*>(&v_k_nope);
                auto* dst = reinterpret_cast<vector_t<D_ROPE, 8>*>(&v_v_nope);
                static_for<mxscl_per_row>([&](auto b) {
                    constexpr int dw   = b.value / 8;
                    constexpr int half = (b.value / 4) % 2;
                    constexpr int byte = b.value % 4;
                    constexpr int sel  = ((byte & 1) << 2) | (byte & 2) | half;
                    dst[2 * b.value + 0] = __builtin_amdgcn_cvt_scale_pk8_bf16_fp8(src[2 * b.value + 0], scale_k[dw], sel);
                    dst[2 * b.value + 1] = __builtin_amdgcn_cvt_scale_pk8_bf16_fp8(src[2 * b.value + 1], scale_k[dw], sel);
                });
            }
        });
    };

    auto compute_pv = [&]() __attribute__((always_inline)) {
        static_for<T::GEMM1_STAGE_N * T::GEMM1_STAGE_K>([&](auto sg) {
            constexpr int stage_n = sg.value % T::GEMM1_STAGE_N;
            constexpr int stage_k = sg.value / T::GEMM1_STAGE_N;
            v_v = tr_load<T::VEC_ROPE>(s_pv[stage_k], u_rv + number<stage_n * (T::D_HEAD_SIZE / T::GEMM1_STAGE_N)>{});
            s_wait_dscnt(0_I);
            v_o_stages[stage_n] = mma1(v_p_stages[stage_k], v_v, v_o_stages[stage_n]);
        });
    };

    for (int tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        const u32x16_t row_ids = load_row_ids(tile_idx);
        s_wait_kmcnt(0_I);
        if (tile_idx + 1 < num_kv_tiles) issue_kv_tile(row_ids, tile_idx, false_type{});
        else                             issue_kv_tile(row_ids, tile_idx, true_type{});

        s_wait_tensorcnt(0_I);
        __builtin_amdgcn_s_barrier();

        if (wave_rot & 1) compute_qk(number<1>{});
        else              compute_qk(number<0>{});

        __builtin_amdgcn_s_barrier();
        store<T::VEC_NOPE>(s_v, v_v_nope, u_rv_nope);
        store<T::VEC_ROPE>(s_v, v_k_rope, u_rv_rope + T::D_NOPE_SIZE);

        attn_mask_oob_score<T>(v_s, valid_kv_len, tile_idx, seg_base_rows);
        constexpr index_t s_len = vector_traits<decltype(v_s)>::size();
        const D_ACC row_max   = max(m_row, attn_row_max<T>(v_s) * temperature_scale);
        const D_ACC rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
        m_row = row_max;
        attn_row_scale_sub<T>(v_s, temperature_scale, row_max);
        attn_exp2_slice<T, 0, s_len>(v_s);
        l_row = l_row * rescale_m + attn_row_sum<T>(v_s);
        v_p = cast<D_ROPE>(v_s);
        scale_output_tile<T>(v_o, rescale_m);

        s_wait_dscnt(0_I);
        __builtin_amdgcn_s_barrier();

        compute_pv();
        __builtin_amdgcn_s_barrier();
    }
}

} // namespace pa_16mx4_64nx1_fp8

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void pa_prefill_16mx4_64nx1_fp8_kernel(pa_fp8_kargs kargs) {
    using namespace opus;
    using namespace pa_16mx4_64nx1_fp8;
    using T = opus::remove_cvref_t<Traits>;
    using D_NOPE = typename T::D_NOPE;
    using D_ROPE = typename T::D_ROPE;
    using D_ACC  = typename T::D_ACC;

    const int q_token_idx = block_id_x();
    const int h_block_idx = block_id_y();
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    const int h_block_start = h_block_idx * T::NUM_WARPS * T::Q_TILE_SIZE;
    const int64_t q_nope_gmem_offset = static_cast<int64_t>(q_token_idx) * kargs.stride_q_nope_n + static_cast<int64_t>(h_block_start) * kargs.stride_q_nope_h;
    const int64_t q_rope_gmem_offset = static_cast<int64_t>(q_token_idx) * kargs.stride_q_rope_n + static_cast<int64_t>(h_block_start) * kargs.stride_q_rope_h;

    auto g_q_nope = make_gmem(reinterpret_cast<const D_NOPE*>(kargs.q_nope_ptr) + q_nope_gmem_offset, (kargs.H - h_block_start) * kargs.stride_q_nope_h * sizeof(D_NOPE));
    auto g_q_rope = make_gmem(reinterpret_cast<const D_ROPE*>(kargs.q_rope_ptr) + q_rope_gmem_offset, (kargs.H - h_block_start) * kargs.stride_q_rope_h * sizeof(D_ROPE));

    auto u_q_nope  = make_layout_q_nope<T>(warp_id, lane_id, kargs.stride_q_nope_h);
    auto u_q_mxscl = make_layout_q_mxscl<T>(warp_id, lane_id, kargs.stride_q_nope_h);
    auto u_q_rope  = make_layout_q_rope<T>(warp_id, lane_id, kargs.stride_q_rope_h);

    auto v_q_nope  = load<T::VEC_NOPE>(g_q_nope, u_q_nope);
    auto v_q_mxscl = load<T::VEC_MXSCL>(g_q_nope, u_q_mxscl + T::D_NOPE_SIZE);
    auto v_q_rope  = load<T::VEC_ROPE>(g_q_rope, u_q_rope);
    s_wait_loadcnt(0_I);

    zero_nope_pad<T>(v_q_nope);
    zero_mxscl_pad<T>(v_q_mxscl, lane_id >= T::W_M);

    __shared__ char smem_kv_buf[T::smem_size_bytes()];

    constexpr D_ACC LOG2_E = 1.44269504089f;
    const D_ACC temperature_scale = kargs.softmax_scale * LOG2_E;

    vector_t<D_ACC, T::Q_TILE_SIZE * T::D_HEAD_SIZE / T::WARP_SIZE> v_o;
    clear(v_o);
    D_ACC m_row = opus::numeric_limits<D_ACC>::lowest();
    D_ACC l_row = 0.0f;

    // Prefix segment: indices point into unified_kv_{nope,rope}[total_pages]
    {
        const int page_idx_begin = kargs.kv_indptr_prefix[q_token_idx];
        const int valid_kv_len   = kargs.kv_indptr_prefix[q_token_idx + 1] - page_idx_begin;
        const int num_kv_tiles   = ceil_div(valid_kv_len, T::KV_TILE_SIZE);
        pa_prefill_accum_pipelined<Traits>(kargs, kargs.unified_kv_nope_ptr, kargs.unified_kv_rope_ptr, kargs.total_pages,
                                           kargs.kv_indices_prefix, page_idx_begin, valid_kv_len, num_kv_tiles,
                                           smem_kv_buf, v_q_nope, v_q_rope, v_q_mxscl, v_o, m_row, l_row, temperature_scale);
    }
    __builtin_amdgcn_s_barrier();
    // Extend segment: indices point into kv_{nope,rope}[total_tokens]
    {
        const int page_idx_begin = kargs.kv_indptr_extend[q_token_idx];
        const int valid_kv_len   = kargs.kv_indptr_extend[q_token_idx + 1] - page_idx_begin;
        const int num_kv_tiles   = ceil_div(valid_kv_len, T::KV_TILE_SIZE);
        pa_prefill_accum_pipelined<Traits>(kargs, kargs.kv_nope_ptr, kargs.kv_rope_ptr, kargs.total_tokens,
                                           kargs.kv_indices_extend, page_idx_begin, valid_kv_len, num_kv_tiles,
                                           smem_kv_buf, v_q_nope, v_q_rope, v_q_mxscl, v_o, m_row, l_row, temperature_scale);
    }

    // Fold the per-head sink into the denominator, normalize O, store it out.
    const int sink_head_idx = h_block_start + warp_id * T::Q_TILE_SIZE + (lane_id % T::W_M);
    auto g_attn_sink = make_gmem(reinterpret_cast<const D_ACC*>(kargs.attn_sink_ptr), kargs.H * sizeof(D_ACC));
    D_ACC sink_log2 = load(g_attn_sink, sink_head_idx)[0] * LOG2_E;
    D_ACC m_final = max(m_row, sink_log2);
    D_ACC alpha = __builtin_amdgcn_exp2f(m_row - m_final);
    D_ACC l_final = l_row * alpha + __builtin_amdgcn_exp2f(sink_log2 - m_final);
    D_ACC o_scale = (l_final > D_ACC(0.0f)) ? (alpha / l_final) : D_ACC(0.0f);
    scale_output_tile<T>(v_o, o_scale);

    using D_OUT = typename T::D_OUT;
    const int64_t o_gmem_offset = static_cast<int64_t>(q_token_idx) * kargs.stride_o_n + static_cast<int64_t>(h_block_start) * kargs.stride_o_h;
    auto g_o = make_gmem(reinterpret_cast<D_OUT*>(kargs.out_ptr) + o_gmem_offset, (kargs.H - h_block_start) * kargs.stride_o_h * sizeof(D_OUT));
    auto u_o = make_layout_o<T>(warp_id, lane_id, kargs.stride_o_h);
    store<T::VEC_O>(g_o, cast<D_OUT>(v_o), u_o);
}
