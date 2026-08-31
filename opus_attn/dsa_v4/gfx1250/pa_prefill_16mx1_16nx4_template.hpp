#pragma once

#include <opus/opus.hpp>
#include "pa_traits.h"
#include <cstdint>
#include <bit>

using opus::operator""_I;

namespace pa_16mx1_16nx4 {

template<class T>
__device__ inline auto make_layout_q(int lane_id) {
    constexpr auto q_block_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_M>{},
        opus::number<T::W_M>{},
        opus::number<T::D_TILE_SIZE / T::W_K>{},
        opus::number<T::W_M * T::W_K / (T::WARP_SIZE * T::VEC_Q)>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_Q>{});

    constexpr auto q_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        q_block_shape,
        opus::unfold_x_stride(q_block_dim, q_block_shape, opus::tuple{opus::number<T::Q_ROW_LDS_ELEMS>{}, 1_I}),
        opus::unfold_p_coord(q_block_dim, opus::tuple{lane_id % T::W_M, lane_id / T::W_M}));
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

template<typename T, typename V>
__device__ inline void scale_output_tile(V& v_o, typename T::D_ACC scale) {
    constexpr opus::index_t o_len = opus::vector_traits<V>::size();
    opus::static_for<o_len>([&](auto i) { v_o[i.value] *= scale; });
}

template<class Traits>
__device__ __attribute__((always_inline)) void pa_prefill_accum_pipelined(
        pa_kargs kargs,
        const void* kv_ptr, int kv_rows,
        const int* kv_indices, int page_idx_begin, int valid_kv_len, int num_kv_tiles,
        char* smem_kv_buf,
        opus::vector_t<typename Traits::D_ATTN, Traits::Q_TILE_SIZE * Traits::D_TILE_SIZE / Traits::WARP_SIZE>& v_q,
        opus::vector_t<typename Traits::D_ACC, Traits::Q_TILE_SIZE * Traits::D_TILE_SIZE / (Traits::T_N * Traits::WARP_SIZE)>& v_o,
        typename Traits::D_ACC& m_row,
        typename Traits::D_ACC& l_row,
        typename Traits::D_ACC temperature_scale) {
    
}

} // namespace pa_16mx1_16nx4

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void pa_prefill_16mx1_16nx4_kernel(pa_kargs kargs) {
    using namespace opus;
    using namespace pa_16mx1_16nx4;
    using T = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC = typename T::D_ACC;
    using D_OUT = typename T::D_OUT;

    const int q_token_idx = block_id_x();
    const int h_block_idx = block_id_y();
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    const int h_block_start = h_block_idx * T::Q_TILE_SIZE;
    const int64_t qo_token_offset = static_cast<int64_t>(q_token_idx) * kargs.stride_qo_n;
    const u32_t   qo_head_origin  = (u32_t)(h_block_start + warp_id * T::QO_ROWS_PER_WAVE);

    __shared__ char smem_buf[T::smem_size_bytes()];
    char* const qo_tile = smem_buf;

    vector_t<D_ATTN, T::Q_TILE_SIZE * T::D_TILE_SIZE / T::WARP_SIZE> v_q;
    vector_t<D_ACC,  T::Q_TILE_SIZE * T::D_TILE_SIZE / (T::T_N * T::WARP_SIZE)> v_o;

    constexpr D_ACC LOG2_E = 1.44269504089f;
    const D_ACC temperature_scale = kargs.softmax_scale * LOG2_E;

    {
        using q_window = tdm<D_ATTN, seq<T::D_TILE_SIZE, T::QO_ROWS_PER_WAVE>,
                             tdm_traits::padding_auto<D_ATTN, T::D_TILE_SIZE>>;

        auto tdm_q = make_tdm<q_window>(
            (u32_t)reinterpret_cast<uintptr_t>(qo_tile),
            reinterpret_cast<const D_ATTN*>(kargs.q_ptr) + qo_token_offset,
            /*shape0=*/ (u32_t)T::D_TILE_SIZE,
            /*shape1=*/ (u32_t)kargs.H,
            /*stride=*/ (u64_t)kargs.stride_qo_h,
            /*origin0=*/ 0u,
            /*origin1=*/ qo_head_origin);
        tdm_q.async_load((u32_t)(warp_id * T::QO_ROWS_PER_WAVE * T::Q_ROW_LDS_ELEMS));

        auto s_q = make_smem(reinterpret_cast<D_ATTN*>(qo_tile));
        s_wait_tensorcnt(0_I);
        __builtin_amdgcn_s_barrier();
        v_q = load<T::VEC_Q>(s_q, make_layout_q<T>(lane_id));
        s_wait_dscnt(0_I);
        __builtin_amdgcn_s_barrier();
    }

    clear(v_o);
    D_ACC m_row = opus::numeric_limits<D_ACC>::lowest();
    D_ACC l_row = D_ACC(0.0f);

    // Prefix segment: indices point into unified_kv[total_pages]
    {
        const int page_idx_begin = kargs.kv_indptr_prefix[q_token_idx];
        const int valid_kv_len   = kargs.kv_indptr_prefix[q_token_idx + 1] - page_idx_begin;
        const int num_kv_tiles   = ceil_div(valid_kv_len, T::KV_TILE_SIZE);
        pa_prefill_accum_pipelined<Traits>(kargs, kargs.unified_kv_ptr, kargs.total_pages,
                                           kargs.kv_indices_prefix, page_idx_begin, valid_kv_len, num_kv_tiles,
                                           smem_buf, v_q, v_o, m_row, l_row, temperature_scale);
    }

    __builtin_amdgcn_s_barrier();

    // Extend segment: indices point into kv[total_tokens]
    {
        const int page_idx_begin = kargs.kv_indptr_extend[q_token_idx];
        const int valid_kv_len   = kargs.kv_indptr_extend[q_token_idx + 1] - page_idx_begin;
        const int num_kv_tiles   = ceil_div(valid_kv_len, T::KV_TILE_SIZE);
        pa_prefill_accum_pipelined<Traits>(kargs, kargs.kv_ptr, kargs.total_tokens,
                                           kargs.kv_indices_extend, page_idx_begin, valid_kv_len, num_kv_tiles,
                                           smem_buf, v_q, v_o, m_row, l_row, temperature_scale);
    }

    // Sink finalization, normalize O.
    const int sink_head_idx = h_block_start + (lane_id % T::W_M);
    auto g_attn_sink = make_gmem(reinterpret_cast<const D_ACC*>(kargs.attn_sink_ptr), kargs.H * sizeof(D_ACC));
    D_ACC sink_log2 = load(g_attn_sink, sink_head_idx)[0] * LOG2_E;
    D_ACC m_final = max(m_row, sink_log2);
    D_ACC alpha = __builtin_amdgcn_exp2f(m_row - m_final);
    D_ACC l_final = l_row * alpha + __builtin_amdgcn_exp2f(sink_log2 - m_final);
    D_ACC o_scale = (l_final > D_ACC(0.0f)) ? (alpha / l_final) : D_ACC(0.0f);
    scale_output_tile<T>(v_o, o_scale);

    {
        using o_window = tdm<D_OUT, seq<T::O_ROW_LDS_ELEMS, T::QO_ROWS_PER_WAVE>>;

        s_wait_tensorcnt(0_I);
        s_wait_dscnt(0_I);
        __builtin_amdgcn_s_barrier();

        auto s_o = make_smem(reinterpret_cast<D_OUT*>(qo_tile));
        store<T::VEC_O>(s_o, cast<D_OUT>(v_o), make_layout_o<T>(warp_id, lane_id));

        auto tdm_o = make_tdm<o_window>(
            (u32_t)reinterpret_cast<uintptr_t>(qo_tile),
            reinterpret_cast<D_OUT*>(kargs.out_ptr) + qo_token_offset,
            /*shape0=*/ (u32_t)T::D_TILE_SIZE,
            /*shape1=*/ (u32_t)kargs.H,
            /*stride=*/ (u64_t)kargs.stride_qo_h,
            /*origin0=*/ 0u,
            /*origin1=*/ qo_head_origin);

        s_wait_dscnt(0_I);
        __builtin_amdgcn_s_barrier();
        tdm_o.async_store((u32_t)(warp_id * T::QO_ROWS_PER_WAVE * T::O_ROW_LDS_ELEMS));
        s_wait_tensorcnt(0_I);
    }
}
