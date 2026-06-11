#include <opus/opus.hpp>
#include "pa_defs.h"

using opus::operator""_I;

namespace pa_16mx1_16nx4_fp8 {

// Create layout for loading Q matrix from global memory
template<class T>
__device__ inline auto make_layout_q(int lane_id, int stride_q_h) {
    constexpr auto q_block_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_M>{},
        opus::number<T::W_M>{},
        opus::number<T::D_TILE_SIZE / T::W_K>{},
        opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_Q_NOPE>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_Q_NOPE>{});

    constexpr auto q_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        q_block_shape,
        opus::unfold_x_stride(q_block_dim, q_block_shape, opus::tuple{stride_q_h, 1_I}),
        opus::unfold_p_coord(q_block_dim, opus::tuple{lane_id % T::W_M, lane_id / T::W_M}));
}

template<class T>
__device__ inline auto make_layout_q_mxscl(int lane_id, int stride_q_h) {
    constexpr auto q_block_shape = opus::make_tuple(
        opus::number<T::W_M>{},
        opus::number<T::VEC_Q_NOPE>{});

    constexpr auto q_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        q_block_shape,
        opus::unfold_x_stride(q_block_dim, q_block_shape, opus::tuple{stride_q_h, 1_I}),
        opus::unfold_p_coord(q_block_dim, opus::tuple{lane_id % T::W_M}));
}

template<class Traits, class VQ, class VO>
__device__ void pa_prefill_16mx1_16nx4_fp8_pipeline(pa_kargs kargs,
                                                    const void* kv_ptr, int kv_rows,
                                                    const int* kv_indices, int page_idx_begin,
                                                    int valid_kv_len, int num_kv_tiles,
                                                    char* smem_kv, char* smem_ml, char* smem_p,
                                                    VQ& v_q, VO& v_o,
                                                    typename Traits::D_ACC& m_row,
                                                    typename Traits::D_ACC& l_row) {
    (void)kargs; (void)kv_ptr; (void)kv_rows; (void)kv_indices; (void)page_idx_begin;
    (void)valid_kv_len; (void)num_kv_tiles; (void)smem_kv; (void)smem_ml; (void)smem_p;
    (void)v_q; (void)v_o; (void)m_row; (void)l_row;
}

} // namespace pa_16mx1_16nx4_fp8

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void pa_prefill_16mx1_16nx4_fp8_kernel(pa_kargs kargs) {
    using namespace opus;
    using namespace pa_16mx1_16nx4_fp8;
    using T = opus::remove_cvref_t<Traits>;
    using D_NOPE = typename T::D_NOPE;
    using D_ROPE = typename T::D_ROPE;
    using D_ACC = typename T::D_ACC;

    const int q_token_idx = block_id_x();
    const int h_block_idx = block_id_y();

    const int lane_id = thread_id_x() % T::WARP_SIZE;

    const int h_block_start = h_block_idx * T::T_M * T::Q_TILE_SIZE;
    const int qo_gmem_offset = q_token_idx * kargs.stride_qo_n + h_block_start * kargs.stride_qo_h;

    auto g_q = make_gmem(reinterpret_cast<const D_NOPE*>(kargs.q_ptr) + qo_gmem_offset, (kargs.H - h_block_start) * kargs.stride_qo_h * sizeof(D_NOPE));
    auto u_q = make_layout_q<T>(lane_id, kargs.stride_qo_h);

    auto v_q = load<T::VEC_Q_NOPE>(g_q, u_q);

    // NoPE tile (fp8)
    constexpr index_t q_len       = vector_traits<decltype(v_q)>::size();
    constexpr index_t q_nope_len  = T::Q_TILE_SIZE * T::D_NOPE_PADDED_SIZE / T::WARP_SIZE;  // NoPE values + scales + padding
    constexpr index_t q_nope_vals = T::Q_TILE_SIZE * T::D_NOPE_SIZE        / T::WARP_SIZE;  // NoPE values only
    auto v_q_nope = slice(v_q, number<q_nope_len>{});
    static_for([&](auto i) { v_q_nope[i.value] = static_cast<D_NOPE>(0); }, number<q_nope_vals>{}, number<q_nope_len>{});

    // RoPE tile (bf16)
    constexpr index_t q_rope_all = q_len * sizeof(D_NOPE) / sizeof(D_ROPE);
    constexpr index_t q_rope_len = T::Q_TILE_SIZE * T::D_ROPE_SIZE / T::WARP_SIZE;
    auto v_q_rope = slice(reinterpret_cast<vector_t<D_ROPE, q_rope_all>&>(v_q), number<q_rope_all - q_rope_len>{}, number<q_rope_all>{});

    // NoPE mx scales (fp8 E8M0, one per 32-elem K block)
    auto u_q_mxscl = make_layout_q_mxscl<T>(lane_id, kargs.stride_qo_h);
    auto v_q_mxscl = load<T::VEC_Q_NOPE>(g_q, u_q_mxscl + T::D_NOPE_SIZE);
    constexpr index_t q_mxscl_len  = vector_traits<decltype(v_q_mxscl)>::size();  // 16 (padded scale count)
    constexpr index_t q_mxscl_vals = T::D_NOPE_SIZE / 32;                         // 14 real scales
    static_for([&](auto i) { v_q_mxscl[i.value] = static_cast<D_NOPE>(0); }, number<q_mxscl_vals>{}, number<q_mxscl_len>{});
}
