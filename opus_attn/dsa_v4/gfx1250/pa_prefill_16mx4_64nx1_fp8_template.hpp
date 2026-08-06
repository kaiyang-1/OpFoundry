// Paged sparse attention MXFP8 kernel template for D=512 on gfx1250 (wave32 / WMMA).
// 16mx4_64nx1_fp8 variant (T_M=4, T_N=1); include from a .cc that instantiates the traits.
//
// SKELETON: only the plumbing is in place. Everything marked TODO below is unimplemented, so
// the kernel currently leaves the output untouched. The layout helpers are declared but not
// defined on purpose — defining one is what implementing this kernel means.
//
// Intended per-KV-tile data flow, mirroring the bf16 16mx4_64nx1 variant plus the gfx950 MXFP8 path:
//   1. TDM gather of the NoPE (fp8, with inline E8M0 block scales) and RoPE (bf16) rows into LDS.
//   2. S = Q @ K^T, NoPE part on scaled f8f6f4 WMMA 16x16x128, RoPE part on bf16 WMMA 16x16x32.
//   3. Online softmax, sliced into the GEMM stages exactly as the bf16 variant does.
//   4. V dequant fp8 -> bf16 into an LDS staging buffer, then O += P @ V on bf16 WMMA 16x16x32.
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
__device__ inline auto make_layout_q_nope(int warp_id, int lane_id) {
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
        opus::unfold_x_stride(q_block_dim, q_block_shape, opus::tuple{opus::number<T::D_NOPE_PADDED_SIZE>{}, 1_I}),
        opus::unfold_p_coord(q_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

template<class T>
__device__ inline auto make_layout_q_rope(int warp_id, int lane_id) {
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
        opus::unfold_x_stride(q_block_dim, q_block_shape, opus::tuple{opus::number<T::D_ROPE_SIZE>{}, 1_I}),
        opus::unfold_p_coord(q_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

template<class T>
__device__ inline auto make_layout_q_mxscl(int warp_id, int lane_id) {
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
        opus::unfold_x_stride(q_block_dim, q_block_shape, opus::tuple{opus::number<T::D_NOPE_PADDED_SIZE>{}, 1_I}),
        opus::unfold_p_coord(q_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

template<class Traits>
__device__ __attribute__((always_inline)) void pa_prefill_accum_pipelined(pa_fp8_kargs kargs,
                                           const void* kv_nope_ptr, const void* kv_rope_ptr, int kv_rows,
                                           const int* kv_indices, int page_idx_begin, int valid_kv_len, int num_kv_tiles,
                                           char* smem_kv_buf,
                                           opus::vector_t<typename Traits::D_NOPE, Traits::Q_TILE_SIZE * Traits::D_NOPE_PADDED_SIZE / Traits::WARP_SIZE>& v_q_nope,
                                           opus::vector_t<typename Traits::D_ROPE, Traits::Q_TILE_SIZE * Traits::D_ROPE_SIZE / Traits::WARP_SIZE>& v_q_rope,
                                           int scale_q,
                                           opus::vector_t<typename Traits::D_ACC, Traits::Q_TILE_SIZE * Traits::D_HEAD_SIZE / Traits::WARP_SIZE>& v_o,
                                           typename Traits::D_ACC& m_row,
                                           typename Traits::D_ACC& l_row,
                                           typename Traits::D_ACC temperature_scale) {
    if (num_kv_tiles <= 0) return;

    
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
    const int64_t q_nope_gmem_offset = (int64_t)q_token_idx * kargs.stride_q_nope_n + (int64_t)h_block_start * kargs.stride_q_nope_h;
    const int64_t q_rope_gmem_offset = (int64_t)q_token_idx * kargs.stride_q_rope_n + (int64_t)h_block_start * kargs.stride_q_rope_h;

    auto g_q_nope = make_gmem(reinterpret_cast<const D_NOPE*>(kargs.q_nope_ptr) + q_nope_gmem_offset, (kargs.H - h_block_start) * kargs.stride_q_nope_h * sizeof(D_NOPE));
    auto g_q_rope = make_gmem(reinterpret_cast<const D_ROPE*>(kargs.q_rope_ptr) + q_rope_gmem_offset, (kargs.H - h_block_start) * kargs.stride_q_rope_h * sizeof(D_ROPE));

    auto u_q_nope  = make_layout_q_nope<T>(warp_id, lane_id);
    auto u_q_mxscl = make_layout_q_mxscl<T>(warp_id, lane_id);
    auto u_q_rope  = make_layout_q_rope<T>(warp_id, lane_id);

    auto v_q_nope  = load<T::VEC_NOPE>(g_q_nope, u_q_nope);
    auto v_q_mxscl = load<T::VEC_MXSCL>(g_q_nope, u_q_mxscl + T::D_NOPE_SIZE);
    auto v_q_rope  = load<T::VEC_ROPE>(g_q_rope, u_q_rope);
    s_wait_loadcnt(0_I);

    constexpr index_t q_nope_len  = vector_traits<decltype(v_q_nope)>::size();
    constexpr index_t q_nope_vals = T::Q_TILE_SIZE * T::D_NOPE_SIZE / T::WARP_SIZE;
    static_for([&](auto i) { v_q_nope[i.value] = static_cast<D_NOPE>(0); }, number<q_nope_vals>{}, number<q_nope_len>{});

    constexpr index_t q_mxscl_len = vector_traits<decltype(v_q_mxscl)>::size();
    const bool upper_half = lane_id >= T::W_M;
    v_q_mxscl[q_mxscl_len - 2] = upper_half ? static_cast<D_NOPE>(0) : v_q_mxscl[q_mxscl_len - 2];
    v_q_mxscl[q_mxscl_len - 1] = upper_half ? static_cast<D_NOPE>(0) : v_q_mxscl[q_mxscl_len - 1];
}
