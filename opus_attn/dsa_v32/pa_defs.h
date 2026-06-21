// Shared types and constants between device kernel and host code
#pragma once

using bf16_t = __bf16;
using fp16_t = __fp16;
// 8-bit float storage types, aliased to match opus's dtype registration.
using fp8_t  = _BitInt(8);
using bf8_t  = unsigned _BitInt(8);

// Kernel arguments for the FP8 PA prefill attention.
struct pa_fp8_kargs {
    const void* __restrict__ q_nope_ptr;          // [N, H, D_NOPE_PADDED] fp8
    const void* __restrict__ q_rope_ptr;          // [N, H, D_ROPE]        bf16
    const void* __restrict__ unified_kv_nope_ptr; // [total_pages, D_NOPE_PADDED] fp8
    const void* __restrict__ unified_kv_rope_ptr; // [total_pages, D_ROPE]        bf16
    const void* __restrict__ kv_nope_ptr;         // [total_tokens, D_NOPE_PADDED] fp8
    const void* __restrict__ kv_rope_ptr;         // [total_tokens, D_ROPE]        bf16
    const void* __restrict__ attn_sink_ptr;       // [H]
    void* __restrict__ out_ptr;                   // [N, H, D_HEAD] bf16
    const int* __restrict__ kv_indptr_prefix;
    const int* __restrict__ kv_indices_prefix;
    const int* __restrict__ kv_indptr_extend;
    const int* __restrict__ kv_indices_extend;
    int N;
    int H;
    int total_pages;
    int total_tokens;
    int stride_q_nope_n;
    int stride_q_nope_h;
    int stride_q_rope_n;
    int stride_q_rope_h;
    int stride_o_n;
    int stride_o_h;
    int stride_kv_nope_page;
    int stride_kv_rope_page;
    float softmax_scale;
};

// Configuration traits for the FP8 16mx8_32nx1 PA kernel variant.
template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 32,
         int D_TILE_SIZE_ = 640,
         int NUM_WARPS_ = 8,
         typename D_NOPE_ = fp8_t,
         typename D_ROPE_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct pa_16mx8_32nx1_fp8_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE = D_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 64; // AMD wavefront size
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    // Packed DSA hdim split
    static constexpr int D_NOPE_SIZE = 448;        // NoPE fp8 elements
    static constexpr int D_NOPE_PADDED_SIZE = 512; // NoPE padded to multiple of 128
    static constexpr int D_ROPE_SIZE = 64;         // RoPE bf16 elements
    static constexpr int D_HEAD_SIZE = D_NOPE_SIZE + D_ROPE_SIZE; // Total head dimension size (512)

    // Data types: NoPE fp8 + RoPE bf16; accumulation fp32.
    using D_NOPE = D_NOPE_;
    using D_ROPE = D_ROPE_;
    using D_ATTN = D_NOPE_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    // MFMA wave layout (identical to the bf16 16mx8_32nx1 variant)
    static constexpr int T_M = NUM_WARPS; // waves along M
    static constexpr int T_N = 1;         // waves along N
    static constexpr int T_K = 1;         // waves along K

    // MFMA base tile: NoPE uses fp8 16x16x128 (scaled f8f6f4 on gfx950);
    // RoPE (bf16 QK^T) and PV (bf16) use 16x16x32.
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K_NOPE = 128;
    static constexpr int W_K_ROPE = 32;

    // D slicing: D=512 iterates D in SLICE_D=32 chunks
    static constexpr int SLICE_D = 32;
    static constexpr int NUM_D_SLICES = D_HEAD_SIZE / SLICE_D;

    // GEMM0: S = Q @ K^T
    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / W_N;
    static constexpr int GEMM0_NOPE_E_K = D_NOPE_PADDED_SIZE / W_K_NOPE;
    static constexpr int GEMM0_ROPE_E_K = D_ROPE_SIZE / W_K_ROPE;

    // GEMM1: O = P @ V
    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = SLICE_D / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K_ROPE;

    // Vector lengths for global load/store
    static constexpr int VEC_Q_NOPE  = 16;
    static constexpr int VEC_Q_ROPE  = 8;
    static constexpr int VEC_KV_NOPE = 16;
    static constexpr int VEC_KV_ROPE = 8;
    static constexpr int VEC_P    = 4;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O    = 4;

    static constexpr int D_128B_NOPE_SIZE = 128 / sizeof(D_NOPE);
    static constexpr int dwordx4_size = 16;
    static constexpr int smem_linear_wave_nope = WARP_SIZE * dwordx4_size / sizeof(D_NOPE);
    static constexpr int smem_n_per_wave = 8; // WARP_SIZE * 16 / 128
    static constexpr int smem_n_rpt = KV_TILE_SIZE / smem_n_per_wave; // 4
    static constexpr int smem_d_rpt_nope = D_NOPE_PADDED_SIZE / D_128B_NOPE_SIZE; // 4
    static constexpr int smem_padding_32B_nope = 32 / sizeof(D_NOPE);

    static constexpr int D_128B_ROPE_SIZE = 128 / sizeof(D_ROPE);
    static constexpr int smem_linear_wave_rope = WARP_SIZE * dwordx4_size / sizeof(D_ROPE);
    static constexpr int smem_d_rpt_all = D_HEAD_SIZE / D_128B_ROPE_SIZE; // 8
    static constexpr int smem_d_rpt_rope = D_ROPE_SIZE / D_128B_ROPE_SIZE; // 1
    static constexpr int smem_padding_32B_rope = 32 / sizeof(D_ROPE);

    static constexpr int kv_buffer_load_insts = (KV_TILE_SIZE * D_NOPE_PADDED_SIZE) / (BLOCK_SIZE * VEC_KV_NOPE) + 2 * (KV_TILE_SIZE * D_ROPE_SIZE) / (BLOCK_SIZE * VEC_KV_ROPE);
    static constexpr int k_nope_ds_read_insts = (GEMM0_E_N * W_N * W_K_NOPE) / (WARP_SIZE * VEC_KV_NOPE);
    static constexpr int k_rope_ds_read_insts = (GEMM0_E_N * W_N * W_K_ROPE) / (WARP_SIZE * VEC_KV_ROPE);
    static constexpr int v_ds_read_insts = (GEMM1_E_N * GEMM1_E_K * W_N * W_K_ROPE) / (WARP_SIZE * VEC_TR_V);
};

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
