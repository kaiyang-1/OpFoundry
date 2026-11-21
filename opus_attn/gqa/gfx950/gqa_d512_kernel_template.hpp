// GQA forward D=512 kernel (gfx950), varlen. 8 waves of 16 Q rows, KV tiles of 32 rows, D in slices of 32, one phase
// per KV tile; waves 4-7 run a cluster behind waves 0-3.
#pragma once

#include <opus/opus.hpp>
#include "gqa_traits.h"
#include <bit>
#include <cstdint>

namespace gqa_d512 {

using opus::operator""_I;

constexpr float LOG2E = 1.44269504088896340736f, LN2 = 0.69314718055994530942f;
// Fast math assumes no infinities: -inf only as bits. A finite M_INIT makes exp2 of a masked logit 0, not NaN.
constexpr float M_INIT = -1.0e30f;
constexpr float NO_SINK = -3.0e38f;  // exp2(NO_SINK - m) = 0
constexpr unsigned NEG_INF_BITS = 0xff800000u;

// s_load through the constant address space
template <class T>
__device__ inline T sload(const void* p, int i) {
    return reinterpret_cast<const __attribute__((address_space(4))) T*>(reinterpret_cast<uintptr_t>(p))[i];
}

// Buffer size: reads past it return 0, writes are dropped
__device__ inline unsigned int buffer_bytes(int64_t bytes) {
    return bytes >= int64_t(0xffffffffu) ? 0xffffffffu : unsigned(bytes);
}

// The workgroup's batch, head and Q tile. Q/O [total_q, H, D], K/V [total_k, H_KV, D], LSE [total_q, H].
struct seq_ctx {
    int sq, sk, q0;
    int q_stride, kv_stride, lse_stride;  // H D, H_KV D, H
    const bf16_t* q;                      // row q0, head h
    bf16_t* o;
    float* lse;                           // null: no LSE
    const bf16_t* k;                      // row 0, KV head
    const bf16_t* v;
    float c;                              // scale log2(e)
    float sink;                           // sink log2(e), or NO_SINK

    __device__ seq_ctx at(int row) const {
        seq_ctx s = *this;
        s.q0 = row;
        s.q += int64_t(row) * q_stride;
        s.o += int64_t(row) * q_stride;
        if (s.lse) s.lse += int64_t(row) * lse_stride;
        return s;
    }
};

// Grid (tiles, H, B), (H, tiles, B) with OPUS_GQA_OPT_REMAP_XY. f(seq_ctx) for Q tile gx, and with
// OPUS_GQA_OPT_DOUBLE_Q (half the tiles) for tile tiles - 1 - gx after it.
template <int D, int Q_ROWS, class F>
__device__ inline void for_each_tile(const opus_gqa_varlen_kargs& kargs, F&& f) {
    int gx = opus::block_id_x(), gy = opus::block_id_y();
    if (kargs.opt & OPUS_GQA_OPT_REMAP_XY) {
        const int t = gx;
        gx = gy;
        gy = t;
    }
    const int b = opus::block_id_z();
    const int q_start = sload<int>(kargs.qseq_addr, b), k_start = sload<int>(kargs.kseq_addr, b);
    seq_ctx s;
    s.sq = sload<int>(kargs.qseq_addr, b + 1) - q_start;
    s.sk = sload<int>(kargs.kseq_addr, b + 1) - k_start;
    const int tiles = (s.sq + Q_ROWS - 1) / Q_ROWS;
    const bool double_q = kargs.opt & OPUS_GQA_OPT_DOUBLE_Q;
    if (gx >= (double_q ? (tiles + 1) / 2 : tiles)) return;

    const int gqa = kargs.gqa, hq = kargs.q_head_num, hkv = hq / gqa;
    // neighbouring workgroups take different KV heads: those of an XCD share theirs
    const int h = (gy % hkv) * gqa + gy / hkv, h_kv = h / gqa;
    s.q0 = 0;
    s.q_stride = hq * D;
    s.kv_stride = hkv * D;
    s.lse_stride = hq;
    const int64_t q_row = int64_t(q_start) * hq + h;
    s.q = reinterpret_cast<const bf16_t*>(kargs.q_addr) + q_row * D;
    s.o = reinterpret_cast<bf16_t*>(kargs.d_addr) + q_row * D;
    s.lse = kargs.lse ? reinterpret_cast<float*>(kargs.lse_addr) + q_row : nullptr;
    const int64_t kv_row = int64_t(k_start) * hkv + h_kv;
    s.k = reinterpret_cast<const bf16_t*>(kargs.k_addr) + kv_row * D;
    s.v = reinterpret_cast<const bf16_t*>(kargs.v_addr) + kv_row * D;
    s.c = kargs.scalar * LOG2E;
    s.sink = kargs.sink_addr ? sload<float>(kargs.sink_addr, h) * LOG2E : NO_SINK;

    // two calls, not a loop: a loop shares the tiles' invariant VGPRs and spills
    f(s.at(gx * Q_ROWS));
    if (double_q && tiles - 1 - gx != gx) {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();  // the first tile's LDS reads are done
        __builtin_amdgcn_sched_barrier(0);
        f(s.at((tiles - 1 - gx) * Q_ROWS));
    }
}

// m and l of a row with the sink logit x
struct row_final {
    float m, l, o_scale;
    __device__ row_final(float m_acc, float l_acc, float x) {
        m = __builtin_fmaxf(m_acc, x);
        const float alpha = __builtin_amdgcn_exp2f(m_acc - m);
        l = l_acc * alpha + __builtin_amdgcn_exp2f(x - m);
        o_scale = l > 0.0f ? alpha / l : 0.0f;
    }
    // natural log, -inf if l = 0
    __device__ float lse() const {
        const float x = (m + __builtin_amdgcn_logf(l)) * LN2;
        return __builtin_bit_cast(float, l > 0.0f ? __builtin_bit_cast(unsigned, x) : NEG_INF_BITS);
    }
};

constexpr int MFMA_MASK    = 0x08;
constexpr int VALU_MASK    = 0x02;
constexpr int SALU_MASK    = 0x04;
constexpr int EXP_MASK     = 0x400;
constexpr int DS_READ_MASK = 0x100;

template<int Group>
__device__ inline void sched_compute_qk() {
    opus::static_for<4>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(EXP_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(SALU_MASK, 1, Group);
    });
    opus::static_for<10>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(VALU_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(SALU_MASK, 1, Group);
    });
    opus::static_for<4>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(VALU_MASK, 2, Group);
    });
}

// Create layout for loading Q matrix from global memory
template<class T>
__device__ inline auto make_layout_q(int warp_id, int lane_id, int stride_q_n) {
    constexpr auto q_block_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::D_TILE_SIZE / T::W_K>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_Q>{});

    constexpr auto q_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        q_block_shape,
        opus::unfold_x_stride(q_block_dim, q_block_shape, opus::tuple{stride_q_n, 1_I}),
        opus::unfold_p_coord(q_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

// Create layout for storing O matrix to global memory
template<class T>
__device__ inline auto make_layout_o(int warp_id, int lane_id, int stride_o_n) {
    constexpr auto o_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::D_TILE_SIZE / T::W_N>{},
        opus::number<T::W_M * T::W_N / T::WARP_SIZE / T::VEC_O>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_O>{});

    constexpr auto o_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        o_block_shape,
        opus::unfold_x_stride(o_block_dim, o_block_shape, opus::tuple{stride_o_n, 1_I}),
        opus::unfold_p_coord(o_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

// Create layout for loading K matrix from global memory
template<typename T>
__device__ inline auto make_layout_gk_gv(int warp_id, int lane_id, int stride_kv_n) {
    constexpr int threads_d = T::D_128B_SIZE / T::VEC_KV;
    constexpr int warps_d = T::NUM_WARPS / T::smem_n_rpt;

    constexpr auto gk_block_shape = opus::make_tuple(
        opus::number<T::smem_d_rpt / warps_d>{},
        opus::number<warps_d>{},
        opus::number<T::smem_n_per_wave>{},
        opus::number<T::smem_n_rpt>{},
        opus::number<threads_d>{},
        opus::number<T::VEC_KV>{});

    constexpr auto gk_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        gk_block_shape,
        opus::unfold_x_stride(gk_block_dim, gk_block_shape, opus::tuple{opus::number<T::D_128B_SIZE>{}, stride_kv_n, 1_I}),
        opus::unfold_p_coord(gk_block_dim, opus::tuple{warp_id / T::smem_n_rpt, lane_id / threads_d, warp_id % T::smem_n_rpt, lane_id % threads_d}));
}

// Create layout for storing K matrix to shared memory
template<typename T>
__device__ inline auto make_layout_sk_sv(int warp_id) {
    constexpr auto sk_block_shape = opus::make_tuple(
        opus::number<T::smem_d_rpt * T::smem_n_rpt / T::NUM_WARPS>{},
        opus::number<T::NUM_WARPS>{},
        opus::number<T::VEC_KV>{});

    constexpr auto sk_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sk_block_shape,
        opus::unfold_x_stride(sk_block_dim, sk_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding_32B>{}, 1_I}),
        opus::unfold_p_coord(sk_block_dim, opus::tuple{warp_id}));
}

// Create layout for reading K matrix from shared memory to registers
template<typename T>
__device__ inline auto make_layout_rk(int lane_id) {
    constexpr auto rk_block_shape = opus::make_tuple(
        opus::number<T::smem_n_rpt>{},
        opus::number<T::GEMM0_E_N>{},
        opus::number<T::W_N / T::smem_n_rpt>{},
        opus::number<T::GEMM0_E_K>{},
        opus::number<opus::get_warp_size() / T::W_N>{},
        opus::number<T::VEC_KV>{});

    constexpr auto rk_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto lane_id_n = lane_id % T::W_N;

    return opus::make_layout(
        rk_block_shape,
        opus::unfold_x_stride(rk_block_dim, rk_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding_32B>{},
                                                                        opus::number<T::D_128B_SIZE>{},
                                                                        1_I}),
        opus::unfold_p_coord(rk_block_dim, opus::tuple{lane_id_n % T::smem_n_rpt, lane_id_n / T::smem_n_rpt, lane_id / T::W_N}));
}

template<class T>
__device__ inline auto make_layout_rv(int lane_id) {
    constexpr int lane_per_grp = 16;
    constexpr int lane_lo = 4;
    constexpr int lane_hi = lane_per_grp / lane_lo;

    constexpr int num_grps = T::WARP_SIZE / lane_per_grp;
    constexpr int grp_n = T::W_N / (lane_lo * T::VEC_TR_V);
    constexpr int grp_k = num_grps / grp_n;

    constexpr auto rv_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_N>{},
        opus::number<T::GEMM1_E_K>{},
        opus::number<lane_hi>{},
        opus::number<T::W_K / (lane_hi * grp_k)>{},
        opus::number<grp_k>{},
        opus::number<grp_n>{},
        opus::number<lane_lo>{},
        opus::number<T::VEC_TR_V>{});

    constexpr auto rv_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    int grp_id = lane_id / lane_per_grp;
    int lane_in_grp = lane_id % lane_per_grp;

    return opus::make_layout(
        rv_block_shape,
        opus::unfold_x_stride(rv_block_dim, rv_block_shape, opus::tuple{opus::number<grp_n * lane_lo * T::VEC_TR_V>{},
                                                                        opus::number<T::smem_linear_wave + T::smem_padding_32B>{},
                                                                        opus::number<T::D_128B_SIZE>{},
                                                                        1_I}),
        opus::unfold_p_coord(rv_block_dim, opus::tuple{lane_in_grp / lane_lo, grp_id / grp_n, grp_id % grp_n, lane_in_grp % lane_lo}));
}

template<typename T, typename V>
__device__ inline typename T::D_ACC attn_row_max(const V& v_s) {
    using D_ACC = typename T::D_ACC;
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    D_ACC row_max = gqa_d512::M_INIT;
    opus::static_for<s_len>([&](auto i) {
        row_max = max(row_max, v_s[i.value]);
    });
    // lanes i, i + 32
    opus::vector_t<opus::u32_t, 2> res32 = __builtin_amdgcn_permlane32_swap(std::bit_cast<opus::u32_t>(row_max), std::bit_cast<opus::u32_t>(row_max), false, true);
    row_max = max(std::bit_cast<float>(res32.x), std::bit_cast<float>(res32.y));
    // lanes i, i + 16
    opus::vector_t<opus::u32_t, 2> res16 = __builtin_amdgcn_permlane16_swap(std::bit_cast<opus::u32_t>(row_max), std::bit_cast<opus::u32_t>(row_max), false, true);
    return max(std::bit_cast<float>(res16.x), std::bit_cast<float>(res16.y));
}

template<typename T, typename V>
__device__ inline void attn_sub_row(V& v_s, typename T::D_ACC row_max) {
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    opus::static_for<s_len>([&](auto i) {
        v_s[i.value] -= row_max;
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
    D_ACC row_sum = 0.0f;
    opus::static_for<s_len>([&](auto i) {
        row_sum += v_s[i.value];
    });
    // lanes i, i + 32
    opus::vector_t<opus::u32_t, 2> res32 = __builtin_amdgcn_permlane32_swap(std::bit_cast<opus::u32_t>(row_sum), std::bit_cast<opus::u32_t>(row_sum), false, true);
    row_sum = std::bit_cast<float>(res32.x) + std::bit_cast<float>(res32.y);
    // lanes i, i + 16
    opus::vector_t<opus::u32_t, 2> res16 = __builtin_amdgcn_permlane16_swap(std::bit_cast<opus::u32_t>(row_sum), std::bit_cast<opus::u32_t>(row_sum), false, true);
    return std::bit_cast<float>(res16.x) + std::bit_cast<float>(res16.y);
}

template<typename T, typename V>
__device__ inline void scale_output_tile(V& v_o, typename T::D_ACC scale) {
    constexpr opus::index_t o_len = opus::vector_traits<V>::size();
    opus::static_for<o_len>([&](auto i) { v_o[i.value] *= scale;});
}

template<int THR_X, int THR_Y>
__device__ inline void attn_mask_vec2_imm(opus::u32_t rel_vgpr, opus::u32_t neg_inf_vgpr,
                                          opus::u32_t& x_ref, opus::u32_t& y_ref) {
    uint64_t x_mask, y_mask;
    asm volatile(
        // x: rel < THR_X ?
        "v_cmp_lt_i32_e64 %0, %6, %7\n\t"
        // y: rel < THR_Y ?
        "v_cmp_lt_i32_e64 %1, %6, %9\n\t"
        "v_cndmask_b32_e64 %2, %4, %8, %0\n\t"
        "v_cndmask_b32_e64 %3, %5, %8, %1\n\t"
        : "=s"(x_mask), "=s"(y_mask), "=v"(x_ref), "=v"(y_ref)
        : "v"(x_ref), "v"(y_ref), "v"(rel_vgpr),
          "n"(THR_X), "v"(neg_inf_vgpr), "n"(THR_Y)
        : "vcc"
    );
}

// -inf at the columns of KV tile kv_tile_idx past last_key: the causal diagonal or the end of the sequence
template<typename T, typename V>
__device__ inline void attn_mask_tile(V& v_s, int last_key, int kv_tile_idx, opus::u32_t neg_inf_v, int lane_id) {
    using D_ACC = typename T::D_ACC;
    using D_ACC_X2 = opus::vector_t<D_ACC, 2>;
    using U32_X2 = opus::vector_t<opus::u32_t, 2>;

    constexpr int elems_per_wave_tile = (T::W_M * T::W_N) / T::WARP_SIZE;
    constexpr int c_pack = 4;
    constexpr int c_rept = elems_per_wave_tile / c_pack;
    constexpr int c_rept_stride = (T::WARP_SIZE / T::W_M) * c_pack;

    const int k_start_pos = kv_tile_idx * T::KV_TILE_SIZE;
    const int lane_group = lane_id / T::W_M;

    opus::static_for<T::GEMM0_E_N>([&](auto i_n) {
        constexpr int base_idx = i_n.value * elems_per_wave_tile;
        const int k_pos = k_start_pos + i_n.value * T::W_N + lane_group * c_pack;
        const opus::u32_t rel = static_cast<opus::u32_t>(last_key - k_pos);

        opus::static_for<c_rept>([&](auto i_rept) {
            constexpr int rept_base_idx = base_idx + i_rept.value * c_pack;
            constexpr int thr_base = i_rept.value * c_rept_stride;
            opus::static_for<c_pack / 2>([&](auto i_pair) {
                constexpr int idx = rept_base_idx + i_pair.value * 2;
                constexpr int thr_x = thr_base + i_pair.value * 2;
                constexpr int thr_y = thr_x + 1;

                auto pair_acc = opus::slice(v_s, opus::number<idx>{}, opus::number<idx + 2>{});
                auto pair_bits = __builtin_bit_cast(U32_X2, pair_acc);
                opus::u32_t x_ref = pair_bits[0];
                opus::u32_t y_ref = pair_bits[1];
                attn_mask_vec2_imm<thr_x, thr_y>(rel, neg_inf_v, x_ref, y_ref);
                pair_bits[0] = x_ref;
                pair_bits[1] = y_ref;
                opus::set_slice(v_s, __builtin_bit_cast(D_ACC_X2, pair_bits), opus::number<idx>{}, opus::number<idx + 2>{});
            });
        });
    });
}

} // namespace gqa_d512

// One Q tile
template<class Traits>
__device__ __attribute__((always_inline)) void gqa_d512_tile(const gqa_d512::seq_ctx& sc, char* smem_buf) {
    using namespace opus;
    using namespace gqa_d512;
    using T = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC = typename T::D_ACC;

    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int stagger = warp_id / 4;

    const int q_block_size = T::NUM_WARPS * T::Q_TILE_SIZE;
    const int seqlen_q = sc.sq, seqlen_kv = sc.sk;
    const int q_block_start = sc.q0;
    const int q_rows = seqlen_q - q_block_start;  // rows of the tile in the sequence

    // Buffers of the sequence's rows: rows past the end read 0, O rows there are not written
    auto g_q = make_gmem(sc.q, gqa_d512::buffer_bytes(int64_t(q_rows) * sc.q_stride * sizeof(D_ATTN)));
    auto g_k = make_gmem(sc.k, gqa_d512::buffer_bytes(int64_t(seqlen_kv) * sc.kv_stride * sizeof(D_ATTN)));
    auto g_v = make_gmem(sc.v, gqa_d512::buffer_bytes(int64_t(seqlen_kv) * sc.kv_stride * sizeof(D_ATTN)));
    auto g_o = make_gmem(sc.o, gqa_d512::buffer_bytes(int64_t(q_rows) * sc.q_stride * sizeof(D_ATTN)));

    // K/V double buffer. Selected by address: a run-time indexed array of smem objects is placed in LDS.
    D_ATTN* const k_buf0 = reinterpret_cast<D_ATTN*>(smem_buf);
    D_ATTN* const k_buf1 = k_buf0 + T::smem_buffer_elems;
    auto s_k = [&](int i) { return make_smem(i ? k_buf1 : k_buf0); };
    auto s_v = [&](int i) { return make_smem((i ? k_buf1 : k_buf0) + T::smem_k_tile_elems); };

    // GEMM0: S = Q @ K^T
    auto mma0 = make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(
        seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    // GEMM1: O = P @ V
    auto mma1 = make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(
        seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    // Partition layouts
    auto u_q  = make_layout_q<T>(warp_id, lane_id, sc.q_stride);
    auto u_gkv = make_layout_gk_gv<T>(warp_id, lane_id, sc.kv_stride);
    auto u_skv = make_layout_sk_sv<T>(warp_id);
    auto u_rk = make_layout_rk<T>(lane_id);
    auto u_rv = make_layout_rv<T>(lane_id);

    // Register fragments
    typename decltype(mma0)::vtype_b v_k[2];
    typename decltype(mma0)::vtype_c v_s[2];
    typename decltype(mma1)::vtype_a v_p;
    typename decltype(mma1)::vtype_b v_v[2];
    vector_t<D_ATTN, T::Q_TILE_SIZE * T::D_TILE_SIZE / T::WARP_SIZE> v_q;
    vector_t<D_ACC,  T::Q_TILE_SIZE * T::D_TILE_SIZE / T::WARP_SIZE> v_o;
    auto v_q_slices = reinterpret_cast<vector_t<D_ATTN, T::Q_TILE_SIZE * T::SLICE_D / T::WARP_SIZE>*>(&v_q);
    auto v_o_slices = reinterpret_cast<vector_t<D_ACC,  T::Q_TILE_SIZE * T::SLICE_D / T::WARP_SIZE>*>(&v_o);

    clear(v_o);

    // Vector sizes
    constexpr index_t q_len = vector_traits<decltype(v_q)>::size();
    constexpr index_t s_len = vector_traits<typename decltype(mma0)::vtype_c>::size();
    constexpr index_t s_half_len = s_len / 2;
    constexpr index_t o_len = vector_traits<decltype(v_o)>::size();

    // Online softmax, log2 domain; O and l are rescaled when a row max grows by more than RESCALE_THRESHOLD
    constexpr D_ACC RESCALE_THRESHOLD = D_ACC(8.0f);
    D_ACC m_row = gqa_d512::M_INIT;
    D_ACC l_row = 0.0f;
    D_ACC rescale_m = 1.0f;

    // Causal, bottom-right: row i sees keys [0, i + sk - sq]
    const int kv_tile_stride = T::KV_TILE_SIZE * sc.kv_stride;
    [[maybe_unused]] const int causal_offset = seqlen_kv - seqlen_q;
    int max_num_tiles = ceil_div(seqlen_kv, T::KV_TILE_SIZE);
    if constexpr (T::CAUSAL) {
        const int q_end = q_rows < q_block_size ? seqlen_q : q_block_start + q_block_size;
        max_num_tiles = ceil_div(q_end + causal_offset, T::KV_TILE_SIZE);  // <= 0 without keys
    }
    auto kv_tile = [&](int tile_idx) { return tile_idx * kv_tile_stride; };
    // prefetches past the last tile load it again: constant vmcnt waits
    auto clamp_tile = [&](int x) { return x < max_num_tiles ? x : max_num_tiles - 1; };
    auto skv_slice = [](auto slice_idx) {
        constexpr int s = decltype(slice_idx)::value;
        return opus::number<(s / 2) * T::smem_n_rpt * (T::smem_linear_wave + T::smem_padding_32B) + (s % 2) * T::SLICE_D>{};
    };

    auto compute_qk = [&](auto& s, const auto& q, auto& k, auto& sk) {
        clear(s);
        static_for<T::NUM_D_SLICES>([&](auto i) {
            constexpr int idx = i.value;
            constexpr int slot = idx & 1;
            s = mma0(q[idx], k[slot], s);
            if constexpr (idx + 2 < T::NUM_D_SLICES) {
                k[slot] = load<T::VEC_KV>(sk, u_rk + skv_slice(number<idx + 2>{}));
                s_waitcnt_lgkmcnt(number<T::k_ds_read_insts>{});
            } else if constexpr (idx + 1 < T::NUM_D_SLICES) {
                s_waitcnt_lgkmcnt(0_I);
            }
        });
    };

    auto compute_pv = [&](const auto& p, auto& v, auto& o, auto& sv) {
        static_for<T::NUM_D_SLICES - 2>([&](auto i) {
            constexpr int idx = i.value;
            constexpr int slot = idx & 1;
            o[idx] = mma1(p, v[slot], o[idx]);
            v[slot] = tr_load<T::VEC_TR_V>(sv, u_rv + skv_slice(number<idx + 2>{}));
            s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
            __builtin_amdgcn_sched_barrier(0);
        });
        o[T::NUM_D_SLICES - 2] = mma1(p, v[(T::NUM_D_SLICES - 2) & 1], o[T::NUM_D_SLICES - 2]);
        s_waitcnt_lgkmcnt(0_I);
        o[T::NUM_D_SLICES - 1] = mma1(p, v[(T::NUM_D_SLICES - 1) & 1], o[T::NUM_D_SLICES - 1]);
    };

    // the wave's first row in KV coordinates
    [[maybe_unused]] const int q_start_pos_c = q_block_start + warp_id * T::Q_TILE_SIZE + causal_offset;
    const opus::u32_t neg_inf_v = gqa_d512::NEG_INF_BITS;
    [[maybe_unused]] const bool kv_tail = seqlen_kv % T::KV_TILE_SIZE != 0;
    auto mask = [&](auto& s, int t) {
        if constexpr (T::CAUSAL) {
            if (q_start_pos_c < (t + 1) * T::KV_TILE_SIZE)
                attn_mask_tile<T>(s, q_start_pos_c + lane_id % T::W_M, t, neg_inf_v, lane_id);
        } else {
            if (kv_tail && t == max_num_tiles - 1) attn_mask_tile<T>(s, seqlen_kv - 1, t, neg_inf_v, lane_id);
        }
    };

    // O and the LSE with the sink
    auto store_result = [&]() {
        const gqa_d512::row_final r(m_row, l_row, sc.sink);
        if (sc.lse != nullptr && lane_id < T::W_M) {
            auto g_lse = make_gmem(sc.lse, gqa_d512::buffer_bytes(int64_t(q_rows) * sc.lse_stride * sizeof(float)));
            g_lse.store(r.lse(), (warp_id * T::Q_TILE_SIZE + lane_id) * sc.lse_stride);
        }
        static_for<o_len>([&](auto i) { v_o[i.value] *= r.o_scale; });
        auto u_o = make_layout_o<T>(warp_id, lane_id, sc.q_stride);
        auto v_o_bf16 = opus::cast<D_ATTN>(v_o);
        store<T::VEC_O>(g_o, v_o_bf16, u_o);
    };

    // no keys for the tile: O = 0
    if (max_num_tiles <= 0) {
        store_result();
        return;
    }

    auto stage_end = [&]() {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };

    // Phase t: QK(t) and softmax head of t; softmax tail and PV of t - 1. K and V prefetched one tile ahead.
    auto run_phase = [&](auto& vs_cur, auto& vs_prev, int cur, int prev, int t) {
        auto sk_cur = s_k(cur), sv_cur = s_v(cur), sv_prev = s_v(prev);
        // cluster 0 [mem]: read K(t), load K(t + 1)
        async_load<T::VEC_KV>(g_k, s_k(prev).ptr, u_gkv, u_skv, kv_tile(clamp_tile(t + 1)));
        v_k[0] = load<T::VEC_KV>(sk_cur, u_rk);
        v_k[1] = load<T::VEC_KV>(sk_cur, u_rk + skv_slice(1_I));
        s_waitcnt_lgkmcnt(number<T::k_ds_read_insts>{});
        s_waitcnt_vmcnt(number<T::k_buffer_load_insts>{});
        stage_end();

        // cluster 1 [compute]: QK(t); softmax tail of t - 1
        __builtin_amdgcn_s_setprio(1);
        compute_qk(vs_cur, v_q_slices, v_k, sk_cur);
        attn_exp2_slice<T, s_half_len, s_half_len>(vs_prev);
        l_row += attn_row_sum<T>(vs_prev);
        v_p = opus::cast<D_ATTN>(vs_prev);
        asm volatile("" : "+v"(v_p) ::);
        sched_compute_qk<0>();
        __builtin_amdgcn_s_setprio(0);
        stage_end();

        // cluster 2 [mem]: read V(t - 1), load V(t); mask of t
        async_load<T::VEC_KV>(g_v, sv_cur.ptr, u_gkv, u_skv, kv_tile(t));
        v_v[0] = tr_load<T::VEC_TR_V>(sv_prev, u_rv);
        v_v[1] = tr_load<T::VEC_TR_V>(sv_prev, u_rv + skv_slice(1_I));
        mask(vs_cur, t);
        s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
        s_waitcnt_vmcnt(number<T::v_buffer_load_insts>{});
        stage_end();

        // cluster 3 [compute]: PV(t - 1); softmax head of t
        __builtin_amdgcn_s_setprio(1);
        compute_pv(v_p, v_v, v_o_slices, sv_prev);
        D_ACC row_max = attn_row_max<T>(vs_cur);
        bool below_thresh = ((row_max - m_row) <= RESCALE_THRESHOLD);
        bool all_below = (__builtin_amdgcn_ballot_w64(below_thresh) == __builtin_amdgcn_read_exec());
        row_max = all_below ? m_row : max(m_row, row_max);
        attn_sub_row<T>(vs_cur, row_max);
        attn_exp2_slice<T, 0, s_half_len>(vs_cur);
        asm volatile("" : "+v"(vs_cur) ::);
        __builtin_amdgcn_sched_barrier(0);
        if (!all_below) {
            rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
            l_row *= rescale_m;
            m_row = row_max;
            scale_output_tile<T>(v_o, rescale_m);
        }
        __builtin_amdgcn_s_setprio(0);
        stage_end();
    };

    // Prologue: QK(0) and its softmax head
    auto sk0 = s_k(0);
    async_load<T::VEC_KV>(g_k, sk0.ptr, u_gkv, u_skv, kv_tile(0));
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    v_q = load<T::VEC_Q>(g_q, u_q);
    auto v_q_f32 = opus::cast<float>(v_q);
    static_for<q_len>([&](auto i) { v_q_f32[i.value] *= sc.c; });
    v_q = opus::cast<D_ATTN>(v_q_f32);

    async_load<T::VEC_KV>(g_k, s_k(1).ptr, u_gkv, u_skv, kv_tile(clamp_tile(1)));
    async_load<T::VEC_KV>(g_v, s_v(0).ptr, u_gkv, u_skv, kv_tile(0));
    v_k[0] = load<T::VEC_KV>(sk0, u_rk);
    v_k[1] = load<T::VEC_KV>(sk0, u_rk + skv_slice(1_I));
    __builtin_amdgcn_sched_barrier(0);
    s_waitcnt_lgkmcnt(number<T::k_ds_read_insts>{});
    s_waitcnt_vmcnt(number<T::v_buffer_load_insts>{});  // K(1)

    compute_qk(v_s[0], v_q_slices, v_k, sk0);

    // stagger: one barrier more now, the other group one more at the end
    if (stagger) {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    mask(v_s[0], 0);
    m_row = attn_row_max<T>(v_s[0]);
    attn_sub_row<T>(v_s[0], m_row);
    attn_exp2_slice<T, 0, s_half_len>(v_s[0]);
    asm volatile("" : "+v"(v_s[0]) ::);
    stage_end();

    // Main loop: one phase per iteration (two spill), S(t - 1) in v_s[0], S(t) in v_s[1]
    for (int t = 1; t < max_num_tiles; ++t) {
        run_phase(v_s[1], v_s[0], t & 1, (t & 1) ^ 1, t);
        v_s[0] = v_s[1];
    }

    // Epilogue: softmax tail and PV of tile T - 1
    s_waitcnt_vmcnt(0_I);  // V(T - 1)
    stage_end();
    __builtin_amdgcn_s_setprio(1);
    attn_exp2_slice<T, s_half_len, s_half_len>(v_s[0]);
    l_row += attn_row_sum<T>(v_s[0]);
    v_p = opus::cast<D_ATTN>(v_s[0]);
    asm volatile("" : "+v"(v_p) ::);
    __builtin_amdgcn_s_setprio(0);
    stage_end();
    auto sv_last = s_v((max_num_tiles - 1) & 1);
    v_v[0] = tr_load<T::VEC_TR_V>(sv_last, u_rv);
    v_v[1] = tr_load<T::VEC_TR_V>(sv_last, u_rv + skv_slice(1_I));
    s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
    stage_end();
    compute_pv(v_p, v_v, v_o_slices, sv_last);

    if (!stagger) {
        __builtin_amdgcn_s_barrier();
    }

    store_result();
}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void gqa_d512_kernel(opus_gqa_varlen_kargs kargs) {
    __shared__ char smem_buf[2 * Traits::smem_buffer_elems * sizeof(typename Traits::D_ATTN)];
    gqa_d512::for_each_tile<Traits::D_TILE_SIZE, Traits::WG_Q_ROWS>(
        kargs, [&](const gqa_d512::seq_ctx& sc) { gqa_d512_tile<Traits>(sc, smem_buf); });
}
