// Paged sparse attention kernel template for D=512 on gfx1250 (wave32 / WMMA).
// 16mx4_64nx1 variant (T_M=4, T_N=1); include from a .cc that instantiates the traits.
#pragma once

#include <opus/opus.hpp>
#include "pa_traits.h"
#include <cstdint>
#include <bit>

using opus::operator""_I;

namespace pa_16mx4_64nx1 {

constexpr int VALU_MASK    = 0x002;
constexpr int SALU_MASK    = 0x004;
constexpr int MFMA_MASK    = 0x008;
constexpr int DS_READ_MASK = 0x100;
constexpr int EXP_MASK     = 0x400;
constexpr int TDM_MASK     = 0x800;

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

template<class T>
__device__ inline auto make_layout_q(int warp_id, int lane_id, int stride_q_h) {
    constexpr auto q_block_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::D_TILE_SIZE / T::W_K>{},
        opus::number<T::W_M * T::W_K / (T::WARP_SIZE * T::VEC_Q)>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_Q>{});

    constexpr auto q_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        q_block_shape,
        opus::unfold_x_stride(q_block_dim, q_block_shape, opus::tuple{stride_q_h, 1_I}),
        opus::unfold_p_coord(q_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

template<class T>
__device__ inline auto make_layout_o(int warp_id, int lane_id, int stride_o_h) {
    constexpr int dwordx32_rpt = 4 * 32 / sizeof(typename T::D_ATTN) / T::VEC_O;

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

template<class T>
__device__ inline auto make_layout_rk(int lane_id) {
    constexpr auto k_block_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_N>{},
        opus::number<T::W_N>{},
        opus::number<T::D_TILE_SIZE / T::W_K>{},
        opus::number<T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_KV)>{},
        opus::number<T::WARP_SIZE / T::W_N>{},
        opus::number<T::VEC_KV>{});

    constexpr auto k_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        k_block_shape,
        opus::unfold_x_stride(k_block_dim, k_block_shape, opus::tuple{opus::number<T::D_TILE_SIZE + T::KV_ROW_PAD_SIZE>{}, 1_I}),
        opus::unfold_p_coord(k_block_dim, opus::tuple{lane_id % T::W_N, lane_id / T::W_N}));
}

template<class T>
__device__ inline auto make_layout_rv(int lane_id) {
    constexpr int lane_per_grp = 16;
    constexpr int lane_n = 2;
    constexpr int lane_k = lane_per_grp / lane_n;

    constexpr int dwordx32_rpt = 4 * 32 / sizeof(typename T::D_ATTN) / T::VEC_KV;

    constexpr auto v_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_N / dwordx32_rpt>{},
        opus::number<lane_n>{},
        opus::number<dwordx32_rpt>{},
        opus::number<T::W_K / (T::WARP_SIZE / lane_per_grp) / T::VEC_KV>{},
        opus::number<T::WARP_SIZE / lane_per_grp>{},
        opus::number<lane_k>{},
        opus::number<T::VEC_KV>{});
    
    constexpr auto v_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    
    return opus::make_layout(
        v_block_shape,
        opus::unfold_x_stride(v_block_dim, v_block_shape, opus::tuple{opus::number<T::VEC_KV>{}, opus::number<T::D_TILE_SIZE + T::KV_ROW_PAD_SIZE>{}, 1_I}),
        opus::unfold_p_coord(v_block_dim, opus::tuple{(lane_id % lane_per_grp) / lane_k, lane_id / lane_per_grp, (lane_id % lane_per_grp) % lane_k}));
}

template<int REG, int SLICE_REGS, typename V>
__device__ inline V pin_tile(V v) {
    using D = typename opus::vector_traits<V>::dtype;
    constexpr int slice_elems = SLICE_REGS * 4 / (int)sizeof(D);
    constexpr int slices      = opus::vector_traits<V>::size() / slice_elems;
    static_assert(slices * slice_elems == opus::vector_traits<V>::size());

    auto s = reinterpret_cast<opus::vector_t<D, slice_elems>*>(&v);
    opus::static_for<slices>([&](auto i) {
        s[i.value] = __builtin_amdgcn_pin_vgpr(s[i.value], REG + i.value * SLICE_REGS);
    });
    return v;
}

template<class T>
struct pin_layout {
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    // Widths one WMMA consumes as a unit, which is the granularity a pin has to
    // name: pinning a whole tile and reading it back in pieces leaves the pieces
    // with no place of their own.
    static constexpr int B_REGS   = T::W_N * T::W_K / T::WARP_SIZE * (int)sizeof(D_ATTN) / 4;
    static constexpr int ACC_REGS = T::W_M * T::W_N / T::WARP_SIZE * (int)sizeof(D_ACC) / 4;

    static constexpr int Q_REGS  = T::Q_TILE_SIZE * T::D_TILE_SIZE / T::WARP_SIZE * (int)sizeof(D_ATTN) / 4;
    static constexpr int O_REGS  = T::Q_TILE_SIZE * T::D_TILE_SIZE / T::WARP_SIZE * (int)sizeof(D_ACC) / 4;
    static constexpr int KV_REGS = T::GEMM0_E_N * T::GEMM0_E_K * T::W_N * T::W_K / T::WARP_SIZE * (int)sizeof(D_ATTN) / 4;
    static constexpr int O_STAGE_REGS = O_REGS / T::GEMM1_STAGE_N;

    static constexpr int S_REGS = T::Q_TILE_SIZE * T::KV_TILE_SIZE / T::WARP_SIZE * (int)sizeof(D_ACC) / 4;
    static constexpr int P_REGS = T::Q_TILE_SIZE * T::KV_TILE_SIZE / T::WARP_SIZE * (int)sizeof(D_ATTN) / 4;

    static constexpr int Q   = 0;               // v0   .. v127
    static constexpr int O   = Q + Q_REGS;      // v128 .. v383
    static constexpr int KV0 = O + O_REGS;      // v384 .. v511
    static constexpr int KV1 = KV0 + KV_REGS;   // v512 .. v639
    static constexpr int S0  = KV1 + KV_REGS;   // v640 .. v671
    static constexpr int S1  = S0 + S_REGS;     // v672 .. v703
    static constexpr int P   = S1 + S_REGS;     // v704 .. v719
    static constexpr int M    = P + P_REGS;     // v720, the running row max
    static constexpr int LSUM = M + 1;          // v721, the running row sum
    static constexpr int RMAX = LSUM + 1;       // v722, this tile's row max

    static_assert(P_REGS == T::GEMM1_STAGE_K * B_REGS, "v_p must tile into WMMA operands");
    static_assert(S_REGS == T::GEMM0_STAGE_N * ACC_REGS, "v_s must tile into WMMA accumulators");
    static_assert(T::GEMM0_E_N * T::GEMM0_E_K == T::GEMM1_E_N * T::GEMM1_E_K,
                  "K and V must be the same width to share a slot");
    static_assert(RMAX + 1 <= 1024, "the layout must fit the gfx1250 VGPR file");
};

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

    int res16 = __builtin_amdgcn_permlane_xor(std::bit_cast<int>(row_max), 16, 32);
    return max(row_max, std::bit_cast<float>(res16));
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

    int res16 = __builtin_amdgcn_permlane_xor(std::bit_cast<int>(row_sum), 16, 32);
    return row_sum + std::bit_cast<float>(res16);
}

template<typename T, typename V>
__device__ inline void scale_output_tile(V& v_o, typename T::D_ACC scale) {
    constexpr opus::index_t o_len = opus::vector_traits<V>::size();
    opus::static_for<o_len>([&](auto i) { v_o[i.value] *= scale; });
}

template<typename T, typename V>
__device__ inline void attn_mask_oob_score(V& v_s, int valid_kv_len, int kv_tile_idx, int seg_rot_rows) {
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
        // odd waves hold the rotated KV rows
        const int rel = rel_base - ((i_s.value * T::W_N + seg_rot_rows) & (T::KV_TILE_SIZE - 1));
        opus::static_for<elems_per_stage>([&](auto i_reg) {
            constexpr int idx = i_s.value * elems_per_stage + i_reg.value;
            v_s[idx] = (i_reg.value > rel) ? neg_inf : v_s[idx];
        });
    });
}

template<class Traits>
__device__ __attribute__((always_inline)) void pa_prefill_accum_pipelined(pa_kargs kargs,
                                           const void* kv_ptr, int kv_rows,
                                           const int* kv_indices, int page_idx_begin, int valid_kv_len, int num_kv_tiles,
                                           char* smem_kv_buf,
                                           opus::vector_t<typename Traits::D_ATTN, Traits::Q_TILE_SIZE * Traits::D_TILE_SIZE / Traits::WARP_SIZE>& v_q,
                                           opus::vector_t<typename Traits::D_ACC,  Traits::Q_TILE_SIZE * Traits::D_TILE_SIZE / Traits::WARP_SIZE>& v_o,
                                           typename Traits::D_ACC& m_row,
                                           typename Traits::D_ACC& l_row,
                                           typename Traits::D_ACC temperature_scale) {
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC = typename T::D_ACC;

    if (num_kv_tiles <= 0) return;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    // Odd waves start on the other LDS half to spread traffic.
    const int seg_rot_bytes = (warp_id & 1) * T::SEG_BYTES;
    const int seg_rot_rows  = (warp_id & 1) * T::ROWS_PER_SEG;

    // Three tile slots: QK(t+2) runs before PV(t+1), so t, t+1 and the in-flight t+2 coexist.
    smem<D_ATTN> s_qk[T::SEGS_PER_BUF] = {
        make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf + seg_rot_bytes)),
        make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf + (T::SEG_BYTES - seg_rot_bytes))),
    };
    smem<D_ATTN> s_pv[T::SEGS_PER_BUF] = { s_qk[0], s_qk[1] };

    constexpr int SLOT_BYTES = T::KV_BUF_BYTES;
    constexpr int RING_BYTES = T::NUM_KV_BUFS * T::KV_BUF_BYTES;
    int qk_off = 0, pv_off = 0, tdm_off = 0;

    auto slot_step = [](int& off) {
        const int prev = off;
        const int next = off + SLOT_BYTES;
        off = (next == RING_BYTES) ? 0 : next;
        return off - prev;
    };
    auto advance = [&](auto& s, int& off) {
        const int delta = slot_step(off);
        static_for<T::SEGS_PER_BUF>([&](auto i) { s[i.value].ptr += delta; });
    };

    const u32x4_t kv_indices_rsrc = make_buffer_rsrc_raw(kv_indices + page_idx_begin, (u32_t)(valid_kv_len * sizeof(int)));

    constexpr tdm_cfg kv_gather_cfg{
        .tile_dim          = { (u32_t)T::D_TILE_SIZE, (u32_t)T::INDICES_PER_TDM },
        .gather            = true,
        .gather_index_size = 1,
        .lds_pad_en        = true,
        .pad_interval      = 7,
        .pad_amount        = 3,
    };

    auto tdm_kv = make_tdm<D_ATTN, kv_gather_cfg>(
        smem_kv_buf + (warp_id / T::WAVES_PER_SEG) * T::SEG_BYTES + (warp_id % T::WAVES_PER_SEG) * T::WAVE_LDS_BYTES,
        reinterpret_cast<const D_ATTN*>(kv_ptr),
        /*lds_off=*/ 0,
        /*td0=*/ T::D_TILE_SIZE,
        /*td1=*/ kv_rows,
        /*s0=*/ kargs.stride_kv_page);

    auto mma0 = make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(
        seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        wmma_adaptor_swap_ab{});
    auto mma1 = make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(
        seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        wmma_adaptor_swap_ab{});

    auto u_rk = make_layout_rk<T>(lane_id);
    auto u_rv = make_layout_rv<T>(lane_id);

    vector_t<D_ATTN, T::GEMM0_E_N * T::GEMM0_E_K * T::W_N * T::W_K / T::WARP_SIZE> v_k[2];
    vector_t<D_ATTN, T::GEMM1_E_N * T::GEMM1_E_K * T::W_N * T::W_K / T::WARP_SIZE> v_v[2];
    vector_t<D_ACC, T::Q_TILE_SIZE * T::KV_TILE_SIZE / T::WARP_SIZE> v_s[2];
    vector_t<D_ATTN, T::Q_TILE_SIZE * T::KV_TILE_SIZE / T::WARP_SIZE> v_p;
    auto v_p_stages = reinterpret_cast<vector_t<D_ATTN, T::W_M * T::W_K / T::WARP_SIZE>*>(&v_p);
    auto v_o_stages = reinterpret_cast<vector_t<D_ACC, T::W_M * T::D_TILE_SIZE / T::GEMM1_STAGE_N / T::WARP_SIZE>*>(&v_o);

    using L = pin_layout<T>;
    auto pin_o = [&] { v_o = pin_tile<L::O, L::ACC_REGS>(v_o); };

    auto anchor_s = [&](auto buf, auto lo, auto hi) {
        constexpr int per_stage = T::W_M * T::W_N / T::WARP_SIZE;
        auto* e = reinterpret_cast<D_ACC*>(&v_s[buf.value]);
        #pragma unroll
        for (int i = lo.value * per_stage; i < hi.value * per_stage; ++i)
            asm volatile("" : "+v"(e[i]) ::);
    };
    constexpr auto S_ALL  = number<T::GEMM0_STAGE_N>{};
    constexpr auto S_HALF = number<T::GEMM0_STAGE_N / 2>{};

    // Online softmax state, sliced into the GEMM stage loops below.
    constexpr D_ACC RESCALE_THRESHOLD = D_ACC(8.0f);
    constexpr index_t s_len = T::Q_TILE_SIZE * T::KV_TILE_SIZE / T::WARP_SIZE;
    static_assert(s_len / 2 % (T::W_M * T::W_N / T::WARP_SIZE) == 0,
                  "a softmax half must end on a v_s stage boundary for anchor_s to cover it");
    D_ACC row_max;
    bool all_below;

    u32x16_t row_ids;

    auto load_row_ids = [&](int tile_idx) {
        const int idx_byte_off = (tile_idx * T::KV_TILE_SIZE + warp_id * T::ROWS_PER_WAVE) * (int)sizeof(int);
        return s_buffer_load_b512(kv_indices_rsrc, idx_byte_off);
    };

    auto issue_kv_tile = [&](const u32x16_t& ids, int tile_idx, auto clamp_tail) {
        constexpr int lds_step = T::INDICES_PER_TDM * T::KV_ROW_LDS_BYTES;
        [[maybe_unused]] const int wave_valid = valid_kv_len - (tile_idx * T::KV_TILE_SIZE + warp_id * T::ROWS_PER_WAVE);

        static_for<T::TDM_LOADS_PER_WAVE>([&](auto d) {
            constexpr int ld = d.value;
            static_for<T::INDICES_PER_TDM>([&](auto r) {
                constexpr int slot = ld * T::INDICES_PER_TDM + r.value;
                u32_t id = ids[slot];
                if constexpr (decltype(clamp_tail)::value) id = slot < wave_valid ? id : (u32_t)kv_rows;
                tdm_kv.set_gather_row_index(r.value, __builtin_amdgcn_readfirstlane(id));
            });
            tdm_kv.load();
            if constexpr (ld + 1 < T::TDM_LOADS_PER_WAVE)
                tdm_kv.move(0_I, 0_I, 0_I, 0_I, 0_I, number<lds_step>{});
        });
        tdm_kv.move(0_I, 0_I, 0_I, 0_I, 0_I, number<-(T::TDM_LOADS_PER_WAVE - 1) * lds_step>{});
    };

    // Gather a tile into the next slot, prefetch the row indices after it.
    auto issue_tile = [&](int tile, auto clamp_tail) {
        s_wait_kmcnt_for(row_ids);
        tdm_kv.move(0_I, 0_I, 0_I, 0_I, 0_I, slot_step(tdm_off));
        issue_kv_tile(row_ids, tile, clamp_tail);
        row_ids = load_row_ids(tile + 1);
    };

    auto load_k = [&](auto sn, auto buf) {
        constexpr int kv_row  = sn.value * T::W_N;
        constexpr int seg     = kv_row / T::ROWS_PER_SEG;
        constexpr int row_off = (kv_row % T::ROWS_PER_SEG) * T::KV_ROW_LDS_ELEMS;
        constexpr int reg     = buf.value ? L::KV1 : L::KV0;
        v_k[buf.value] = pin_tile<reg, L::B_REGS>(load<T::VEC_KV>(s_qk[seg], u_rk + number<row_off>{}));
    };

    auto tr_load_v = [&](auto sn, auto sk, auto buf) {
        constexpr int stage_n = sn.value;
        constexpr int stage_k = sk.value;
        constexpr int reg     = buf.value ? L::KV1 : L::KV0;
        v_v[buf.value] = pin_tile<reg, L::B_REGS>(tr_load<T::VEC_KV>(s_pv[stage_k], u_rv + number<stage_n * (T::D_TILE_SIZE / T::GEMM1_STAGE_N)>{}));
    };

    auto compute_qk = [&](auto dst, auto prev, auto fuse_softmax, int tile, auto clamp_tail) __attribute__((always_inline)) {
        auto v_s_stages = reinterpret_cast<vector_t<D_ACC, T::W_M * T::W_N / T::WARP_SIZE>*>(&v_s[dst.value]);
        static_for<T::GEMM0_STAGE_N>([&](auto j) {
            constexpr int stage = j.value;
            constexpr int buf   = j.value & 1;
            __builtin_amdgcn_sched_barrier(0);
            s_wait_dscnt(0_I);
            __builtin_amdgcn_sched_barrier(0);

            constexpr int s_reg = (dst.value ? L::S1 : L::S0) + stage * L::ACC_REGS;
            clear(v_s_stages[stage]);
            v_s_stages[stage] = pin_tile<s_reg, L::ACC_REGS>(mma0(v_q, v_k[buf], v_s_stages[stage]));

            if constexpr (stage == 0) {
                issue_tile(tile, clamp_tail);
                __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                __builtin_amdgcn_sched_group_barrier(SALU_MASK,    6, 0);   // ring step + D# lds_addr
                __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                __builtin_amdgcn_sched_group_barrier(TDM_MASK,     1, 0);   // tensor_load_to_lds #0
                __builtin_amdgcn_sched_group_barrier(SALU_MASK,    5, 0);   // lds bump + 2nd D#
                __builtin_amdgcn_sched_group_barrier(TDM_MASK,     1, 0);   // tensor_load_to_lds #1
            }

            if constexpr (stage + 1 < T::GEMM0_STAGE_N) load_k(number<stage + 1>{}, number<(stage + 1) & 1>{});
            else                                        tr_load_v(0_I, 0_I, 0_I);

            if constexpr (decltype(fuse_softmax)::value) {
                if constexpr (stage == 0) {
                    attn_exp2_slice<T, s_len / 2, s_len / 2>(v_s[prev.value]);
                    anchor_s(prev, S_HALF, S_ALL);
                    static_for<8>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, 0);
                        __builtin_amdgcn_sched_group_barrier(EXP_MASK,     1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 2, 0);
                        __builtin_amdgcn_sched_group_barrier(EXP_MASK,     1, 0);
                    });
                    static_for<4>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK,     1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK,  2, 0);
                    });
                }
                if constexpr (stage == 1) {
                    l_row += attn_row_sum<T>(v_s[prev.value]);
                    __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                    __builtin_amdgcn_sched_group_barrier(VALU_MASK,    2, 0);
                    static_for<7>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, 0);
                        __builtin_amdgcn_sched_group_barrier(VALU_MASK,    1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 2, 0);
                        __builtin_amdgcn_sched_group_barrier(VALU_MASK,    1, 0);
                    });
                    static_for<4>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 2, 0);
                        __builtin_amdgcn_sched_group_barrier(VALU_MASK,    1, 0);
                    });
                    static_for<4>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 2, 0);
                    });
                }
                if constexpr (stage == 2) {
                    v_p = pin_tile<L::P, L::B_REGS>(cast<D_ATTN>(v_s[prev.value]));
                    #pragma unroll
                    for (int k = 0; k < T::GEMM1_STAGE_K; ++k) asm volatile("" : "+v"(v_p_stages[k]) ::);
                    static_for<10>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 3, 0);
                    });
                    __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                    __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(VALU_MASK,    1, 0);
                    __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, 0);
                    static_for<5>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, 0);
                        __builtin_amdgcn_sched_group_barrier(VALU_MASK, 3, 0);
                    });
                }
                if constexpr (stage == 3) {
                    __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(VALU_MASK, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(SALU_MASK, 6, 0);
                    static_for<10>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 3, 0);
                    });
                    __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                    __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 2, 0);
                    static_for<4>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, 0);
                    });
                }
            }
        });
        advance(s_qk, qk_off);
    };

    auto compute_pv = [&](auto cur, auto fuse_softmax) __attribute__((always_inline)) {
        constexpr int GEMM1_STAGES = T::GEMM1_STAGE_N * T::GEMM1_STAGE_K;
        static_for<GEMM1_STAGES>([&](auto sg) {
            constexpr int stage_n = sg.value % T::GEMM1_STAGE_N;
            constexpr int stage_k = sg.value / T::GEMM1_STAGE_N;
            constexpr int buf     = sg.value & 1;

            __builtin_amdgcn_sched_barrier(0);
            s_wait_dscnt(0_I);
            __builtin_amdgcn_sched_barrier(0);

            v_o_stages[stage_n] = pin_tile<L::O + stage_n * L::O_STAGE_REGS, L::ACC_REGS>(
                mma1(v_p_stages[stage_k], v_v[buf], v_o_stages[stage_n]));

            if constexpr (sg.value + 1 < GEMM1_STAGES) {
                constexpr int n_sn = (sg.value + 1) % T::GEMM1_STAGE_N;
                constexpr int n_sk = (sg.value + 1) / T::GEMM1_STAGE_N;
                tr_load_v(number<n_sn>{}, number<n_sk>{}, number<(sg.value + 1) & 1>{});
            } else if constexpr (decltype(fuse_softmax)::value) {
                s_wait_tensorcnt(0_I);
                __builtin_amdgcn_s_barrier();
                load_k(0_I, 0_I);
                constexpr int K_ANCHOR_STRIDE = T::GEMM0_E_K / T::GEMM0_STAGE_N;
                auto ks = reinterpret_cast<vector_t<D_ATTN, T::W_N * T::W_K / T::WARP_SIZE>*>(&v_k[0]);
                #pragma unroll
                for (int i = 0; i < T::GEMM0_E_K; i += K_ANCHOR_STRIDE)
                    asm volatile("" : "+v"(ks[i]) ::);
            }

            if constexpr (decltype(fuse_softmax)::value) {
                if constexpr (sg.value == 0) {
                    row_max = attn_row_max<T>(v_s[cur.value]) * temperature_scale;
                    all_below = __builtin_amdgcn_ballot_w32((row_max - m_row) <= RESCALE_THRESHOLD)
                             == __builtin_amdgcn_read_exec_lo();
                    row_max = __builtin_amdgcn_pin_vgpr(all_below ? m_row : max(m_row, row_max), L::RMAX);
                    __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                    __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(VALU_MASK,    1, 0);
                    static_for<10>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, 0);
                        __builtin_amdgcn_sched_group_barrier(VALU_MASK,    1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 2, 0);
                        __builtin_amdgcn_sched_group_barrier(VALU_MASK,    1, 0);
                    });
                    __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                    __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(VALU_MASK,    1, 0);
                    static_for<3>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, 0);
                        __builtin_amdgcn_sched_group_barrier(VALU_MASK, 2, 0);
                    });
                    __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(SALU_MASK, 2, 0);
                    __builtin_amdgcn_sched_group_barrier(VALU_MASK, 1, 0);
                }
                if constexpr (sg.value == 1) {
                    attn_row_scale_sub<T>(v_s[cur.value], temperature_scale, row_max);
                    anchor_s(cur, 0_I, S_ALL);
                    __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(VALU_MASK, 1, 0);
                    static_for<8>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, 0);
                        __builtin_amdgcn_sched_group_barrier(VALU_MASK,    1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, 0);
                        __builtin_amdgcn_sched_group_barrier(VALU_MASK,    1, 0);
                    });
                    static_for<5>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK,     1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK,  3, 0);
                    });
                    __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                    __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                }
                if constexpr (sg.value == 2) {
                    attn_exp2_slice<T, 0, s_len / 2>(v_s[cur.value]);
                    anchor_s(cur, 0_I, S_HALF);
                    static_for<8>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, 0);
                        __builtin_amdgcn_sched_group_barrier(EXP_MASK,     1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, 0);
                        __builtin_amdgcn_sched_group_barrier(EXP_MASK,     1, 0);
                    });
                    static_for<5>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK,     1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK,  3, 0);
                    });
                    __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                    __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    2, 0);
                }
                if constexpr (sg.value == 3) {   // only valid once every mma1 has landed in v_o
                    static_for<10>([](auto) {
                        __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 3, 0);
                    });
                    __builtin_amdgcn_sched_group_barrier(MFMA_MASK,    1, 0);
                    __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 2, 0);
                    
                    if (!all_below) {
                        const D_ACC rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
                        m_row = row_max;
                        l_row *= rescale_m;
                        scale_output_tile<T>(v_o, rescale_m);
                        pin_o();
                    }
                }
            }
        });
        advance(s_pv, pv_off);
    };

    // Prologue
    row_ids = load_row_ids(0);
    s_wait_kmcnt_for(row_ids);
    issue_kv_tile(row_ids, 0, true_type{});
    row_ids = load_row_ids(1);
    s_wait_tensorcnt(0_I);
    __builtin_amdgcn_s_barrier();
    load_k(0_I, 0_I);
    compute_qk(0_I, 0_I, false_type{}, 1, true_type{});
    attn_mask_oob_score<T>(v_s[0], valid_kv_len, 0, seg_rot_rows);

    s_wait_tensorcnt(0_I);
    __builtin_amdgcn_s_barrier();
    load_k(0_I, 0_I);

    // Softmax head of tile 0
    row_max = attn_row_max<T>(v_s[0]) * temperature_scale;
    all_below = __builtin_amdgcn_ballot_w32((row_max - m_row) <= RESCALE_THRESHOLD)
             == __builtin_amdgcn_read_exec_lo();
    row_max = __builtin_amdgcn_pin_vgpr(all_below ? m_row : max(m_row, row_max), L::RMAX);
    attn_row_scale_sub<T>(v_s[0], temperature_scale, row_max);
    attn_exp2_slice<T, 0, s_len / 2>(v_s[0]);
    if (!all_below) {
        const D_ACC rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
        m_row = row_max;
        l_row *= rescale_m;
        scale_output_tile<T>(v_o, rescale_m);
        pin_o();
    }

    // Main loop
    int t = 1;
    for (; t + 3 < num_kv_tiles; t += 2) {
        compute_qk(1_I, 0_I, true_type{}, t + 1, false_type{});   // QK(t)   + tail(t-1) + gather(t+1)
        compute_pv(1_I, true_type{});                             // PV(t-1) + head(t)
        compute_qk(0_I, 1_I, true_type{}, t + 2, false_type{});   // QK(t+1) + tail(t)   + gather(t+2)
        compute_pv(0_I, true_type{});                             // PV(t)   + head(t+1)
    }

    // Epilogue
    #pragma clang loop unroll(disable)
    for (; t < num_kv_tiles; ++t) {
        compute_qk(1_I, 0_I, true_type{}, t + 1, true_type{});    // QK(t) + tail(t-1) + gather(t+1)
        attn_mask_oob_score<T>(v_s[1], valid_kv_len, t, seg_rot_rows);
        compute_pv(1_I, true_type{});                             // PV(t-1) + head(t)
        v_s[0] = v_s[1];
    }
    // Softmax tail of the last tile, with no GEMM left to ride along.
    attn_exp2_slice<T, s_len / 2, s_len / 2>(v_s[0]);
    l_row += attn_row_sum<T>(v_s[0]);
    v_p = pin_tile<L::P, L::B_REGS>(cast<D_ATTN>(v_s[0]));

    tr_load_v(0_I, 0_I, 0_I);
    compute_pv(0_I, false_type{});            // PV of the tile that ended the chain
}

} // namespace pa_16mx4_64nx1

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void pa_prefill_16mx4_64nx1_kernel(pa_kargs kargs) {
    using namespace opus;
    using namespace pa_16mx4_64nx1;
    using T = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC = typename T::D_ACC;

    const int q_token_idx = block_id_x();
    const int h_block_idx = block_id_y();
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    const int h_block_start = h_block_idx * T::NUM_WARPS * T::Q_TILE_SIZE;
    const int64_t qo_gmem_offset = static_cast<int64_t>(q_token_idx) * kargs.stride_qo_n + static_cast<int64_t>(h_block_start) * kargs.stride_qo_h;

    __shared__ char smem_kv_buf[T::smem_size_bytes()];

    // Load Q once (shared across both segments)
    auto g_q = make_gmem(reinterpret_cast<const D_ATTN*>(kargs.q_ptr) + qo_gmem_offset, (kargs.H - h_block_start) * kargs.stride_qo_h * sizeof(D_ATTN));
    auto u_q = make_layout_q<T>(warp_id, lane_id, kargs.stride_qo_h);

    vector_t<D_ATTN, T::Q_TILE_SIZE * T::D_TILE_SIZE / T::WARP_SIZE> v_q;
    vector_t<D_ACC,  T::Q_TILE_SIZE * T::D_TILE_SIZE / T::WARP_SIZE> v_o;

    constexpr D_ACC LOG2_E = 1.44269504089f;
    const D_ACC temperature_scale = kargs.softmax_scale * LOG2_E;

    using L = pin_layout<T>;
    v_q = pin_tile<L::Q, L::B_REGS>(load<T::VEC_Q>(g_q, u_q));
    s_wait_loadcnt(0_I);

    clear(v_o);
    v_o = pin_tile<L::O, L::ACC_REGS>(v_o);
    D_ACC m_row = __builtin_amdgcn_pin_vgpr(opus::numeric_limits<D_ACC>::lowest(), L::M);
    D_ACC l_row = __builtin_amdgcn_pin_vgpr(D_ACC(0.0f), L::LSUM);

    // Prefix segment: indices point into unified_kv[total_pages]
    {
        const int page_idx_begin = kargs.kv_indptr_prefix[q_token_idx];
        const int valid_kv_len   = kargs.kv_indptr_prefix[q_token_idx + 1] - page_idx_begin;
        const int num_kv_tiles   = ceil_div(valid_kv_len, T::KV_TILE_SIZE);
        pa_prefill_accum_pipelined<Traits>(kargs, kargs.unified_kv_ptr, kargs.total_pages, kargs.kv_indices_prefix, page_idx_begin, valid_kv_len, num_kv_tiles, smem_kv_buf, v_q, v_o, m_row, l_row, temperature_scale);
    }

    __builtin_amdgcn_s_barrier();

    // Extend segment: indices point into kv[total_tokens]
    {
        const int page_idx_begin = kargs.kv_indptr_extend[q_token_idx];
        const int valid_kv_len   = kargs.kv_indptr_extend[q_token_idx + 1] - page_idx_begin;
        const int num_kv_tiles   = ceil_div(valid_kv_len, T::KV_TILE_SIZE);
        pa_prefill_accum_pipelined<Traits>(kargs, kargs.kv_ptr, kargs.total_tokens, kargs.kv_indices_extend, page_idx_begin, valid_kv_len, num_kv_tiles, smem_kv_buf, v_q, v_o, m_row, l_row, temperature_scale);
    }

    // Sink finalization, normalize O, store to gmem
    const int sink_head_idx = h_block_start + warp_id * T::Q_TILE_SIZE + (lane_id % T::W_M);
    auto g_attn_sink = make_gmem(reinterpret_cast<const D_ACC*>(kargs.attn_sink_ptr), kargs.H * sizeof(D_ACC));
    D_ACC sink_log2 = load(g_attn_sink, sink_head_idx)[0] * LOG2_E;
    D_ACC m_final = max(m_row, sink_log2);
    D_ACC alpha = __builtin_amdgcn_exp2f(m_row - m_final);
    D_ACC l_final = l_row * alpha + __builtin_amdgcn_exp2f(sink_log2 - m_final);
    D_ACC o_scale = (l_final > D_ACC(0.0f)) ? (alpha / l_final) : D_ACC(0.0f);
    scale_output_tile<T>(v_o, o_scale);

    using D_OUT = typename T::D_OUT;
    auto g_o = make_gmem(reinterpret_cast<D_OUT*>(kargs.out_ptr) + qo_gmem_offset, (kargs.H - h_block_start) * kargs.stride_qo_h * sizeof(D_OUT));
    int lane_id_o = thread_id_x() % T::WARP_SIZE;
    int warp_id_o = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    auto u_o = make_layout_o<T>(warp_id_o, lane_id_o, kargs.stride_qo_h);
    store<T::VEC_O>(g_o, cast<D_OUT>(v_o), u_o);
}