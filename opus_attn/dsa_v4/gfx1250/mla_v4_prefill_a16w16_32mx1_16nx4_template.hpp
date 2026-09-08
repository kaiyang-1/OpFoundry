#pragma once

#include <opus/opus.hpp>
#include "mla_v4_traits.h"
#include <cstdint>
#include <bit>

using opus::operator""_I;

namespace opus_mla_v4_prefill_a16w16_32mx1_16nx4 {

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

template<class T>
__device__ inline auto make_layout_k(int lane_id) {
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
        opus::unfold_x_stride(k_block_dim, k_block_shape, opus::tuple{opus::number<T::KV_ROW_LDS_ELEMS>{}, 1_I}),
        opus::unfold_p_coord(k_block_dim, opus::tuple{lane_id % T::W_N, lane_id / T::W_N}));
}

template<class T>
__device__ inline auto make_layout_v(int lane_id) {
    constexpr int lane_per_grp = 16;
    constexpr int lane_n = 2;
    constexpr int lane_k = lane_per_grp / lane_n;

    constexpr int dwordx32_rpt = 4 * 32 / sizeof(typename T::D_ATTN) / T::VEC_KV;

    constexpr auto v_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_N / dwordx32_rpt>{},
        opus::number<lane_n>{},
        opus::number<dwordx32_rpt>{},
        opus::number<T::GEMM1_E_K * T::W_K / (T::WARP_SIZE / lane_per_grp) / T::VEC_KV>{},
        opus::number<T::WARP_SIZE / lane_per_grp>{},
        opus::number<lane_k>{},
        opus::number<T::VEC_KV>{});

    constexpr auto v_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        v_block_shape,
        opus::unfold_x_stride(v_block_dim, v_block_shape, opus::tuple{opus::number<T::VEC_KV>{}, opus::number<T::KV_ROW_LDS_ELEMS>{}, 1_I}),
        opus::unfold_p_coord(v_block_dim, opus::tuple{(lane_id % lane_per_grp) / lane_k, lane_id / lane_per_grp, (lane_id % lane_per_grp) % lane_k}));
}

template<int Lo, int Hi, typename V, typename Op>
__device__ inline auto tree_reduce(const V& v, Op op) {
    if constexpr (Hi - Lo == 1) return v[Lo];
    else {
        constexpr int Mid = (Lo + Hi) / 2;
        return op(tree_reduce<Lo, Mid>(v, op), tree_reduce<Mid, Hi>(v, op));
    }
}

template<class T, typename S>
__device__ inline void advance_kv_slot(S& s, int& off) {
    constexpr int WRAP = T::KV_LDS_BYTES - T::KV_BUF_BYTES;
    s.ptr += (off == WRAP) ? -WRAP : T::KV_BUF_BYTES;
    off    = (off == WRAP) ? 0 : off + T::KV_BUF_BYTES;
}

// The lane pair (lane, lane ^ 16) splits one M block's KV columns, hence the permlane.
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

// Every M block spans the same KV columns, so one relative bound masks them all.
template<typename T, typename V>
__device__ inline void attn_mask_oob_score(V& v_s, int valid_kv_len, int kv_tile_idx,
                                           int wave_kv_base, int lane_id) {
    using D_ACC = typename T::D_ACC;

    if ((kv_tile_idx + 1) * T::KV_TILE_SIZE <= valid_kv_len) return;

    constexpr int elems_per_mma   = T::W_M * T::W_N / T::WARP_SIZE;
    constexpr int lane_hi_kv_step = T::W_N / (T::WARP_SIZE / T::W_M);
    static_assert(opus::vector_traits<V>::size() == T::GEMM0_E_M * T::GEMM0_E_N * elems_per_mma);

    const D_ACC neg_inf = -opus::numeric_limits<D_ACC>::infinity();
    const int lane_hi = lane_id / T::W_M;   // {0, 1} on wave32
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

// One dword per lane publishes every M block: lane group lane_id / W_M holds its partial.
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
        store<T::VEC_P>(s_p, cast<typename T::D_ATTN>(f), blk * T::P_BLOCK_ELEMS + lane_id * T::VEC_P);
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

template<class Traits>
__device__ __attribute__((always_inline)) void mla_v4_prefill_accum_pipelined(
        opus_mla_v4_prefill_kargs kargs,
        const void* kv_ptr, int kv_rows,
        const int* kv_indices, int page_idx_begin, int valid_kv_len, int num_kv_tiles,
        char* smem_buf,
        opus::vector_t<typename Traits::D_ATTN, Traits::Q_TILE_SIZE * Traits::D_TILE_SIZE / Traits::WARP_SIZE>& v_q,
        opus::vector_t<typename Traits::D_ACC, Traits::Q_TILE_SIZE * Traits::D_TILE_SIZE / (Traits::T_N * Traits::WARP_SIZE)>& v_o,
        opus::vector_t<typename Traits::D_ACC, Traits::GEMM0_E_M>& m_row,
        opus::vector_t<typename Traits::D_ACC, Traits::GEMM0_E_M>& l_part,
        typename Traits::D_ACC temperature_scale) {
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC = typename T::D_ACC;

    if (num_kv_tiles <= 0) return;

    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int wave_kv_base = warp_id * T::ROWS_PER_WAVE;

    auto s_k = make_smem(reinterpret_cast<D_ATTN*>(smem_buf + warp_id * T::WAVE_LDS_BYTES));
    auto s_v = make_smem(reinterpret_cast<D_ATTN*>(smem_buf) + warp_id * (T::D_TILE_SIZE / T::T_N));
    auto s_m = make_smem(reinterpret_cast<D_ACC*>(smem_buf + T::M_LDS_OFF));
    auto s_p = make_smem(reinterpret_cast<D_ATTN*>(smem_buf + T::P_LDS_OFF));

    // s_k runs one slot ahead of s_v: round t reads K of tile t+1 and V of tile t.
    int k_off = 0, v_off = 0;

    const u32x4_t kv_indices_rsrc = make_buffer_rsrc_raw(kv_indices + page_idx_begin, (u32_t)(valid_kv_len * sizeof(int)));

    using kv_window = tdm<D_ATTN, seq<T::D_TILE_SIZE, T::INDICES_PER_TDM>,
                          tdm_traits::gather<32>,
                          tdm_traits::cache<tdm_traits::make_cache_policy(
                              tdm_traits::load_temporal_hint::regular, tdm_traits::scope::cu)>,
                          tdm_traits::padding_auto<D_ATTN, T::D_TILE_SIZE>>;

    auto tdm_kv = make_tdm<kv_window>((u32_t)reinterpret_cast<uintptr_t>(smem_buf + warp_id * T::WAVE_LDS_BYTES),
                                      reinterpret_cast<const D_ATTN*>(kv_ptr),
                                      /*shape0=*/ (u32_t)T::D_TILE_SIZE,
                                      /*shape1=*/ (u32_t)kv_rows,
                                      /*stride=*/ (u64_t)kargs.stride_kv_page);

    auto load_row_ids = [&](int tile_idx) {
        const int idx_byte_off = (tile_idx * T::KV_TILE_SIZE + wave_kv_base) * (int)sizeof(int);
        return s_buffer_load_b512(kv_indices_rsrc, idx_byte_off);
    };

    auto issue_kv_tile = [&](const u32x16_t& ids, int tile_idx, u32_t lds_slot_off, auto clamp_tail) {
        constexpr int LOAD_ELEMS = T::INDICES_PER_TDM * T::KV_ROW_LDS_ELEMS;
        [[maybe_unused]] const int wave_valid = valid_kv_len - (tile_idx * T::KV_TILE_SIZE + wave_kv_base);

        static_for<T::TDM_LOADS_PER_WAVE>([&](auto d) {
            constexpr int ld = d.value;
            u32_t idx[T::INDICES_PER_TDM];
            static_for<T::INDICES_PER_TDM>([&](auto r) {
                constexpr int slot = ld * T::INDICES_PER_TDM + r.value;
                u32_t id = ids[slot];
                if constexpr (decltype(clamp_tail)::value) id = slot < wave_valid ? id : (u32_t)kv_rows;
                idx[r.value] = __builtin_amdgcn_readfirstlane(id);
            });
            tdm_kv.set_indices(idx, T::INDICES_PER_TDM);
            tdm_kv.async_load(lds_slot_off + u32_t(ld * LOAD_ELEMS));
        });
    };

    u32x16_t row_ids = load_row_ids(0);
    int   gather_tile = 0;
    u32_t gather_off  = 0;
    auto issue_next_gather = [&]() {
        s_wait_kmcnt_for(row_ids);
        if (gather_tile + 1 < num_kv_tiles) issue_kv_tile(row_ids, gather_tile, gather_off, false_type{});
        else                                issue_kv_tile(row_ids, gather_tile, gather_off, true_type{});
        row_ids = load_row_ids(++gather_tile);
        gather_off = (gather_off + (u32_t)T::KV_BUF_ELEMS) % (u32_t)(T::NUM_KV_BUFS * T::KV_BUF_ELEMS);
    };

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

    auto u_rk = make_layout_k<T>(lane_id);
    auto u_rv = make_layout_v<T>(lane_id);

    constexpr D_ACC RESCALE_THRESHOLD = D_ACC(8.0f);
    constexpr index_t s_len_per_m = T::Q_TILE_SIZE * T::KV_TILE_SIZE / (T::T_N * T::WARP_SIZE) / T::GEMM0_E_M;
    constexpr index_t o_len_per_m = T::Q_TILE_SIZE * T::D_TILE_SIZE / (T::T_N * T::WARP_SIZE) / T::GEMM1_E_M;

    typename decltype(mma0)::vtype_b v_k;
    typename decltype(mma1)::vtype_b v_v;
    typename decltype(mma1)::vtype_a v_p;
    typename decltype(mma0)::vtype_c v_s, v_s_next;

    // Prologue: the entry state round 0 expects -- ring, scores, published row max
    static_for<T::GATHER_AHEAD>([&](auto) { issue_next_gather(); });
    s_wait_tensorcnt(number<T::TDM_LOADS_PER_WAVE>{});
    v_k = load<T::VEC_KV>(s_k, u_rk);
    advance_kv_slot<T>(s_k, k_off);
    clear(v_s);
    v_s = mma0(v_q, v_k, v_s);
    attn_mask_oob_score<T>(v_s, valid_kv_len, 0, wave_kv_base, lane_id);
    ml_publish<T>(s_m, attn_row_max_blocks<T>(v_s), warp_id, lane_id);
    s_wait_dscnt(0_I);
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);

    auto round = [&](int tile, auto mask_next) __attribute__((always_inline)) {
        // (c) softmax(t)
        auto tile_max = ml_reduce<T>(s_m, lane_id, [](D_ACC x, D_ACC y) { return max(x, y); });
        bool below = true;
        static_for<T::GEMM0_E_M>([&](auto i) {
            tile_max[i.value] *= temperature_scale;
            below = below && ((tile_max[i.value] - m_row[i.value]) <= RESCALE_THRESHOLD);
        });
        // One ballot for all M blocks: rescaling every block or none stays correct.
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

        v_k = load<T::VEC_KV>(s_k, u_rk);
        advance_kv_slot<T>(s_k, k_off);
        s_wait_dscnt(number<T::k_ds_load_insts>{});   // the P stores, issued ahead of these reads
        __builtin_amdgcn_s_barrier_signal(-1);

        // (d) QK(t+1); without the sched_barrier most of it sinks past the wait
        clear(v_s_next);
        v_s_next = mma0(v_q, v_k, v_s_next);
        if constexpr (decltype(mask_next)::value) {
            attn_mask_oob_score<T>(v_s_next, valid_kv_len, tile + 1, wave_kv_base, lane_id);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier_wait(-1);

        // (e) gather P(t)
        gather_p<T>(s_p, v_p, lane_id);
        v_s = v_s_next;

        // (a) publish max(t+1), read V(t)
        ml_publish<T>(s_m, attn_row_max_blocks<T>(v_s), warp_id, lane_id);
        v_v = tr_load<T::VEC_KV>(s_v, u_rv);
        advance_kv_slot<T>(s_v, v_off);
        s_wait_dscnt(number<T::v_ds_load_insts>{});
        __builtin_amdgcn_s_barrier_signal(-1);

        // (b) PV(t)
        v_o = mma1(v_p, v_v, v_o);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier_wait(-1);
    };

    int tile = 0;
    for (; tile + 1 < num_kv_tiles - 1; ++tile) {
        s_wait_tensorcnt(0_I);
        issue_next_gather();
        round(tile, false_type{});
    }

    #pragma clang loop unroll(disable)
    for (; tile < num_kv_tiles; ++tile) {
        s_wait_tensorcnt(0_I);
        issue_next_gather();
        round(tile, true_type{});
    }
}

} // namespace opus_mla_v4_prefill_a16w16_32mx1_16nx4

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void opus_mla_v4_prefill_a16w16_32mx1_16nx4_kernel(opus_mla_v4_prefill_kargs kargs) {
    using namespace opus;
    using namespace opus_mla_v4_prefill_a16w16_32mx1_16nx4;
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

    constexpr index_t o_len_per_m = T::Q_TILE_SIZE * T::D_TILE_SIZE / (T::T_N * T::WARP_SIZE) / T::GEMM1_E_M;
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
    vector_t<D_ACC, T::GEMM0_E_M> m_row, l_part;
    static_for<T::GEMM0_E_M>([&](auto i) {
        m_row[i.value] = opus::numeric_limits<D_ACC>::lowest();
        l_part[i.value] = D_ACC(0.0f);
    });

    // Prefix segment: indices point into unified_kv[total_pages]
    {
        const int page_idx_begin = kargs.kv_indptr_prefix[q_token_idx];
        const int valid_kv_len   = kargs.kv_indptr_prefix[q_token_idx + 1] - page_idx_begin;
        const int num_kv_tiles   = ceil_div(valid_kv_len, T::KV_TILE_SIZE);
        mla_v4_prefill_accum_pipelined<Traits>(kargs, kargs.unified_kv_ptr, kargs.total_pages,
                                           kargs.kv_indices_prefix, page_idx_begin, valid_kv_len, num_kv_tiles,
                                           smem_buf, v_q, v_o, m_row, l_part, temperature_scale);
    }

    // Extend segment: indices point into kv[total_tokens]
    {
        const int page_idx_begin = kargs.kv_indptr_extend[q_token_idx];
        const int valid_kv_len   = kargs.kv_indptr_extend[q_token_idx + 1] - page_idx_begin;
        const int num_kv_tiles   = ceil_div(valid_kv_len, T::KV_TILE_SIZE);
        mla_v4_prefill_accum_pipelined<Traits>(kargs, kargs.kv_ptr, kargs.total_tokens,
                                           kargs.kv_indices_extend, page_idx_begin, valid_kv_len, num_kv_tiles,
                                           smem_buf, v_q, v_o, m_row, l_part, temperature_scale);
    }

    // l was only ever summed over each wave's own KV columns
    {
        auto s_l = make_smem(reinterpret_cast<D_ACC*>(smem_buf + T::L_LDS_OFF));
        ml_publish<T>(s_l, l_part, warp_id, lane_id);
        s_wait_dscnt(0_I);
        __builtin_amdgcn_s_barrier();
        l_part = ml_reduce<T>(s_l, lane_id, [](D_ACC x, D_ACC y) { return x + y; });
    }

    // Sink finalization, normalize O.
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
