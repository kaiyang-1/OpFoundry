#pragma once

#include <opus/opus.hpp>
#include "pa_traits.h"
#include <cstdint>
#include <bit>

using opus::operator""_I;

namespace pa_16mx1_16nx4 {

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
__device__ inline typename T::D_ACC attn_row_sum(const V& v_s) {
    using D_ACC = typename T::D_ACC;
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    D_ACC row_sum = tree_reduce<0, s_len>(v_s, [](D_ACC x, D_ACC y) { return x + y; });

    int res16 = __builtin_amdgcn_permlane_xor(std::bit_cast<int>(row_sum), 16, 32);
    return row_sum + std::bit_cast<float>(res16);
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

template<typename T, typename S>
__device__ inline void ml_arrive(S& s_ml, typename T::D_ACC v, int warp_id, int lane_id) {
    opus::store(s_ml, v, (lane_id % T::W_M) * T::T_N + warp_id);
    opus::s_wait_dscnt(opus::number<0>{});
    __builtin_amdgcn_s_barrier_signal(-1);
}

template<typename T, typename S, typename Op>
__device__ inline typename T::D_ACC ml_reduce(S& s_ml, int lane_id, Op op) {
    __builtin_amdgcn_s_barrier_wait(-1);
    auto parts = opus::load<T::T_N>(s_ml, (lane_id % T::W_M) * T::T_N);
    return tree_reduce<0, T::T_N>(parts, op);
}

template<typename T, typename V>
__device__ inline void scale_output_tile(V& v_o, typename T::D_ACC scale) {
    constexpr opus::index_t o_len = opus::vector_traits<V>::size();
    opus::static_for<o_len>([&](auto i) { v_o[i.value] *= scale; });
}

template<typename T, typename V>
__device__ inline void attn_mask_oob_score(V& v_s, int valid_kv_len, int kv_tile_idx, int wave_kv_base) {
    using D_ACC = typename T::D_ACC;

    if ((kv_tile_idx + 1) * T::KV_TILE_SIZE <= valid_kv_len) return;

    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    constexpr int lane_hi_kv_step = T::W_N / (T::WARP_SIZE / T::W_M);
    static_assert(s_len == (T::W_M * T::W_N) / T::WARP_SIZE);

    const D_ACC neg_inf = -opus::numeric_limits<D_ACC>::infinity();
    const int lane_hi = (opus::thread_id_x() % T::WARP_SIZE) / T::W_M;
    const int rel = (valid_kv_len - 1) - kv_tile_idx * T::KV_TILE_SIZE - wave_kv_base - lane_hi * lane_hi_kv_step;

    opus::static_for<s_len>([&](auto i_reg) {
        v_s[i_reg.value] = (i_reg.value > rel) ? neg_inf : v_s[i_reg.value];
    });
}

template<class Traits>
__device__ __attribute__((always_inline)) void pa_prefill_accum_pipelined(
        pa_kargs kargs,
        const void* kv_ptr, int kv_rows,
        const int* kv_indices, int page_idx_begin, int valid_kv_len, int num_kv_tiles,
        char* smem_buf,
        opus::vector_t<typename Traits::D_ATTN, Traits::Q_TILE_SIZE * Traits::D_TILE_SIZE / Traits::WARP_SIZE>& v_q,
        opus::vector_t<typename Traits::D_ACC, Traits::Q_TILE_SIZE * Traits::D_TILE_SIZE / (Traits::T_N * Traits::WARP_SIZE)>& v_o,
        typename Traits::D_ACC& m_row,
        typename Traits::D_ACC& l_row,
        typename Traits::D_ACC temperature_scale) {
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC = typename T::D_ACC;

    if (num_kv_tiles <= 0) return;

    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    // Wave placement. wave 0/2 -> segment 0, wave 1/3 -> segment 1; the slot picks which half of a
    // segment. KV rows: wave 0 -> 0..15, wave 2 -> 16..31, wave 1 -> 32..47, wave 3 -> 48..63.
    const int kv_seg  = warp_id & 1;
    const int kv_slot = warp_id >> 1;
    const int wave_lds_off = kv_seg * T::KV_SEG_BYTES + kv_slot * T::WAVE_LDS_BYTES;
    const int wave_kv_base = kv_seg * T::ROWS_PER_SEG + kv_slot * T::ROWS_PER_WAVE;
    const int p_block      = wave_kv_base / T::ROWS_PER_WAVE;

    auto s_k = make_smem(reinterpret_cast<D_ATTN*>(smem_buf + wave_lds_off));
    auto s_m = make_smem(reinterpret_cast<D_ACC*>(smem_buf + T::ML_LDS_OFF));
    auto s_l = make_smem(reinterpret_cast<D_ACC*>(smem_buf + T::ML_LDS_OFF) + T::T_N * T::W_M);
    auto s_p = make_smem(reinterpret_cast<D_ATTN*>(smem_buf + T::P_LDS_OFF));

    const int seg_rot_bytes = kv_seg * T::KV_SEG_BYTES;
    const int v_col_off     = warp_id * (T::D_TILE_SIZE / T::T_N);
    smem<D_ATTN> s_v[T::NUM_KV_SEGS] = {
        make_smem(reinterpret_cast<D_ATTN*>(smem_buf + seg_rot_bytes) + v_col_off),
        make_smem(reinterpret_cast<D_ATTN*>(smem_buf + (T::KV_SEG_BYTES - seg_rot_bytes)) + v_col_off),
    };

    const u32x4_t kv_indices_rsrc = make_buffer_rsrc_raw(kv_indices + page_idx_begin, (u32_t)(valid_kv_len * sizeof(int)));

    using kv_window = tdm<D_ATTN, seq<T::D_TILE_SIZE, T::INDICES_PER_TDM>,
                          tdm_traits::gather<32>,
                          tdm_traits::padding_auto<D_ATTN, T::D_TILE_SIZE>>;

    auto tdm_kv = make_tdm<kv_window>((u32_t)reinterpret_cast<uintptr_t>(smem_buf + wave_lds_off),
                                      reinterpret_cast<const D_ATTN*>(kv_ptr),
                                      /*shape0=*/ (u32_t)T::D_TILE_SIZE,
                                      /*shape1=*/ (u32_t)kv_rows,
                                      /*stride=*/ (u64_t)kargs.stride_kv_page);

    if constexpr (T::CLUSTER_Y > 1) {
        tdm_kv.set_workgroup_mask(tdm_traits::peers_along_y<1, T::CLUSTER_Y>());
    }

    auto load_row_ids = [&](int tile_idx) {
        const int idx_byte_off = (tile_idx * T::KV_TILE_SIZE + wave_kv_base) * (int)sizeof(int);
        return s_buffer_load_b512(kv_indices_rsrc, idx_byte_off);
    };

    auto issue_kv_tile = [&](const u32x16_t& ids, int tile_idx, auto clamp_tail) {
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
            tdm_kv.async_load(u32_t(ld * LOAD_ELEMS));
        });
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

    constexpr index_t s_len = T::Q_TILE_SIZE * T::KV_TILE_SIZE / (T::T_N * T::WARP_SIZE);
    vector_t<D_ATTN, T::GEMM0_E_N * T::GEMM0_E_K * T::W_N * T::W_K / T::WARP_SIZE> v_k;
    vector_t<D_ACC,  s_len> v_s;
    typename decltype(mma1)::vtype_b v_v[2];

    vector_t<D_ATTN, T::NUM_WARPS * s_len> v_p;
    auto v_p_blocks = reinterpret_cast<vector_t<D_ATTN, s_len>*>(&v_p);
    auto v_p_stages = reinterpret_cast<typename decltype(mma1)::vtype_a*>(&v_p);

    u32x16_t row_ids = load_row_ids(0);

    for (int tile = 0; tile < num_kv_tiles; ++tile) {
        s_wait_kmcnt_for(row_ids);
        if (tile + 1 < num_kv_tiles) issue_kv_tile(row_ids, tile, false_type{});
        else                         issue_kv_tile(row_ids, tile, true_type{});
        row_ids = load_row_ids(tile + 1);

        s_wait_tensorcnt(0_I);
        v_k = load<T::VEC_KV>(s_k, u_rk);

        clear(v_s);
        v_s = mma0(v_q, v_k, v_s);
        attn_mask_oob_score<T>(v_s, valid_kv_len, tile, wave_kv_base);

        ml_arrive<T>(s_m, attn_row_max<T>(v_s), warp_id, lane_id);
        D_ACC row_max = ml_reduce<T>(s_m, lane_id, [](D_ACC x, D_ACC y) { return max(x, y); });

        row_max = max(m_row, row_max * temperature_scale);
        const D_ACC rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
        m_row = row_max;

        attn_row_scale_sub<T>(v_s, temperature_scale, row_max);
        attn_exp2_slice<T, 0, s_len>(v_s);

        auto v_p_own = cast<D_ATTN>(v_s);
        store<s_len>(s_p, v_p_own, p_block * T::P_BLOCK_ELEMS + lane_id * s_len);

        ml_arrive<T>(s_l, attn_row_sum<T>(v_s), warp_id, lane_id);
        l_row *= rescale_m;
        scale_output_tile<T>(v_o, rescale_m);
        l_row += ml_reduce<T>(s_l, lane_id, [](D_ACC x, D_ACC y) { return x + y; });

        const int p_rot = kv_seg * (T::NUM_WARPS / T::NUM_KV_SEGS);
        static_for<T::NUM_WARPS>([&](auto i) {
            const int src = (i.value + p_rot) & (T::NUM_WARPS - 1);
            v_p_blocks[i.value] = load<s_len>(s_p, src * T::P_BLOCK_ELEMS + lane_id * s_len);
        });

        v_v[0] = tr_load<T::VEC_KV>(s_v[0], u_rv);
        static_for<T::GEMM1_STAGE_K>([&](auto k) {
            constexpr int buf = k.value & 1;
            if constexpr (k.value + 1 < T::GEMM1_STAGE_K) {
                v_v[buf ^ 1] = tr_load<T::VEC_KV>(s_v[k.value + 1], u_rv);
            }
            v_o = mma1(v_p_stages[k.value], v_v[buf], v_o);
        });

        __builtin_amdgcn_s_barrier();
    }
}

} // namespace pa_16mx1_16nx4

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void pa_prefill_16mx1_16nx4_kernel(pa_kargs kargs) {
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
