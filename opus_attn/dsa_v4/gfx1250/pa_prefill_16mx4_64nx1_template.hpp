// Paged sparse attention kernel template for D=512 on gfx1250 (wave32 / WMMA).
// 16mx4_64nx1 variant (T_M=4, T_N=1).
//
// Placeholder — no implementation yet.
//
// Include this header from per-variant .cc files that instantiate specific traits.
#pragma once

#include <opus/opus.hpp>
#include "pa_traits.h"
#include <cstdint>
#include <bit>

using opus::operator""_I;

namespace pa_16mx4_64nx1 {

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
    constexpr auto o_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::D_TILE_SIZE / T::W_N>{},
        opus::number<T::W_M * T::W_N / (T::WARP_SIZE * T::VEC_O)>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_O>{});

    constexpr auto o_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

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

    constexpr auto v_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_N>{},
        opus::number<T::WARP_SIZE / lane_per_grp>{},
        opus::number<lane_k>{},
        opus::number<lane_n>{},
        opus::number<T::VEC_KV>{});
    
    constexpr auto v_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    
    return opus::make_layout(
        v_block_shape,
        opus::unfold_x_stride(v_block_dim, v_block_shape, opus::tuple{opus::number<lane_n * T::VEC_KV>{}, opus::number<T::D_TILE_SIZE + T::KV_ROW_PAD_SIZE>{}, 1_I}),
        opus::unfold_p_coord(v_block_dim, opus::tuple{lane_id / lane_per_grp, (lane_id % lane_per_grp) % lane_k, (lane_id % lane_per_grp) / lane_k}));
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

template<int THR_X, int THR_Y>
__device__ inline void attn_mask_vec2_imm(opus::u32_t rel_vgpr, opus::u32_t neg_inf_vgpr,
                                          opus::u32_t& x_ref, opus::u32_t& y_ref) {
    uint32_t x_mask, y_mask;
    asm volatile(
        "v_cmp_lt_i32_e64 %0, %6, %7\n\t"
        "v_cmp_lt_i32_e64 %1, %6, %9\n\t"
        "v_cndmask_b32_e64 %2, %4, %8, %0\n\t"
        "v_cndmask_b32_e64 %3, %5, %8, %1\n\t"
        : "=s"(x_mask), "=s"(y_mask), "=v"(x_ref), "=v"(y_ref)
        : "v"(x_ref), "v"(y_ref), "v"(rel_vgpr),
          "n"(THR_X), "v"(neg_inf_vgpr), "n"(THR_Y)
        : "vcc"
    );
}

template<typename T, typename V>
__device__ inline void attn_mask_oob_score(V& v_s, int valid_kv_len, int kv_tile_idx, opus::u32_t neg_inf_v) {
    using D_ACC = typename T::D_ACC;
    using D_ACC_X2 = opus::vector_t<D_ACC, 2>;
    using U32_X2 = opus::vector_t<opus::u32_t, 2>;

    if ((kv_tile_idx + 1) * T::KV_TILE_SIZE <= valid_kv_len) return;

    constexpr int elems_per_stage = (T::W_M * T::W_N) / T::WARP_SIZE;      // 8
    constexpr int lane_hi_kv_step = T::W_N / (T::WARP_SIZE / T::W_M);      // 8

    const int last_valid_kv_pos = valid_kv_len - 1;
    const int k_start_pos = kv_tile_idx * T::KV_TILE_SIZE;
    int lane_id = opus::thread_id_x() % T::WARP_SIZE;
    const int lane_hi = lane_id / T::W_M;   // {0, 1} on wave32

    opus::static_for<T::GEMM0_STAGE_N>([&](auto i_s) {
        constexpr int stage = i_s.value;
        const int k_pos = k_start_pos + stage * T::W_N + lane_hi * lane_hi_kv_step;
        const opus::u32_t rel = static_cast<opus::u32_t>(last_valid_kv_pos - k_pos);

        opus::static_for<elems_per_stage / 2>([&](auto i_pair) {
            constexpr int reg0  = i_pair.value * 2;
            constexpr int idx   = stage * elems_per_stage + reg0;
            constexpr int thr_x = reg0;
            constexpr int thr_y = reg0 + 1;

            auto pair_acc  = opus::slice(v_s, opus::number<idx>{}, opus::number<idx + 2>{});
            auto pair_bits = __builtin_bit_cast(U32_X2, pair_acc);
            opus::u32_t x_ref = pair_bits[0];
            opus::u32_t y_ref = pair_bits[1];
            attn_mask_vec2_imm<thr_x, thr_y>(rel, neg_inf_v, x_ref, y_ref);
            pair_bits[0] = x_ref;
            pair_bits[1] = y_ref;
            opus::set_slice(v_s, __builtin_bit_cast(D_ACC_X2, pair_bits), opus::number<idx>{}, opus::number<idx + 2>{});
        });
    });
}

template<class Traits, int WARP>
__device__ void pa_prefill_accum_le2_tiles(pa_kargs kargs,
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

    int lane_id = thread_id_x() % T::WARP_SIZE;
    constexpr int warp_id = WARP;

    smem<D_ATTN> s_kv[T::NUM_WARPS] = {
        make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf + 0 * T::SEG_BYTES)),
        make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf + 1 * T::SEG_BYTES)),
        make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf + 2 * T::SEG_BYTES)),
        make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf + 3 * T::SEG_BYTES)),
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
        s_kv[warp_id].ptr,
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
    vector_t<D_ACC, T::Q_TILE_SIZE * T::KV_TILE_SIZE / T::WARP_SIZE> v_s;
    vector_t<D_ATTN, T::Q_TILE_SIZE * T::KV_TILE_SIZE / T::WARP_SIZE> v_p;
    auto v_s_stages = reinterpret_cast<vector_t<D_ACC, T::W_M * T::W_N / T::WARP_SIZE>*>(&v_s);
    auto v_p_stages = reinterpret_cast<vector_t<D_ATTN, T::W_M * T::W_K / T::WARP_SIZE>*>(&v_p);
    auto v_o_stages = reinterpret_cast<vector_t<D_ACC, T::W_M * T::D_TILE_SIZE / T::GEMM1_STAGE_N / T::WARP_SIZE>*>(&v_o);

    const u32_t neg_inf_v = std::bit_cast<u32_t>(-numeric_limits<D_ACC>::infinity());

    auto load_kv_tile = [&](int tile_idx) {
        constexpr int lds_step = T::INDICES_PER_TDM * T::KV_ROW_LDS_BYTES;

        const int idx_byte_off = (tile_idx * T::KV_TILE_SIZE + warp_id * T::ROWS_PER_WAVE) * (int)sizeof(int);
        const u32x16_t row_ids = llvm_amdgcn_s_buffer_load_v16i32(kv_indices_rsrc, idx_byte_off, /*aux=*/0);
        s_wait_kmcnt(0_I);

        static_for<T::TDM_LOADS_PER_WAVE>([&](auto d) {
            constexpr int ld = d.value;
            static_for<T::INDICES_PER_TDM>([&](auto r) {
                tdm_kv.set_gather_row_index(r.value, __builtin_amdgcn_readfirstlane(row_ids[ld * T::INDICES_PER_TDM + r.value]));
            });
            tdm_kv.load();
            if constexpr (ld + 1 < T::TDM_LOADS_PER_WAVE)
                tdm_kv.move(0_I, 0_I, 0_I, 0_I, 0_I, number<lds_step>{});
        });
        tdm_kv.move(0_I, 0_I, 0_I, 0_I, 0_I, number<-(T::TDM_LOADS_PER_WAVE - 1) * lds_step>{});
    };

    auto tr_load_v = [&](auto sn, auto sk, auto buf) {
        constexpr int stage_n = sn.value;
        constexpr int stage_k = sk.value;
        constexpr int b       = buf.value;

        auto v_v0 = tr_load<T::VEC_KV>(s_kv[2 * stage_k],     u_rv + number<stage_n * (T::D_TILE_SIZE / 4)>{});
        auto v_v1 = tr_load<T::VEC_KV>(s_kv[2 * stage_k + 1], u_rv + number<stage_n * (T::D_TILE_SIZE / 4)>{});

        constexpr int chunk  = T::VEC_KV;
        constexpr int groups = opus::vector_traits<decltype(v_v0)>::size() / chunk;
        static_for<groups>([&](auto g) {
            static_for<chunk>([&](auto i) {
                v_v[b][2 * chunk * g.value + i.value]         = v_v0[chunk * g.value + i.value];
                v_v[b][2 * chunk * g.value + chunk + i.value] = v_v1[chunk * g.value + i.value];
            });
        });
    };

    for (int tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        load_kv_tile(tile_idx);
        s_wait_tensorcnt(0_I);
        __builtin_amdgcn_s_barrier();

        constexpr int gemmk_perm[4] = {0, 2, 1, 3};
        v_k[0] = load<T::VEC_KV>(s_kv[WARP ^ gemmk_perm[0]], u_rk);
        static_for<T::GEMM0_STAGE_N>([&](auto j) {
            constexpr int stage = WARP ^ gemmk_perm[j.value];
            constexpr int cur   = j.value & 1;
            if constexpr (j.value + 1 < T::GEMM0_STAGE_N) {
                v_k[(j.value + 1) & 1] = load<T::VEC_KV>(s_kv[WARP ^ gemmk_perm[j.value + 1]], u_rk);
                s_wait_dscnt(number<T::k_ds_load_insts>{});
            } else {
                s_wait_dscnt(0_I);
            }
            __builtin_amdgcn_sched_barrier(0);
            clear(v_s_stages[stage]);
            v_s_stages[stage] = mma0(v_q, v_k[cur], v_s_stages[stage]);
        });

        attn_mask_oob_score<T>(v_s, valid_kv_len, tile_idx, neg_inf_v);
        D_ACC row_max = max(m_row, attn_row_max<T>(v_s) * temperature_scale);
        D_ACC rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
        m_row = row_max;
        attn_row_scale_sub<T>(v_s, temperature_scale, row_max);
        constexpr index_t s_len = vector_traits<decltype(v_s)>::size();
        attn_exp2_slice<T, 0, s_len>(v_s);
        l_row *= rescale_m;
        l_row += attn_row_sum<T>(v_s);
        v_p = cast<D_ATTN>(v_s);
        scale_output_tile<T>(v_o, rescale_m);

        constexpr int GEMM1_STAGES = T::GEMM1_STAGE_N * T::GEMM1_STAGE_K;
        tr_load_v(0_I, number<WARP & 1>{}, 0_I);
        static_for<GEMM1_STAGES>([&](auto s) {
            constexpr int stage_n = s.value % T::GEMM1_STAGE_N;
            constexpr int stage_k = WARP & 1 ? 1 - (s.value / T::GEMM1_STAGE_N) : s.value / T::GEMM1_STAGE_N;
            constexpr int cur     = s.value & 1;
            if constexpr (s.value + 1 < GEMM1_STAGES) {
                constexpr int n_sn = (s.value + 1) % T::GEMM1_STAGE_N;
                constexpr int n_sk = WARP & 1 ? 1 - ((s.value + 1) / T::GEMM1_STAGE_N) : (s.value + 1) / T::GEMM1_STAGE_N;
                tr_load_v(number<n_sn>{}, number<n_sk>{}, number<(s.value + 1) & 1>{});
                s_wait_dscnt(number<T::v_ds_load_insts>{});
            } else {
                s_wait_dscnt(0_I);
            }
            __builtin_amdgcn_sched_barrier(0);
            v_o_stages[stage_n] = mma1(v_p_stages[stage_k], v_v[cur], v_o_stages[stage_n]);
        });

        __builtin_amdgcn_s_barrier();
    }
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
    const int64_t qo_gmem_offset = (int64_t)q_token_idx * kargs.stride_qo_n + (int64_t)h_block_start * kargs.stride_qo_h;

    __shared__ char smem_kv_buf[T::smem_size_bytes()];

    // Load Q once (shared across both segments)
    auto g_q = make_gmem(reinterpret_cast<const D_ATTN*>(kargs.q_ptr) + qo_gmem_offset, (kargs.H - h_block_start) * kargs.stride_qo_h * sizeof(D_ATTN));
    auto u_q = make_layout_q<T>(warp_id, lane_id, kargs.stride_qo_h);

    vector_t<D_ATTN, T::Q_TILE_SIZE * T::D_TILE_SIZE / T::WARP_SIZE> v_q;
    vector_t<D_ACC,  T::Q_TILE_SIZE * T::D_TILE_SIZE / T::WARP_SIZE> v_o;

    constexpr D_ACC LOG2_E = 1.44269504089f;
    const D_ACC temperature_scale = kargs.softmax_scale * LOG2_E;

    v_q = load<T::VEC_Q>(g_q, u_q);
    s_wait_loadcnt(0_I);

    // Initialize shared attention state
    clear(v_o);
    D_ACC m_row = opus::numeric_limits<D_ACC>::lowest();
    D_ACC l_row = 0.0f;

    // ── Prefix segment: kv_indices_prefix index into unified_kv[total_pages] ──
    {
        const int page_idx_begin = kargs.kv_indptr_prefix[q_token_idx];
        const int valid_kv_len   = kargs.kv_indptr_prefix[q_token_idx + 1] - page_idx_begin;
        const int num_kv_tiles   = ceil_div(valid_kv_len, T::KV_TILE_SIZE);
        switch (warp_id & (T::NUM_WARPS - 1)) {
            case 0:  pa_prefill_accum_le2_tiles<Traits, 0>(kargs, kargs.unified_kv_ptr, kargs.total_pages, kargs.kv_indices_prefix, page_idx_begin, valid_kv_len, num_kv_tiles, smem_kv_buf, v_q, v_o, m_row, l_row, temperature_scale); break;
            case 1:  pa_prefill_accum_le2_tiles<Traits, 1>(kargs, kargs.unified_kv_ptr, kargs.total_pages, kargs.kv_indices_prefix, page_idx_begin, valid_kv_len, num_kv_tiles, smem_kv_buf, v_q, v_o, m_row, l_row, temperature_scale); break;
            case 2:  pa_prefill_accum_le2_tiles<Traits, 2>(kargs, kargs.unified_kv_ptr, kargs.total_pages, kargs.kv_indices_prefix, page_idx_begin, valid_kv_len, num_kv_tiles, smem_kv_buf, v_q, v_o, m_row, l_row, temperature_scale); break;
            default: pa_prefill_accum_le2_tiles<Traits, 3>(kargs, kargs.unified_kv_ptr, kargs.total_pages, kargs.kv_indices_prefix, page_idx_begin, valid_kv_len, num_kv_tiles, smem_kv_buf, v_q, v_o, m_row, l_row, temperature_scale); break;
        }
    }
    __builtin_amdgcn_s_barrier();

    // ── Extend segment: kv_indices_extend index into kv[total_tokens] ──
    {
        const int page_idx_begin = kargs.kv_indptr_extend[q_token_idx];
        const int valid_kv_len   = kargs.kv_indptr_extend[q_token_idx + 1] - page_idx_begin;
        const int num_kv_tiles   = ceil_div(valid_kv_len, T::KV_TILE_SIZE);
        switch (warp_id & (T::NUM_WARPS - 1)) {
            case 0:  pa_prefill_accum_le2_tiles<Traits, 0>(kargs, kargs.kv_ptr, kargs.total_tokens, kargs.kv_indices_extend, page_idx_begin, valid_kv_len, num_kv_tiles, smem_kv_buf, v_q, v_o, m_row, l_row, temperature_scale); break;
            case 1:  pa_prefill_accum_le2_tiles<Traits, 1>(kargs, kargs.kv_ptr, kargs.total_tokens, kargs.kv_indices_extend, page_idx_begin, valid_kv_len, num_kv_tiles, smem_kv_buf, v_q, v_o, m_row, l_row, temperature_scale); break;
            case 2:  pa_prefill_accum_le2_tiles<Traits, 2>(kargs, kargs.kv_ptr, kargs.total_tokens, kargs.kv_indices_extend, page_idx_begin, valid_kv_len, num_kv_tiles, smem_kv_buf, v_q, v_o, m_row, l_row, temperature_scale); break;
            default: pa_prefill_accum_le2_tiles<Traits, 3>(kargs, kargs.kv_ptr, kargs.total_tokens, kargs.kv_indices_extend, page_idx_begin, valid_kv_len, num_kv_tiles, smem_kv_buf, v_q, v_o, m_row, l_row, temperature_scale); break;
        }
    }

    // ──── Sink finalization, normalize O, and store to gmem ────
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