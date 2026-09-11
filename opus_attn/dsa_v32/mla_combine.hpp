#pragma once

#include <opus/opus.hpp>
#include "defs.h"

template<class T>
__device__ inline void mla_write_lse(const dsa_kargs& kargs, int b, int head, int lane,
                                     typename T::D_ACC m, typename T::D_ACC denom) {
    using D_ACC = typename T::D_ACC;
    if (lane != 0) return;
    D_ACC* lse = reinterpret_cast<D_ACC*>(kargs.lse_ptr);
    lse[b * kargs.stride_lse_b + head] = (denom > D_ACC(0.0f))
        ? (m + log2f(denom)) * D_ACC(DSA_V32_LN_2)
        : opus::numeric_limits<D_ACC>::infinity();
}

template<class T, int HEADS_PER_BLOCK>
__device__ void mla_combine_online(const dsa_kargs& kargs, int b, int head,
                                   int lane, int start, int ns) {
    using D_OUT = typename T::D_OUT;
    using D_ACC = typename T::D_ACC;
    using D_ACCx4 = opus::vector_t<D_ACC, 4>;
    constexpr int WARP = T::WARP_SIZE;
    constexpr int D    = T::D_NOPE_SIZE;
    constexpr int VEC  = 4;
    constexpr int NVEC = D / (WARP * VEC);

    const int H = kargs.H;
    const D_ACC* lse_accum = reinterpret_cast<const D_ACC*>(kargs.lse_accum);
    const D_ACC* o_accum   = reinterpret_cast<const D_ACC*>(kargs.o_accum);

    auto load_split = [&](int i, D_ACCx4* dst) {
        const D_ACC* p = o_accum + (size_t)((start + i) * H + head) * D + lane * VEC;
        #pragma unroll
        for (int v = 0; v < NVEC; ++v)
            dst[v] = *reinterpret_cast<const D_ACCx4*>(p + v * WARP * VEC);
    };
    auto load_lse = [&](int i) { return lse_accum[(start + i) * H + head]; };

    D_ACCx4 acc[NVEC];
    #pragma unroll
    for (int v = 0; v < NVEC; ++v) acc[v] = D_ACCx4{0.0f, 0.0f, 0.0f, 0.0f};
    D_ACC m = opus::numeric_limits<D_ACC>::lowest();
    D_ACC denom = 0.0f;

    D_ACCx4 cur[NVEC];
    load_split(0, cur);
    D_ACC lse_cur = load_lse(0);
    for (int i = 0; i < ns; ++i) {
        D_ACCx4 nxt[NVEC];
        D_ACC lse_nxt = 0.0f;
        if (i + 1 < ns) {
            load_split(i + 1, nxt);
            lse_nxt = load_lse(i + 1);
        }
        const D_ACC new_m   = opus::max(m, lse_cur);
        const D_ACC old_scl = __builtin_amdgcn_exp2f(m - new_m);
        const D_ACC cur_scl = (lse_cur > opus::numeric_limits<D_ACC>::lowest())
                                ? __builtin_amdgcn_exp2f(lse_cur - new_m) : D_ACC(0.0f);
        #pragma unroll
        for (int v = 0; v < NVEC; ++v) acc[v] = old_scl * acc[v] + cur_scl * cur[v];
        denom = denom * old_scl + cur_scl;
        m = new_m;
        #pragma unroll
        for (int v = 0; v < NVEC; ++v) cur[v] = nxt[v];
        lse_cur = lse_nxt;
    }

    mla_write_lse<T>(kargs, b, head, lane, m, denom);

    const D_ACC inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    #pragma unroll
    for (int v = 0; v < NVEC; ++v) acc[v] *= inv_denom;

    D_OUT* out = reinterpret_cast<D_OUT*>(kargs.out_ptr)
               + (size_t)b * kargs.stride_o_b + head * kargs.stride_o_h;
    #pragma unroll
    for (int v = 0; v < NVEC; ++v) {
        opus::vector_t<D_OUT, 4> ov;
        #pragma unroll
        for (int e = 0; e < VEC; ++e) ov[e] = static_cast<D_OUT>(acc[v][e]);
        *reinterpret_cast<opus::vector_t<D_OUT, 4>*>(out + lane * VEC + v * WARP * VEC) = ov;
    }
}

template<class T, int HEADS_PER_BLOCK>
__device__ void mla_combine_two_pass(const dsa_kargs& kargs, int b, int head,
                                     int lane, int start, int ns) {
    using D_OUT = typename T::D_OUT;
    using D_ACC = typename T::D_ACC;
    using D_ACCx4 = opus::vector_t<D_ACC, 4>;
    constexpr int WARP = T::WARP_SIZE;
    constexpr int D    = T::D_NOPE_SIZE;
    constexpr int VEC  = 4;
    constexpr int NVEC = D / (WARP * VEC);
    constexpr int MAX_SPLITS = DSA_V32_NUM_CU;
    constexpr int NCHUNK = (MAX_SPLITS + WARP - 1) / WARP;

    const int H = kargs.H;
    const D_ACC* lse_accum = reinterpret_cast<const D_ACC*>(kargs.lse_accum);
    const D_ACC* o_accum   = reinterpret_cast<const D_ACC*>(kargs.o_accum);

    D_ACC local_max = opus::numeric_limits<D_ACC>::lowest();
    for (int s = lane; s < ns; s += WARP)
        local_max = opus::max(local_max, lse_accum[(start + s) * H + head]);
    #pragma unroll
    for (int off = WARP / 2; off >= 1; off >>= 1)
        local_max = opus::max(local_max, opus::shfl(local_max, lane ^ off));
    const D_ACC m = local_max;

    D_ACC scale[NCHUNK];
    D_ACC local_sum = 0.0f;
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
        const int s = lane + c * WARP;
        const D_ACC lse_s = (s < ns) ? lse_accum[(start + s) * H + head]
                                     : opus::numeric_limits<D_ACC>::lowest();
        const D_ACC e = (lse_s > opus::numeric_limits<D_ACC>::lowest())
                          ? __builtin_amdgcn_exp2f(lse_s - m) : D_ACC(0.0f);
        scale[c] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int off = WARP / 2; off >= 1; off >>= 1)
        local_sum += opus::shfl(local_sum, lane ^ off);

    mla_write_lse<T>(kargs, b, head, lane, m, local_sum);

    const D_ACC inv_denom = (local_sum > 0.0f) ? (1.0f / local_sum) : 0.0f;

    D_ACCx4 acc[NVEC];
    #pragma unroll
    for (int v = 0; v < NVEC; ++v) acc[v] = D_ACCx4{0.0f, 0.0f, 0.0f, 0.0f};

    auto load_split = [&](int i, D_ACCx4* dst) {
        const D_ACC* p = o_accum + (size_t)((start + i) * H + head) * D + lane * VEC;
        #pragma unroll
        for (int v = 0; v < NVEC; ++v)
            dst[v] = *reinterpret_cast<const D_ACCx4*>(p + v * WARP * VEC);
    };

    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
        const int base = c * WARP;
        if (base >= ns) break;
        for (int j = 0; j < WARP; ++j) {
            const int i = base + j;
            if (i >= ns) break;
            D_ACCx4 cur[NVEC];
            load_split(i, cur);
            const D_ACC w = opus::shfl(scale[c], j);
            #pragma unroll
            for (int v = 0; v < NVEC; ++v) acc[v] += w * cur[v];
        }
    }

    D_OUT* out = reinterpret_cast<D_OUT*>(kargs.out_ptr)
               + (size_t)b * kargs.stride_o_b + head * kargs.stride_o_h;
    #pragma unroll
    for (int v = 0; v < NVEC; ++v) {
        opus::vector_t<D_OUT, 4> ov;
        #pragma unroll
        for (int e = 0; e < VEC; ++e) ov[e] = static_cast<D_OUT>(acc[v][e] * inv_denom);
        *reinterpret_cast<opus::vector_t<D_OUT, 4>*>(out + lane * VEC + v * WARP * VEC) = ov;
    }
}

template<class Traits, int HEADS_PER_BLOCK = 8>
__global__ void mla_combine_kernel(dsa_kargs kargs) {
    using T = opus::remove_cvref_t<Traits>;
    constexpr int WARP = T::WARP_SIZE;
    constexpr int ONLINE_MAX_NS = 4;

    const int warp = opus::thread_id_x() / WARP;
    const int lane = opus::thread_id_x() % WARP;
    const int b    = opus::block_id_x();
    const int head = opus::block_id_y() * HEADS_PER_BLOCK + warp;
    if (head >= kargs.H) return;

    const int start = kargs.num_splits[b];
    const int ns    = kargs.num_splits[b + 1] - start;
    if (ns <= 1) return;

    if (ns <= ONLINE_MAX_NS)
        mla_combine_online<T, HEADS_PER_BLOCK>(kargs, b, head, lane, start, ns);
    else
        mla_combine_two_pass<T, HEADS_PER_BLOCK>(kargs, b, head, lane, start, ns);
}
