#pragma once

#include <opus/opus.hpp>
#include "defs.h"

template<class Traits, int HEADS_PER_BLOCK = 8>
__global__ void mla_combine_kernel(dsa_v32_fp8_kargs kargs) {
    using T = opus::remove_cvref_t<Traits>;
    using D_OUT = typename T::D_OUT;
    using D_ACC = typename T::D_ACC;
    using f4    = opus::vector_t<D_ACC, 4>;
    constexpr int WARP = T::WARP_SIZE;
    constexpr int D    = T::D_NOPE_SIZE;
    constexpr int VEC  = 4;
    constexpr int NVEC = D / (WARP * VEC);

    const int b    = opus::block_id_x();
    const int warp = opus::thread_id_x() / WARP;
    const int lane = opus::thread_id_x() % WARP;
    const int head = opus::block_id_y() * HEADS_PER_BLOCK + warp;
    if (head >= kargs.H) return;

    const int start = kargs.num_splits[b];
    const int end   = kargs.num_splits[b + 1];
    const int ns    = end - start;
    if (ns <= 1) return;

    const int H = kargs.H;
    const D_ACC* lse_accum = reinterpret_cast<const D_ACC*>(kargs.lse_accum);
    const D_ACC* o_accum   = reinterpret_cast<const D_ACC*>(kargs.o_accum);

    D_ACC local_max = -3.0e38f;
    for (int s = lane; s < ns; s += WARP)
        local_max = opus::max(local_max, lse_accum[(start + s) * H + head]);
    #pragma unroll
    for (int off = WARP / 2; off >= 1; off >>= 1)
        local_max = opus::max(local_max, opus::shfl(local_max, lane ^ off));
    const D_ACC m = local_max;

    D_ACC local_sum = 0.0f;
    if (m > -3.0e38f)
        for (int s = lane; s < ns; s += WARP)
            local_sum += __builtin_amdgcn_exp2f(lse_accum[(start + s) * H + head] - m);
    #pragma unroll
    for (int off = WARP / 2; off >= 1; off >>= 1)
        local_sum += opus::shfl(local_sum, lane ^ off);
    const D_ACC inv_denom = (local_sum > 0.0f) ? (1.0f / local_sum) : 0.0f;

    f4 acc[NVEC];
    #pragma unroll
    for (int v = 0; v < NVEC; ++v) acc[v] = f4{0.0f, 0.0f, 0.0f, 0.0f};

    auto load_split = [&](int i, f4* dst) {
        const D_ACC* p = o_accum + (size_t)((start + i) * H + head) * D + lane * VEC;
        #pragma unroll
        for (int v = 0; v < NVEC; ++v)
            dst[v] = *reinterpret_cast<const f4*>(p + v * WARP * VEC);
    };
    auto weight = [&](int i) {
        return __builtin_amdgcn_exp2f(lse_accum[(start + i) * H + head] - m) * inv_denom;
    };

    f4 cur[NVEC];
    load_split(0, cur);
    D_ACC w = weight(0);
    for (int i = 0; i < ns; ++i) {
        f4 nxt[NVEC];
        D_ACC wn = 0.0f;
        if (i + 1 < ns) {
            load_split(i + 1, nxt);
            wn = weight(i + 1);
        }
        #pragma unroll
        for (int v = 0; v < NVEC; ++v) acc[v] += w * cur[v];
        #pragma unroll
        for (int v = 0; v < NVEC; ++v) cur[v] = nxt[v];
        w = wn;
    }

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
