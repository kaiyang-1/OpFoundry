#pragma once

#include <opus/opus.hpp>
#include "defs.h"

template<class Traits, int HEADS_PER_BLOCK = 8>
__global__ void mla_combine_kernel(dsa_v32_fp8_kargs kargs) {
    using T = opus::remove_cvref_t<Traits>;
    using D_OUT = typename T::D_OUT;
    using D_ACC = typename T::D_ACC;
    constexpr int WARP = T::WARP_SIZE;
    constexpr int D = T::D_NOPE_SIZE;
    constexpr int ELEMS = D / WARP;

    const int b = opus::block_id_x();
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

    D_ACC m = -3.0e38f;
    for (int i = start; i < end; ++i)
        m = fmaxf(m, lse_accum[i * H + head]);

    D_ACC denom = 0.0f;
    if (m > -3.0e38f)
        for (int i = start; i < end; ++i)
            denom += exp2f(lse_accum[i * H + head] - m);

    D_ACC acc[ELEMS];
    #pragma unroll
    for (int k = 0; k < ELEMS; ++k) acc[k] = 0.0f;

    if (denom > 0.0f) {
        const D_ACC inv_denom = 1.0f / denom;
        for (int i = start; i < end; ++i) {
            const D_ACC w = exp2f(lse_accum[i * H + head] - m) * inv_denom;
            const D_ACC* op = o_accum + (size_t)(i * H + head) * D;
            #pragma unroll
            for (int k = 0; k < ELEMS; ++k) acc[k] += w * op[lane + k * WARP];
        }
    }

    D_OUT* out = reinterpret_cast<D_OUT*>(kargs.out_ptr) + (size_t)b * kargs.stride_o_b + head * kargs.stride_o_h;
    #pragma unroll
    for (int k = 0; k < ELEMS; ++k) out[lane + k * WARP] = static_cast<D_OUT>(acc[k]);
}
