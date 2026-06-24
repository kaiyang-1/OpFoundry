#pragma once

#include <opus/opus.hpp>
#include "defs.h"

template<class Traits>
__global__ void get_mla_metadata_kernel(dsa_v32_fp8_kargs kargs) {
    using T = opus::remove_cvref_t<Traits>;
    constexpr int WARP = T::WARP_SIZE;

    const int B = kargs.B;
    const int num_parts = kargs.num_parts;
    const int lane = opus::thread_id_x();

    extern __shared__ int smem[];
    int* nt_shared     = smem;
    int* nsplit_shared = smem + B;

    int local_total = 0;
    for (int b = lane; b < B; b += WARP) {
        const int len = kargs.kv_indptr[b + 1] - kargs.kv_indptr[b];
        int nt = ceil_div(len, T::KV_TILE_SIZE);
        if (nt < 1) nt = 1;
        nt_shared[b] = nt;
        local_total += nt + DSA_V32_FIXED_OVERHEAD;
    }

    int total = local_total;
    #pragma unroll
    for (int offset = WARP / 2; offset >= 1; offset /= 2)
        total += __builtin_amdgcn_ds_bpermute((lane ^ offset) << 2, total);

    __builtin_amdgcn_s_barrier();

    if (lane == 0) {
        int payload = (total + num_parts - 1) / num_parts + DSA_V32_FIXED_OVERHEAD;
        if (payload < 1) payload = 1;

        DsaSchedMeta* meta = const_cast<DsaSchedMeta*>(kargs.sched_meta);
        nsplit_shared[0] = 0;

        int req = 0, tile = 0, split_idx = 0, cum_splits = 0;
        for (int p = 0; p < num_parts; ++p) {
            DsaSchedMeta m;
            if (req >= B) {
                m.begin_req_idx = B; m.end_req_idx = B;
                m.begin_tile_idx = 0; m.end_tile_idx = 0; m.begin_split_idx = 0;
                meta[p] = m;
                continue;
            }

            m.begin_req_idx = req;
            m.begin_tile_idx = tile;
            m.begin_split_idx = split_idx;
            m.end_req_idx = req;
            m.end_tile_idx = tile;

            int remaining = payload;
            const bool is_last_part = (p == num_parts - 1);

            while (req < B) {
                const int nt = nt_shared[req];
                const int avail = nt - tile;
                const int cost = avail + DSA_V32_FIXED_OVERHEAD;

                if (remaining >= cost || is_last_part) {

                    remaining -= cost;
                    m.end_req_idx = req;
                    m.end_tile_idx = nt;
                    cum_splits += split_idx + 1;
                    nsplit_shared[req + 1] = cum_splits;
                    req++; tile = 0; split_idx = 0;

                } else {

                    const int consume = remaining - DSA_V32_FIXED_OVERHEAD;
                    if (consume > 0) {
                        tile += consume;
                        split_idx++;
                        m.end_req_idx = req;
                        m.end_tile_idx = tile;
                    }
                    break;
                }
            }
            meta[p] = m;
        }
    }

    __builtin_amdgcn_s_barrier();

    int* num_splits = const_cast<int*>(kargs.num_splits);
    for (int i = lane; i <= B; i += WARP)
        num_splits[i] = nsplit_shared[i];
}
