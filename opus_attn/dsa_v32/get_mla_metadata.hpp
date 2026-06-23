#pragma once

#include <opus/opus.hpp>
#include "defs.h"

template<class Traits>
__global__ void get_mla_metadata_kernel(dsa_v32_fp8_kargs kargs) {
    using T = opus::remove_cvref_t<Traits>;
    if (opus::thread_id_x() != 0) return;

    const int B = kargs.B;
    const int num_parts = kargs.num_parts;

    auto num_tiles_of = [&](int b) -> int {
        const int len = kargs.kv_indptr[b + 1] - kargs.kv_indptr[b];
        const int nt = ceil_div(len, T::KV_TILE_SIZE);
        return nt < 1 ? 1 : nt;
    };

    long total = 0;
    for (int b = 0; b < B; ++b) total += num_tiles_of(b) + DSA_V32_FIXED_OVERHEAD;
    int payload = (int)((total + num_parts - 1) / num_parts) + DSA_V32_FIXED_OVERHEAD;
    if (payload < 1) payload = 1;

    DsaSchedMeta* meta = const_cast<DsaSchedMeta*>(kargs.sched_meta);
    int* num_splits = const_cast<int*>(kargs.num_splits);
    num_splits[0] = 0;

    int req = 0, tile = 0, split_idx = 0;
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
            const int nt = num_tiles_of(req);
            const int avail = nt - tile;
            const int cost = avail + DSA_V32_FIXED_OVERHEAD;

            if (remaining >= cost || is_last_part) {

                remaining -= cost;
                m.end_req_idx = req;
                m.end_tile_idx = nt;
                num_splits[req + 1] = num_splits[req] + split_idx + 1;
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
