#pragma once

#include <cfloat>
#include <cstdint>

// Split-K partials, same layout as aiter's mla_prefill_split.h
struct opus_mla_prefill_split_args {
    float* __restrict__ partial_o;   // [N, num_splits, H, 512]
    float* __restrict__ partial_max; // [N, num_splits, H]
    float* __restrict__ partial_sum; // [N, num_splits, H]
    int num_splits;
};

template<class Kargs>
struct opus_mla_prefill_split_kargs {
    Kargs base;
    opus_mla_prefill_split_args split;
};

template<class Kargs>
__host__ __device__ inline const Kargs& opus_mla_prefill_base(const Kargs& kargs) {
    return kargs;
}

template<class Kargs>
__host__ __device__ inline const Kargs& opus_mla_prefill_base(const opus_mla_prefill_split_kargs<Kargs>& kargs) {
    return kargs.base;
}

struct opus_mla_prefill_split_rows {
    int prefix_begin, prefix_len, extend_begin, extend_len;
};

// A split's share of the token's prefix rows, then extend rows, in whole KV tiles
template<int KV_TILE>
__device__ inline opus_mla_prefill_split_rows opus_mla_prefill_split_rows_of(int prefix_len, int extend_len, int num_splits, int split) {
    auto clamp = [](int x, int hi) { return x < 0 ? 0 : (x > hi ? hi : x); };
    const int chunk = ((prefix_len + extend_len + num_splits - 1) / num_splits + KV_TILE - 1) / KV_TILE * KV_TILE;
    const int lo = split * chunk;
    const int hi = lo + chunk;
    const int pb = clamp(lo, prefix_len);
    const int eb = clamp(lo - prefix_len, extend_len);
    return {pb, clamp(hi, prefix_len) - pb, eb, clamp(hi - prefix_len, extend_len) - eb};
}

__device__ inline int64_t opus_mla_prefill_split_row(const opus_mla_prefill_split_args& split, int token, int split_idx, int H, int head) {
    return (static_cast<int64_t>(token) * split.num_splits + split_idx) * H + head;
}

// m in log2 units, stored in natural ones; a split without keys stores -FLT_MAX
__device__ inline void opus_mla_prefill_split_store_ml(const opus_mla_prefill_split_args& split, int64_t row, float m, float l) {
    split.partial_max[row] = l > 0.0f ? m * 0.69314718056f : -FLT_MAX;
    split.partial_sum[row] = l;
}
