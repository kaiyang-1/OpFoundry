#include <opus/hip_minimal.hpp>
#include "defs.h"

using a8w8_traits_t          = opus_mla_decode_splitkv_a8w8_16mx8_32nx1_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>;
using a16w16_16mx4_traits_t  = opus_mla_decode_splitkv_a16w16_16mx4_64nx1_traits<16, 64, 4, bf16_t, bf16_t>;
using a16w16_32mx1_traits_t  = opus_mla_decode_splitkv_a16w16_32mx1_16nx4_traits<32, 64, 4, bf16_t, bf16_t>;

#define MLA_DECODE_SPLITKV_COMBINE_INSTANTIATIONS                                                      \
    template __global__ void mla_combine_kernel<a8w8_traits_t,         opus_mla_decode_splitkv_fp8_kargs>(opus_mla_decode_splitkv_fp8_kargs);      \
    template __global__ void mla_combine_kernel<a16w16_16mx4_traits_t, opus_mla_decode_splitkv_kargs>(opus_mla_decode_splitkv_kargs);  \
    template __global__ void mla_combine_kernel<a16w16_32mx1_traits_t, opus_mla_decode_splitkv_kargs>(opus_mla_decode_splitkv_kargs);

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits, typename KArgs, int HEADS_PER_BLOCK = 8> __global__ void mla_combine_kernel(KArgs kargs) {}
MLA_DECODE_SPLITKV_COMBINE_INSTANTIATIONS
#else
#include "mla_combine.hpp"
MLA_DECODE_SPLITKV_COMBINE_INSTANTIATIONS
#endif
