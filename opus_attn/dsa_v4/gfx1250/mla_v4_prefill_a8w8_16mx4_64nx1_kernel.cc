// MLA-v4 prefill attention D=512 MXFP8 kernel instantiation (gfx1250 16mx4_64nx1_fp8 variant)
// Host pass: empty stub for __device_stub__ generation
// Device pass: includes full kernel template
#include <opus/hip_minimal.hpp>
#include "mla_v4_traits.h"

#define MLA_V4_FOR_EACH_CLUSTER_Y(X) X(1) X(2)

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void opus_mla_v4_prefill_a8w8_16mx4_64nx1_kernel(opus_mla_v4_prefill_fp8_kargs kargs) {}
#else
#include "mla_v4_prefill_a8w8_16mx4_64nx1_template.hpp"
#endif

#define MLA_V4_INSTANTIATE(CY)                                            \
    template __global__ void opus_mla_v4_prefill_a8w8_16mx4_64nx1_kernel< \
        opus_mla_v4_prefill_a8w8_16mx4_64nx1_traits<16, 64, 4, CY, fp8_t, bf16_t, bf16_t>>(opus_mla_v4_prefill_fp8_kargs);
MLA_V4_FOR_EACH_CLUSTER_Y(MLA_V4_INSTANTIATE)
#undef MLA_V4_INSTANTIATE
