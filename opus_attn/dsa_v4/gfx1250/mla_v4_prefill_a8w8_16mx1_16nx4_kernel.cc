// MLA-v4 prefill attention D=512 MXFP8 kernel instantiation (gfx1250 16mx1_16nx4_fp8 variant)
// Host pass: empty stub for __device_stub__ generation
// Device pass: includes full kernel template
#include <opus/hip_minimal.hpp>
#include "mla_v4_traits.h"

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void opus_mla_v4_prefill_a8w8_16mx1_16nx4_kernel(opus_mla_v4_prefill_fp8_kargs kargs) {}
#else
#include "mla_v4_prefill_a8w8_16mx1_16nx4_template.hpp"
#endif

template __global__ void opus_mla_v4_prefill_a8w8_16mx1_16nx4_kernel<opus_mla_v4_prefill_a8w8_16mx1_16nx4_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>>(opus_mla_v4_prefill_fp8_kargs);
