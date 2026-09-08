// MLA-v4 prefill attention D=512 fp8 kernel instantiation (16mx1_16nx4 layout)
// Host pass: empty stub for __device_stub__ generation
// Device pass: includes full kernel template
#include <opus/hip_minimal.hpp>
#include "mla_v4_traits.h"

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void opus_mla_v4_prefill_a8w8_16mx1_16nx4_kernel(opus_mla_v4_prefill_fp8_kargs kargs) {}
template __global__ void opus_mla_v4_prefill_a8w8_16mx1_16nx4_kernel<opus_mla_v4_prefill_a8w8_16mx1_16nx4_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>>(opus_mla_v4_prefill_fp8_kargs);
#else
#include "mla_v4_prefill_a8w8_16mx1_16nx4_template.hpp"
template __global__ void opus_mla_v4_prefill_a8w8_16mx1_16nx4_kernel<opus_mla_v4_prefill_a8w8_16mx1_16nx4_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>>(opus_mla_v4_prefill_fp8_kargs);
#endif
