// PA prefill attention D=512 MXFP8 kernel instantiation (gfx1250 16mx4_64nx1_fp8 variant)
// Host pass: empty stub for __device_stub__ generation
// Device pass: includes full kernel template
#include <opus/hip_minimal.hpp>
#include "pa_traits.h"

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void pa_prefill_16mx4_64nx1_fp8_kernel(pa_fp8_kargs kargs) {}
template __global__ void pa_prefill_16mx4_64nx1_fp8_kernel<pa_16mx4_64nx1_fp8_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>>(pa_fp8_kargs);
#else
#include "pa_prefill_16mx4_64nx1_fp8_template.hpp"
template __global__ void pa_prefill_16mx4_64nx1_fp8_kernel<pa_16mx4_64nx1_fp8_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>>(pa_fp8_kargs);
#endif
