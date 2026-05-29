// PA prefill attention D=512 bf16 kernel instantiation (16mx1_16nx4 variant)
// Host pass: empty stub for __device_stub__ generation
// Device pass: includes full kernel template
#include <opus/hip_minimal.hpp>
#include "pa_defs.h"

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void pa_prefill_16mx1_16nx4_kernel(pa_kargs kargs) {}
template __global__ void pa_prefill_16mx1_16nx4_kernel<pa_traits_16mx1_16nx4<16, 64, 512, 4, bf16_t>>(pa_kargs);
#else
#include "pa_prefill_16mx1_16nx4_template.hpp"
template __global__ void pa_prefill_16mx1_16nx4_kernel<pa_traits_16mx1_16nx4<16, 64, 512, 4, bf16_t>>(pa_kargs);
#endif
