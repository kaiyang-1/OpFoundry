// PA prefill attention D=512 fp8 kernel instantiation (16mx8_32nx1 layout)
// Host pass: empty stub for __device_stub__ generation
// Device pass: includes full kernel template
#include <opus/hip_minimal.hpp>
#include "pa_traits.h"

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void pa_prefill_16mx8_32nx1_fp8_kernel(pa_fp8_kargs kargs) {}
template __global__ void pa_prefill_16mx8_32nx1_fp8_kernel<pa_16mx8_32nx1_fp8_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>>(pa_fp8_kargs);
#else
#include "pa_prefill_16mx8_32nx1_fp8_template.hpp"
template __global__ void pa_prefill_16mx8_32nx1_fp8_kernel<pa_16mx8_32nx1_fp8_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>>(pa_fp8_kargs);
#endif
