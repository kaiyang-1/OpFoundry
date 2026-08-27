// PA prefill attention D=512 MXFP8 kernel instantiation (gfx1250 16mx4_64nx1_fp8 variant)
// Host pass: empty stub for __device_stub__ generation
// Device pass: includes full kernel template
#include <opus/hip_minimal.hpp>
#include "pa_traits.h"

#define PA_FOR_EACH_CLUSTER_Y(X) X(1) X(2)

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void pa_prefill_16mx4_64nx1_fp8_kernel(pa_fp8_kargs kargs) {}
#else
#include "pa_prefill_16mx4_64nx1_fp8_template.hpp"
#endif

#define PA_INSTANTIATE(CY) \
    template __global__ void pa_prefill_16mx4_64nx1_fp8_kernel< \
        pa_16mx4_64nx1_fp8_traits<16, 64, 4, CY, fp8_t, bf16_t, bf16_t>>(pa_fp8_kargs);
PA_FOR_EACH_CLUSTER_Y(PA_INSTANTIATE)
#undef PA_INSTANTIATE
