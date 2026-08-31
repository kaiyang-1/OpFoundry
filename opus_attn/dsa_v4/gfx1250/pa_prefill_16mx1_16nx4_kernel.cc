// PA prefill attention D=512 bf16 kernel instantiation (gfx1250 16mx1_16nx4 variant)
// Host pass: empty stub for __device_stub__ generation
// Device pass: includes full kernel template
#include <opus/hip_minimal.hpp>
#include "pa_traits.h"

#define PA_FOR_EACH_CLUSTER_Y(X) X(1) X(2)

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void pa_prefill_16mx1_16nx4_kernel(pa_kargs kargs) {}
#else
#include "pa_prefill_16mx1_16nx4_template.hpp"
#endif

#define PA_INSTANTIATE(CY) \
    template __global__ void pa_prefill_16mx1_16nx4_kernel< \
        pa_16mx1_16nx4_traits<16, 64, 512, 4, CY, bf16_t, bf16_t>>(pa_kargs);
PA_FOR_EACH_CLUSTER_Y(PA_INSTANTIATE)
#undef PA_INSTANTIATE
