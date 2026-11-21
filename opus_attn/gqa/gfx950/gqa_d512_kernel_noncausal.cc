// GQA forward D=512 kernel (gfx950) — non-causal variant
// Host pass: empty stub for __device_stub__ generation
// Device pass: includes full kernel template
#include <opus/hip_minimal.hpp>
#include "gqa_traits.h"
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void gqa_d512_kernel(opus_gqa_varlen_kargs kargs) {}
#else
#include "gqa_d512_kernel_template.hpp"
#endif
template __global__ void gqa_d512_kernel<gqa_d512_traits<false>>(opus_gqa_varlen_kargs);
