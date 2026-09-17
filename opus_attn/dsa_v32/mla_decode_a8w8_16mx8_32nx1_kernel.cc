#include <opus/hip_minimal.hpp>
#include "defs.h"

using mla_decode_traits_t = opus_mla_decode_a8w8_16mx8_32nx1_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>;

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void opus_mla_decode_a8w8_16mx8_32nx1_kernel(opus_mla_decode_fp8_kargs kargs) {}
template __global__ void opus_mla_decode_a8w8_16mx8_32nx1_kernel<mla_decode_traits_t>(opus_mla_decode_fp8_kargs);
#else
#include "mla_decode_a8w8_16mx8_32nx1_template.hpp"
template __global__ void opus_mla_decode_a8w8_16mx8_32nx1_kernel<mla_decode_traits_t>(opus_mla_decode_fp8_kargs);
#endif
