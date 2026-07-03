#include <opus/hip_minimal.hpp>
#include "defs.h"

using dsa_v32_traits_t = dsa_v32_16mx8_32nx1_fp8_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>;

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void dsa_v32_decode_16mx8_32nx1_fp8_kernel(dsa_kargs kargs) {}
template __global__ void dsa_v32_decode_16mx8_32nx1_fp8_kernel<dsa_v32_traits_t>(dsa_kargs);
#else
#include "dsa_v32_splitkv.hpp"
template __global__ void dsa_v32_decode_16mx8_32nx1_fp8_kernel<dsa_v32_traits_t>(dsa_kargs);
#endif
