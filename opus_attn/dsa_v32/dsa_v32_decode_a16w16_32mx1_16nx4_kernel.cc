#include <opus/hip_minimal.hpp>
#include "defs.h"

using dsa_v32_traits_t = dsa_v32_decode_a16w16_32mx1_16nx4_traits<32, 64, 4, bf16_t, bf16_t>;

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void dsa_v32_decode_a16w16_32mx1_16nx4_kernel(dsa_v32_a16w16_kargs kargs) {}
template __global__ void dsa_v32_decode_a16w16_32mx1_16nx4_kernel<dsa_v32_traits_t>(dsa_v32_a16w16_kargs);
#else
#include "dsa_v32_decode_a16w16_32mx1_16nx4_template.hpp"
template __global__ void dsa_v32_decode_a16w16_32mx1_16nx4_kernel<dsa_v32_traits_t>(dsa_v32_a16w16_kargs);
#endif
