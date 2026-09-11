#include <opus/hip_minimal.hpp>
#include "defs.h"

using a8w8_traits_t          = dsa_v32_decode_a8w8_16mx8_32nx1_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>;
using a16w16_16mx4_traits_t  = dsa_v32_decode_a16w16_16mx4_64nx1_traits<16, 64, 4, bf16_t, bf16_t>;
using a16w16_32mx1_traits_t  = dsa_v32_decode_a16w16_32mx1_16nx4_traits<32, 64, 4, bf16_t, bf16_t>;

#define DSA_V32_SCHED_INSTANTIATIONS                                                        \
    template __global__ void get_mla_metadata_kernel<a8w8_traits_t,         dsa_v32_a8w8_kargs>(dsa_v32_a8w8_kargs);      \
    template __global__ void get_mla_metadata_kernel<a16w16_16mx4_traits_t, dsa_v32_a16w16_kargs>(dsa_v32_a16w16_kargs);  \
    template __global__ void get_mla_metadata_kernel<a16w16_32mx1_traits_t, dsa_v32_a16w16_kargs>(dsa_v32_a16w16_kargs);

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits, typename KArgs> __global__ void get_mla_metadata_kernel(KArgs kargs) {}
DSA_V32_SCHED_INSTANTIATIONS
#else
#include "get_mla_metadata.hpp"
DSA_V32_SCHED_INSTANTIATIONS
#endif
