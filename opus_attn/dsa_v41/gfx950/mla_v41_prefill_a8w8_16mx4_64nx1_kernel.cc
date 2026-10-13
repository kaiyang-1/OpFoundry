#include <opus/hip_minimal.hpp>
#include "common/mla_v41_defs.h"

using mla_v41_traits_t = opus_mla_v41_prefill_a8w8_16mx4_64nx1_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>;

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void opus_mla_v41_prefill_a8w8_16mx4_64nx1_kernel(opus_mla_v41_prefill_kargs kargs) {}
template __global__ void opus_mla_v41_prefill_a8w8_16mx4_64nx1_kernel<mla_v41_traits_t>(opus_mla_v41_prefill_kargs);
#else
#include "mla_v41_prefill_a8w8_16mx4_64nx1_template.hpp"
template __global__ void opus_mla_v41_prefill_a8w8_16mx4_64nx1_kernel<mla_v41_traits_t>(opus_mla_v41_prefill_kargs);
#endif
