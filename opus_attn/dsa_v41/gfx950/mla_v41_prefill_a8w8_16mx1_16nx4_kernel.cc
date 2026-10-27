#include <opus/hip_minimal.hpp>
#include "common/mla_v41_defs.h"

using mla_v41_traits_t = opus_mla_v41_prefill_a8w8_16mx1_16nx4_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>;
using mla_v41_split_traits_t = opus_mla_v41_prefill_a8w8_16mx1_16nx4_traits<16, 64, 4, fp8_t, bf16_t, bf16_t, true>;

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void opus_mla_v41_prefill_a8w8_16mx1_16nx4_kernel(typename Traits::kargs_type kargs) {}
#else
#include "mla_v41_prefill_a8w8_16mx1_16nx4_template.hpp"
#endif

template __global__ void opus_mla_v41_prefill_a8w8_16mx1_16nx4_kernel<mla_v41_traits_t>(opus_mla_v41_prefill_kargs);
template __global__ void opus_mla_v41_prefill_a8w8_16mx1_16nx4_kernel<mla_v41_split_traits_t>(opus_mla_prefill_split_kargs<opus_mla_v41_prefill_kargs>);
