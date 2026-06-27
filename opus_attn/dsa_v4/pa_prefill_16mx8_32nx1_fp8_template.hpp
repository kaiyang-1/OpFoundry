#include <opus/opus.hpp>
#include "pa_defs.h"
#include <bit>
#include <cstdint>

using opus::operator""_I;

namespace pa_16mx8_32nx1_fp8 {

template<class Traits, class VQN, class VQR, class VO>
__device__ void pa_prefill_16mx8_32nx1_fp8_pipeline(
        pa_fp8_kargs kargs, const void* kv_nope_ptr, const void* kv_rope_ptr,
        int kv_rows, const int* kv_indices,
        int page_idx_begin, int valid_kv_len, int num_kv_tiles,
        char* smem_kv, char* smem_ml, char* smem_p,
        VQN& v_q_nope, VQR& v_q_rope, int scale_q, VO& v_o,
        typename Traits::D_ACC& m_row, typename Traits::D_ACC& l_row,
        float temperature_scale) {
    
}

} // namespace pa_16mx8_32nx1_fp8


template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void pa_prefill_16mx8_32nx1_fp8_kernel(pa_fp8_kargs kargs) {
    
}
