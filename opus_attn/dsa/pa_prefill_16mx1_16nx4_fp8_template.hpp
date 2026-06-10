#include <opus/opus.hpp>
#include "pa_defs.h"

using opus::operator""_I;

namespace pa_16mx1_16nx4_fp8 {

template<class Traits, class VQ, class VO>
__device__ void pa_prefill_16mx1_16nx4_fp8_pipeline(pa_kargs kargs,
                                                    const void* kv_ptr, int kv_rows,
                                                    const int* kv_indices, int page_idx_begin,
                                                    int valid_kv_len, int num_kv_tiles,
                                                    char* smem_kv, char* smem_ml, char* smem_p,
                                                    VQ& v_q, VO& v_o,
                                                    typename Traits::D_ACC& m_row,
                                                    typename Traits::D_ACC& l_row) {
    (void)kargs; (void)kv_ptr; (void)kv_rows; (void)kv_indices; (void)page_idx_begin;
    (void)valid_kv_len; (void)num_kv_tiles; (void)smem_kv; (void)smem_ml; (void)smem_p;
    (void)v_q; (void)v_o; (void)m_row; (void)l_row;
}

} // namespace pa_16mx1_16nx4_fp8

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void pa_prefill_16mx1_16nx4_fp8_kernel(pa_kargs kargs) {
    (void)kargs;
}
