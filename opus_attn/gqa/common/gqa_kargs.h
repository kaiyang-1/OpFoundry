// Kernel arguments of the GQA forward kernels, shared by the kernels and the host.
#pragma once
#include <cstddef>

// Fixed-length batch: Q/O [B, N, H, D], K/V [B, N, H_KV, D], with strides (gfx950).
struct opus_gqa_kargs {
    const void* __restrict__ ptr_q;
    const void* __restrict__ ptr_k;
    const void* __restrict__ ptr_v;
    void* __restrict__ ptr_o;
    int B;
    int N;
    int H;
    int H_KV;
    int D;
    int stride_q_b;
    int stride_q_n;
    int stride_q_h;
    int stride_kv_b;
    int stride_kv_n;
    int stride_kv_h;
};

// Varlen: Q/O [total_q, H, D], K/V [total_k, H_KV, D], with cu_seqlens (gfx1250). The packed 88-byte block of aiter's
// fmha_fwd_with_sink_varlen_asm, so the host runs the asm code object on the same block. Naturally aligned: an align-1
// block cannot be loaded into SGPRs.
struct alignas(8) opus_gqa_varlen_kargs {
    void*       d_addr;      // 0x00 O    bf16
    const void* q_addr;      // 0x08 Q    bf16
    const void* k_addr;      // 0x10 K    bf16
    const void* v_addr;      // 0x18 V    bf16
    void*       lse_addr;    // 0x20 LSE  [total_q, H] fp32, natural log
    const void* qseq_addr;   // 0x28 cu_seqlens_q int32[batch + 1]
    const void* kseq_addr;   // 0x30 cu_seqlens_k int32[batch + 1]
    float       scalar;      // 0x38 softmax scale
    int         gqa;         // 0x3C H / H_KV
    int         q_head_num;  // 0x40 H
    int         opt;         // 0x44 OPUS_GQA_OPT_* bits
    int         lse;         // 0x48 1 = write LSE
    int         max_q_len;   // 0x4C max per-batch Q length
    const void* sink_addr;   // 0x50 per-Q-head fp32 sink, already multiplied by scalar; may be null
};

static_assert(sizeof(opus_gqa_varlen_kargs) == 0x58, "the asm code object takes an 88-byte block");
static_assert(offsetof(opus_gqa_varlen_kargs, scalar) == 0x38 && offsetof(opus_gqa_varlen_kargs, opt) == 0x44 &&
              offsetof(opus_gqa_varlen_kargs, sink_addr) == 0x50, "asm kernarg layout");

enum : int { OPUS_GQA_OPT_REVERSE_KV = 1, OPUS_GQA_OPT_DOUBLE_Q = 2, OPUS_GQA_OPT_REMAP_XY = 4 };
