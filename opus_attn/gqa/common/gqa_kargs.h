// Kernel arguments of the GQA forward kernels, shared by the kernels and the host.
#pragma once

// Varlen: Q/O [total_q, H, D], K/V [total_k, H_KV, D], with cu_seqlens. Naturally aligned: an align-1 block cannot be
// loaded into SGPRs.
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
    const void* sink_addr;   // 0x50 per-Q-head fp32 sink, already multiplied by scalar; may be null
};

// DOUBLE_Q: a workgroup takes Q tiles i and tiles - 1 - i. REMAP_XY: the head on grid x.
enum : int { OPUS_GQA_OPT_DOUBLE_Q = 2, OPUS_GQA_OPT_REMAP_XY = 4 };
