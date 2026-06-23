# Sparse Paged Prefill Attention for DeepSeek-V4

Optimized sparse paged prefill attention using the [OPUS](https://github.com/ROCm/aiter) C++ template library for DeepSeek-V4 inference on AMD gfx950.

This directory targets the DeepSeek-V4 MLA prefill shape with configurable query-head count `H_Q` and `D = 512` head dimension. The kernel consumes two sparse K/V sources: a unified prefix cache with layout `[total_pages, D]` and a current extend K/V tensor with layout `[total_tokens, D]`.

Three kernel variants are compiled into one binary: two BF16 variants with different MFMA wave layouts, plus an MXFP8 variant that stores the head dimension as a split NoPE (fp8) / RoPE (bf16) pair. The host selects the variant at runtime from `-dtype` and `H_Q` (see [Kernel Variants and Dispatch](#kernel-variants-and-dispatch)).

## Features

- DeepSeek-V4 prefill attention shape: configurable `H_Q`, `D = 512`, BF16 or MXFP8 inputs and BF16 output.
- MLA layout: Q/O carry the query-head dimension, while K/V rows are shared across query heads.
- Paged sparse attention through two CSR index ranges per query token:
  - `prefix`: indices into `unified_kv_ptr` / `[total_pages, D]`.
  - `extend`: indices into `kv_ptr` / `[total_tokens, D]`.
- Online softmax across both CSR ranges, plus a per-head attention sink in the denominator, with no materialized attention matrix.
- OPUS-based gfx950 kernels using MFMA, double-buffered K/V shared-memory tiles, and FP32 accumulation.
- MXFP8 variant: head dimension split into NoPE (448 fp8 elements with E8M0 block scales every 32) and RoPE (64 bf16). NoPE QK^T uses scaled `f8f6f4` 16x16x128 MFMA; RoPE QK^T and PV use bf16 16x16x32.
- Three kernel variants selected at runtime from `-dtype` and query-head count for best occupancy across `H_Q`.
- Standalone host harness with random sparse/dense index generation and CPU reference validation for both BF16 and MXFP8 paths.

## Files

```text
dsa_v4/
|-- Makefile                                  # Build rules for the standalone executable
|-- pa_defs.h                                 # Kernel argument structs and the three compile-time trait sets
|-- pa_host.cc                                # Host launcher, test harness, and CPU references (bf16 + fp8)
|-- pa_prefill_16mx1_16nx4_kernel.cc          # 16mx1_16nx4 variant instantiation (D=512 bf16)
|-- pa_prefill_16mx1_16nx4_template.hpp       # 16mx1_16nx4 OPUS/HIP kernel implementation
|-- pa_prefill_16mx8_32nx1_kernel.cc          # 16mx8_32nx1 variant instantiation (D=512 bf16)
|-- pa_prefill_16mx8_32nx1_template.hpp       # 16mx8_32nx1 OPUS/HIP kernel implementation
|-- pa_prefill_16mx1_16nx4_fp8_kernel.cc      # 16mx1_16nx4 MXFP8 variant instantiation
`-- pa_prefill_16mx1_16nx4_fp8_template.hpp   # 16mx1_16nx4 MXFP8 OPUS/HIP kernel implementation
```

Each kernel `.cc` includes only its own template header and emits a single explicit
instantiation. The host (`pa_host.cc`) forward-declares both kernel symbols and never
includes a template body, so the three implementations are compiled in separate translation
units. Each template additionally wraps its device helpers in its own namespace
(`pa_16mx1_16nx4` / `pa_16mx8_32nx1` / `pa_16mx1_16nx4_fp8`), so identically-named helpers
cannot collide even if multiple headers were ever included in one translation unit.

## Attention Model

For each query token `i`, the caller provides two CSR rows:

```text
prefix_rows = kv_indices_prefix[kv_indptr_prefix[i] : kv_indptr_prefix[i + 1]]
extend_rows = kv_indices_extend[kv_indptr_extend[i] : kv_indptr_extend[i + 1]]
```

The kernel computes scaled dot-product attention over `prefix_rows` followed by `extend_rows`. Prefix rows index `UnifiedKV`; extend rows index the current `KV` tensor. Both ranges share the same online-softmax state:

```text
scores = concat(
  Q[i, h, :] @ UnifiedKV[prefix_rows, :].T,
  Q[i, h, :] @ KV[extend_rows, :].T
) * softmax_scale
P = exp(scores) / (sum(exp(scores)) + exp(attn_sink[h]))
O[i, h, :] = P_prefix @ UnifiedKV[prefix_rows, :] + P_extend @ KV[extend_rows, :]
```

In the DeepSeek-V4 prefill path, these two logical ranges map naturally to:

- `prefix`: previously available state, such as the sliding-window tail and compressed cache pages.
- `extend`: K/V rows produced by the current prefill chunk.

## Tensor Layout

Q/K/V/O tensor data is BF16. `AttnSink` and accumulation stay FP32.

| Tensor | Shape | Notes |
| --- | --- | --- |
| `Q` | `[N, H_Q, D]` | Query tokens. `H_Q` is configurable, such as `64` for DeepSeek-V4-Flash and `128` for DeepSeek-V4-Pro. |
| `UnifiedKV` | `[total_pages, D]` | Prefix K/V rows. |
| `KV` | `[total_tokens, D]` | Current extend K/V rows. |
| `AttnSink` | `[H_Q]` | Per-head sink score included in the softmax denominator only. |
| `O` | `[N, H_Q, D]` | Output tokens. |
| `kv_indptr_prefix` | `[N + 1]` | CSR row pointers for prefix rows. |
| `kv_indices_prefix` | `[nnz_prefix]` | Row indices into `UnifiedKV`. |
| `kv_indptr_extend` | `[N + 1]` | CSR row pointers for extend rows. |
| `kv_indices_extend` | `[nnz_extend]` | Row indices into `KV`. |

The kernel assumes row-major contiguous layout with `D` as the fastest-changing dimension.

### MXFP8 split layout

The MXFP8 variant (`pa_fp8_kargs`) replaces each BF16 Q/K/V row with two streams. The
NoPE stream packs, per padded row of `D_NOPE_PADDED = 512` fp8 slots:
`[ NoPE fp8 (448) | E8M0 block scales (448/32 = 14) | fp8 zero-pad ]`; the RoPE stream
holds `D_ROPE = 64` bf16 elements. The two streams reconstruct the same `D = 512` head
(`448 + 64`). Output `O` stays `[N, H_Q, 512]` bf16.

| Tensor | Shape | Notes |
| --- | --- | --- |
| `q_nope` / `unified_kv_nope` / `kv_nope` | `[*, 512]` fp8 | NoPE elements plus inline E8M0 block scales. |
| `q_rope` / `unified_kv_rope` / `kv_rope` | `[*, 64]` bf16 | RoPE elements. |

## Kernel Variants and Dispatch

The two BF16 variants share one kernel-argument struct (`pa_kargs`) and CPU reference but
use different MFMA wave layouts; the MXFP8 variant uses its own struct (`pa_fp8_kargs`) and
reference. Dispatch is driven by `-dtype` first, then by query-head count: `-dtype fp8`
always selects the MXFP8 variant; for `-dtype bf16`, `H_Q <= 32` favors `16mx1_16nx4`,
otherwise `16mx8_32nx1`. The BF16 variants are correct for any `H_Q > 0` (partial head
blocks are masked); the threshold is a performance heuristic.

| Variant | Trait | `T_M` × `T_N` | `KV_TILE` | `NUM_WARPS` | `BLOCK_SIZE` | Used when |
| --- | --- | --- | --- | --- | --- | --- |
| `16mx1_16nx4` | `pa_16mx1_16nx4_traits<16, 64, 512, 4, bf16_t>` | `1 × NUM_WARPS` | `64` | `4` | `256` | `-dtype bf16`, `H_Q <= 32` |
| `16mx8_32nx1` | `pa_16mx8_32nx1_traits<16, 32, 512, 8, bf16_t>` | `NUM_WARPS × 1` | `32` | `8` | `512` | `-dtype bf16`, `H_Q > 32` |
| `16mx1_16nx4_fp8` | `pa_16mx1_16nx4_fp8_traits<16, 64, 640, 4, fp8_t, bf16_t, bf16_t>` | `1 × NUM_WARPS` | `64` | `4` | `256` | `-dtype fp8`, any `H_Q` |

Common trait parameters for both: `Q_TILE_SIZE = 16` (query-head tile per wave),
`D_TILE_SIZE = 512` (head dimension), `WARP_SIZE = 64`.

One workgroup covers one query token and up to `Q_TILE_SIZE * T_M` query heads
(`16` for `16mx1_16nx4`, `128` for `16mx8_32nx1`). The grid is sized as
`grid.y = ceil_div(H_Q, Q_TILE_SIZE * T_M)`, so arbitrary `H_Q` is supported; values that
fill a whole head tile give the best occupancy.

## Build

Prerequisites:

- ROCm 7+ with `hipcc`.
- gfx950 GPU target.
- OPUS headers from `aiter`, exposed through `OPUS_INCLUDE_DIR`.
- OpenMP support for the host reference path.

```bash
cd opus_attn/dsa_v4
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include
make -j
```

## Run and Validate

Run with the DeepSeek-V4 MLA shape:

```bash
# BF16
./build/pa_prefill.exe -h_q 128 -n 256 -total_pages 1024 -total_tokens 2048 --verify
# MXFP8 (split NoPE fp8 / RoPE bf16)
./build/pa_prefill.exe -dtype fp8 -h_q 16 -n 256 -total_pages 1024 -total_tokens 2048 --verify
```

Useful options:

| Option | Default | Description |
| --- | --- | --- |
| `-dtype` | `bf16` | Input precision: `bf16` or `fp8`. `fp8` selects the MXFP8 split variant. |
| `-h_q` | `128` | Number of query heads. Supports arbitrary positive values; for `bf16` it selects the wave layout (`<= 32` vs `> 32`). |
| `-n` | `1024` | Number of query tokens in the standalone harness. |
| `-total_pages` | `N` | Number of prefix rows in `UnifiedKV`. |
| `-total_tokens` | `N` | Number of extend rows in `KV`. |
| `--dense` | off | Generate dense CSR rows instead of random sparse rows. |
| `--verify` | off | Compare GPU output against the CPU reference implementation. |

The harness initializes random BF16 attention tensors and random per-head sink scores,
generates prefix and extend CSR index ranges, launches the kernel, optionally checks the
result against the CPU reference in `pa_host.cc` (`pa_attention_ref()` for bf16,
`pa_attention_ref_fp8()` for the MXFP8 split path), and then reports benchmark timing
(TFLOPs and effective TB/s).

## Integration Notes

- The caller owns CSR construction. Causal, sliding-window, compressed-cache, or top-k semantics should already be reflected in `kv_indices_*` and `kv_indptr_*`.
- Empty CSR rows are allowed; with only the sink in the denominator, the output becomes zero.
- Prefix indices must be in `[0, total_pages)`, and extend indices must be in `[0, total_tokens)`.
- The standalone harness uses `softmax_scale = 1 / sqrt(D)`; integrations pass the scale through `pa_kargs`.
- This directory intentionally contains only the D=512 prefill variants (BF16 and MXFP8) used by the DeepSeek-V4 MLA inference path.
