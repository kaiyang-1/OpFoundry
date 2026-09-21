# Sparse Paged Prefill Attention for DeepSeek-V4

Optimized sparse paged prefill attention using the [OPUS](https://github.com/ROCm/aiter) C++ template library for DeepSeek-V4 inference on AMD gfx950 and gfx1250.

This directory targets the DeepSeek-V4 MLA prefill shape with configurable query-head count `H_Q` and `D = 512` head dimension. The kernel consumes two sparse K/V sources: a unified prefix cache with layout `[total_pages, D]` and a current extend K/V tensor with layout `[total_tokens, D]`.

Each architecture ships its own kernel set: gfx950 uses wave64 MFMA with two wave layouts, gfx1250 uses wave32 WMMA with three wave layouts plus TDM gather and cluster launch. Both provide a BF16 and an MXFP8 path, and the host selects the variant at runtime from `-dtype` and `H_Q` (see [Kernel Variants and Dispatch](#kernel-variants-and-dispatch)).

## Features

- DeepSeek-V4 prefill attention shape: configurable `H_Q`, `D = 512`, BF16 or MXFP8 inputs and BF16 output.
- MLA layout: Q/O carry the query-head dimension, while K/V rows are shared across query heads.
- Paged sparse attention through two CSR index ranges per query token:
  - `prefix`: indices into `unified_kv_ptr` / `[total_pages, D]`.
  - `extend`: indices into `kv_ptr` / `[total_tokens, D]`.
- Online softmax across both CSR ranges, plus a per-head attention sink in the denominator, with no materialized attention matrix.
- gfx950 kernels: MFMA, double-buffered K/V shared-memory tiles, FP32 accumulation.
- gfx1250 kernels: WMMA, TDM gather straight into LDS, multi-buffered K/V tiles, and an optional 2-workgroup cluster that multicasts the shared K/V tile.
- MXFP8 variant: head dimension split into NoPE (448 fp8 elements with E8M0 block scales every 32) and RoPE (64 bf16). NoPE QK^T uses scaled `f8f6f4` 16x16x128 MFMA/WMMA; RoPE QK^T and PV use bf16 16x16x32.
- Standalone host harness: seeded page-table and tensor generation, plus CPU reference validation for both BF16 and MXFP8 paths.

## Files

```text
dsa_v4/
|-- Makefile                                         # Per-arch build rules (ARCH=gfx950 | gfx1250)
|-- common/
|   |-- mla_v4_kargs.h                               # Kernel argument structs (bf16 + MXFP8) and dtype aliases
|   |-- mla_v4_parallel.h                            # std::thread parallel_for backing the CPU reference
|   `-- mla_v4_host.cc                               # Host launcher, test harness, CPU references, arch dispatch
|-- gfx950/                                          # wave64 / MFMA
|   |-- mla_v4_traits.h                              # The four gfx950 trait sets
|   |-- mla_v4_global_load.hpp                       # 64-bit global load helpers shared by the gfx950 kernels
|   |-- mla_v4_prefill_a16w16_16mx1_16nx4_kernel.cc  # + _template.hpp
|   |-- mla_v4_prefill_a16w16_16mx8_32nx1_kernel.cc  # + _template.hpp
|   |-- mla_v4_prefill_a8w8_16mx1_16nx4_kernel.cc    # + _template.hpp
|   `-- mla_v4_prefill_a8w8_16mx8_32nx1_kernel.cc    # + _template.hpp
`-- gfx1250/                                         # wave32 / WMMA / TDM gather / cluster launch
    |-- mla_v4_traits.h                              # The six gfx1250 trait sets
    |-- mla_v4_prefill_a16w16_16mx1_16nx4_kernel.cc  # + _template.hpp
    |-- mla_v4_prefill_a16w16_32mx1_16nx4_kernel.cc  # + _template.hpp
    |-- mla_v4_prefill_a16w16_16mx4_64nx1_kernel.cc  # + _template.hpp
    |-- mla_v4_prefill_a8w8_16mx1_16nx4_kernel.cc    # + _template.hpp
    |-- mla_v4_prefill_a8w8_32mx1_16nx4_kernel.cc    # + _template.hpp
    `-- mla_v4_prefill_a8w8_16mx4_64nx1_kernel.cc    # + _template.hpp
```

The Makefile derives `-DMLA_V4_ARCH_<ARCH>` from `ARCH` and compiles only that architecture's directory. `mla_v4_host.cc` is the one host translation unit shared by both: the macro selects which `mla_v4_traits.h` it includes, which kernel symbols it forward-declares, which `mla_v4_prefill_launch` overloads exist (plain `<<<>>>` on gfx950, `hipLaunchKernelEx` with a cluster dimension on gfx1250), and the variant dispatch in `main`.

Each kernel `.cc` includes only its own template header and emits a single explicit instantiation, so the implementations are compiled in separate translation units and `make -j` builds them in parallel. Each template additionally wraps its device helpers in a namespace named after its own variant, so the identically-named helpers cannot collide.

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

The MXFP8 variants (`opus_mla_v4_prefill_fp8_kargs`) replace each BF16 Q/K/V row with two streams. The NoPE stream packs, per padded row of `D_NOPE_PADDED = 512` fp8 slots: `[ NoPE fp8 (448) | E8M0 block scales (448/32 = 14) | 50 unused bytes ]`; the RoPE stream holds `D_ROPE = 64` bf16 elements. The kernel masks the pad off itself, so it may hold anything. The two streams reconstruct the same `D = 512` head (`448 + 64`). Output `O` stays `[N, H_Q, 512]` bf16.

| Tensor | Shape | Notes |
| --- | --- | --- |
| `q_nope` / `unified_kv_nope` / `kv_nope` | `[*, 512]` fp8 | NoPE elements plus inline E8M0 block scales. |
| `q_rope` / `unified_kv_rope` / `kv_rope` | `[*, 64]` bf16 | RoPE elements. |

## Kernel Variants and Dispatch

The BF16 variants share one kernel-argument struct (`opus_mla_v4_prefill_kargs`) and CPU reference; the MXFP8 variants use their own struct (`opus_mla_v4_prefill_fp8_kargs`) and reference. `-dtype` picks the precision, and the remaining choice depends on the architecture. All variants are correct for any `H_Q > 0` (partial head blocks are masked); the thresholds are performance heuristics.

Trait names below are shown without their common `opus_mla_v4_prefill_` prefix.

### gfx950 (wave64, MFMA, `WARP_SIZE = 64`)

Dispatch is by query-head count: `H_Q <= 32` favors `16mx1_16nx4`, otherwise `16mx8_32nx1`.

| Trait | `T_M` × `T_N` | `KV_TILE` | `NUM_WARPS` | `BLOCK_SIZE` | Heads/WG | Used when |
| --- | --- | --- | --- | --- | --- | --- |
| `a16w16_16mx1_16nx4_traits<16, 64, 512, 4, bf16_t, bf16_t>` | `1 × 4` | `64` | `4` | `256` | `16` | `bf16`, `H_Q <= 32` |
| `a16w16_16mx8_32nx1_traits<16, 32, 512, 8, bf16_t, bf16_t>` | `8 × 1` | `32` | `8` | `512` | `128` | `bf16`, `H_Q > 32` |
| `a8w8_16mx1_16nx4_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>` | `1 × 4` | `64` | `4` | `256` | `16` | `fp8`, `H_Q <= 32` |
| `a8w8_16mx8_32nx1_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>` | `8 × 1` | `32` | `8` | `512` | `128` | `fp8`, `H_Q > 32` |

### gfx1250 (wave32, WMMA, `WARP_SIZE = 32`)

Three wave layouts per precision, again selected by `H_Q`: `<= 16` uses `16mx1_16nx4`, `<= 32` uses `32mx1_16nx4`, and above that `16mx4_64nx1`. Only `16mx4_64nx1` takes a `CLUSTER_Y` parameter: the host sets it to `2` when `ceil_div(H_Q, 64)` is even, otherwise `1`, and with `2` the two workgroups in a cluster share one TDM gather of the K/V tile.

| Trait | `T_M` × `T_N` | `KV_TILE` | `NUM_WARPS` | `BLOCK_SIZE` | Heads/WG | Used when |
| --- | --- | --- | --- | --- | --- | --- |
| `a16w16_16mx1_16nx4_traits<16, 64, 512, 4, bf16_t, bf16_t>` | `1 × 4` | `64` | `4` | `128` | `16` | `bf16`, `H_Q <= 16` |
| `a16w16_32mx1_16nx4_traits<32, 64, 512, 4, bf16_t, bf16_t>` | `1 × 4` | `64` | `4` | `128` | `32` | `bf16`, `H_Q <= 32` |
| `a16w16_16mx4_64nx1_traits<16, 64, 512, 4, CLUSTER_Y, bf16_t, bf16_t>` | `4 × 1` | `64` | `4` | `128` | `64` | `bf16`, `H_Q > 32` |
| `a8w8_16mx1_16nx4_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>` | `1 × 4` | `64` | `4` | `128` | `16` | `fp8`, `H_Q <= 16` |
| `a8w8_32mx1_16nx4_traits<32, 64, 4, fp8_t, bf16_t, bf16_t>` | `1 × 4` | `64` | `4` | `128` | `32` | `fp8`, `H_Q <= 32` |
| `a8w8_16mx4_64nx1_traits<16, 64, 4, CLUSTER_Y, fp8_t, bf16_t, bf16_t>` | `4 × 1` | `64` | `4` | `128` | `64` | `fp8`, `H_Q > 32` |

`Q_TILE_SIZE` is the leading trait parameter (`16`, or `32` for the `32mx1` layouts) and `D_TILE_SIZE = 512` is the head dimension. One workgroup covers one query token and up to `Q_TILE_SIZE * T_M` query heads, and the grid is sized as `grid = (N, ceil_div(H_Q, Q_TILE_SIZE * T_M), 1)`, so arbitrary `H_Q` is supported; values that fill a whole head tile give the best occupancy.

## Build

Prerequisites:

- ROCm 7+ with `hipcc`.
- A gfx950 or gfx1250 GPU.
- OPUS headers from `aiter`, exposed through `OPUS_INCLUDE_DIR`.

```bash
cd opus_attn/dsa_v4
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include
make -j                   # ARCH=gfx1250 by default
make -j ARCH=gfx950
```

The executable lands in `build/$(ARCH)/mla_v4_prefill.exe`, so the two architectures can be built side by side. `make clean` removes the whole `build/` tree.

gfx1250 needs a compiler that knows the target; a stock ROCm `hipcc` may reject it with `invalid target ID 'gfx1250'`. Point the build at a toolchain with gfx1250 support via `make HIPCC=/path/to/hipcc` or `HIP_CLANG_PATH`.

Only the gfx1250 kernels are compiled with `-mllvm -amdgpu-expert-scheduling-mode -mllvm -amdgpu-sched-strategy=coexec -mllvm -amdgpu-anti-hints-for-va-vdst`; they are hand-pipelined against the expert scheduler, while the gfx950 kernels are tuned against the default one. The Makefile keeps these in `SCHED_FLAGS_gfx1250`.

## Run and Validate

Run with the DeepSeek-V4 MLA shape:

```bash
./build/gfx1250/mla_v4_prefill.exe
./build/gfx1250/mla_v4_prefill.exe -dtype fp8 -h_q 128 -n 256 --verify
```

Useful options:

| Option | Default | Description |
| --- | --- | --- |
| `-dtype` | `bf16` | Input precision: `bf16` or `fp8`. `fp8` selects the MXFP8 split variant. |
| `-h_q` | `128` | Number of query heads. Supports arbitrary positive values; also selects the wave layout on both architectures, and the cluster size on gfx1250. |
| `-n` | `4096` | Number of query tokens in the prefill chunk. |
| `-topk` | `1024` | Rows each query token selects from each range. Clamped to that range; `0` selects all of it, giving a dense page table. |
| `-total_pages` | `max(N, topk, 65536)` | Prefix cache size, and the main control over how much of the KV gather cache can serve. Nothing derives it from the other options, hence the floor. |
| `-total_tokens` | `N` | Extend rows in `KV`, tied to `N` because the extend range is the chunk's own K/V. Larger only adds rows the causal ramp cannot reach; smaller caps the ramp early. |
| `--verify` | off | Compare GPU output against the CPU reference implementation. |

Prefix rows are visible to every token, so each one selects `min(topk, total_pages)`. Extend rows are the chunk's own K/V, so token `i` draws from `[0, i]` and gets `min(topk, i + 1)` — a ramp that walks every row length below the budget, and with it every KV-tile remainder the kernel masks. The budget applies per range, so a deep token gets up to `2 * topk` rows. Setting either range to `0` empties it, as in a sequence's first chunk.

Every element and index derives from a fixed seed and its own index, so a shape reproduces bit-exactly across runs, machines and worker counts. Selected pages stay scattered, since sorting would hand the gather a near-sequential read that no real page table provides. The CPU reference runs on `std::thread` workers rather than OpenMP; set `MLA_V4_NUM_THREADS` to override the worker count.

## Integration Notes

- The caller owns CSR construction. Causal, sliding-window, compressed-cache, or top-k semantics should already be reflected in `kv_indices_*` and `kv_indptr_*`.
- Empty CSR rows are allowed; with only the sink in the denominator, the output becomes zero.
- Prefix indices must be in `[0, total_pages)`, and extend indices must be in `[0, total_tokens)`.
- The standalone harness uses `softmax_scale = 1 / sqrt(D)`; integrations pass the scale through `opus_mla_v4_prefill_kargs`.
- This directory intentionally contains only the D=512 prefill variants (BF16 and MXFP8) used by the DeepSeek-V4 MLA inference path.
