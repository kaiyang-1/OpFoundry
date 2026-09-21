# Paged MLA Decode for DeepSeek-V3.2

Paged MLA decode attention built on the [OPUS](https://github.com/ROCm/aiter) C++ template library, targeting AMD gfx950. Head dimensions are `D_QK = 576` (512 NoPE + 64 RoPE) and `D_VO = 512`; K/V is a shared latent row per page, so the query-head dimension exists only on Q/O.

Both a BF16 and an MXFP8 path are provided, and the host picks a wave layout at runtime from `-dtype` and the query-head count (see [Kernel Variants and Dispatch](#kernel-variants-and-dispatch)).

## Pipeline

A decode launch is three kernels, wired together by `mla_decode_launch_pipeline()` in [common/host.cc](common/host.cc):

1. `get_mla_metadata_kernel` — splits the per-batch KV ranges into `opus_mla_decode_work_info` entries balanced across `MLA_DECODE_NUM_CU` CUs, and emits the reduce map for the split rows.
2. The decode kernel — one workgroup drains its slice of the work list, running online softmax over the paged KV rows and writing either the final output or a partial `(o_accum, lse_accum)`.
3. `mla_reduce_kernel` — merges the split partials by LSE into the final `O` and `LSE`.

Stages 1 and 3 carry no gfx950-specific instructions and live in `common/`. The kernel-argument structs in [common/defs.h](common/defs.h) mirror aiter's ABI — the `static_assert`s on their sizes and offsets are there because prebuilt code objects are compiled against those exact layouts, so do not reorder fields.

## Files

```text
dsa_v32/
|-- Makefile                                       # Per-arch build rules (ARCH=gfx950)
|-- common/
|   |-- defs.h                                     # dtypes, fp8 helpers, scheduler constants, kargs ABI; pulls in the arch traits
|   |-- parallel.h                                 # std::thread parallel_for backing the CPU reference
|   |-- get_mla_metadata.hpp / _kernel.cc          # Arch-independent work scheduler
|   |-- mla_reduce.hpp / mla_reduce_kernel.cc      # Arch-independent split-KV merge
|   `-- host.cc                                    # Host launcher, test harness, CPU references
`-- gfx950/                                        # wave64 / MFMA
    |-- traits.h                                   # The five gfx950 trait sets
    |-- global_load.hpp                            # global_load_lds helpers
    |-- mla_decode_a16w16_16mx4_64nx1_kernel.cc    # + _template.hpp
    |-- mla_decode_a16w16_32mx1_16nx4_kernel.cc    # + _template.hpp
    |-- mla_decode_a16w16_32mxt_32nx1_kernel.cc    # + _template.hpp (instantiates 32mx4 and 32mx3)
    `-- mla_decode_mxfp8_16mx8_32nx1_kernel.cc     # + _template.hpp
```

`common/defs.h` ends with an `#if defined(MLA_DECODE_ARCH_GFX950)` that includes `gfx950/traits.h`, so every translation unit includes exactly one header to get both the ABI and the traits; the Makefile derives `-DMLA_DECODE_ARCH_<ARCH>` from `ARCH`. The include direction is one-way (`defs.h` -> `traits.h`); `traits.h` must not include `defs.h` back.

Each kernel `.cc` is a thin stub that emits only explicit instantiations, so the kernels compile as separate translation units and `make -j` builds them in parallel. Each template wraps its device helpers in a namespace named after its own variant, so identically-named helpers cannot collide at link time.

## Kernel Variants and Dispatch

All variants use `WARP_SIZE = 64`. The decode grid is persistent: `grid = (MLA_DECODE_NUM_CU, ceil_div(H, Q_TILE_SIZE * T_M), 1)`, so one workgroup covers one head tile on one CU and drains its slice of the work list. Trait names are shown without their common `opus_mla_decode_` prefix. Dispatch happens in `main()` in [common/host.cc](common/host.cc); the thresholds are performance heuristics, not correctness requirements — any `H > 0` works, since a partial head tile is masked.

| Trait | `T_M` × `T_N` | `KV_TILE` | `NUM_WARPS` | `NUM_KV_BUFS` | Used when |
| --- | --- | --- | --- | --- | --- |
| `mxfp8_16mx8_32nx1_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>` | `8 × 1` | `32` | `8` | `2` | `-dtype fp8` (any `H`) |
| `a16w16_32mx1_16nx4_traits<32, 64, 4, bf16_t, bf16_t>` | `1 × 4` | `64` | `4` | `2` | `bf16`, `H <= 32` |
| `a16w16_32mx4_32nx1_traits<32, 32, 4, bf16_t, bf16_t>` | `4 × 1` | `32` | `4` | `4` | `bf16`, `H % 128 == 0` |
| `a16w16_32mx3_32nx1_traits<32, 32, 4, bf16_t, bf16_t>` | `3 × 1` | `32` | `4` | `4` | `bf16`, `H % 96 == 0` |
| `a16w16_16mx4_64nx1_traits<16, 64, 4, bf16_t, bf16_t>` | `4 × 1` | `64` | `4` | `2` | `bf16`, otherwise |

`32mx4` and `32mx3` are two aliases of the same `32mxt_32nx1_traits_base` and share one template. They differ in `NUM_COMPUTE_WARPS`: `32mx4` uses all four warps for MFMA, while `32mx3` computes on three and dedicates the fourth as a producer warp that only issues the KV loads into LDS.

### MXFP8 layout

The MXFP8 variant splits each latent row into a NoPE stream of 512 fp8 elements with E8M0 block scales every 32 (`D_SCALE_SIZE = 16`, padded to 32 per row) and a RoPE stream of 64 bf16 elements. NoPE `QK^T` uses the scaled `f8f6f4` 16x16x128 MFMA; RoPE `QK^T` and `PV` stay bf16 16x16x32. Output `O` and `LSE` are bf16 / fp32 in both paths.

## Build

Prerequisites:

- A gfx950 GPU and ROCm 7+.
- OPUS headers from `aiter`, exposed through `OPUS_INCLUDE_DIR`.
- A compiler built from the [`amdgpu-pin-op-dst`](https://github.com/yuyzhang512/llvm-project/commits/amdgpu-pin-op-dst/) branch of LLVM, pointed at through `HIP_CLANG_PATH`. The kernels are tuned against that toolchain; a stock ROCm `hipcc` is not expected to reproduce the same schedule.

```bash
cd opus_attn/dsa_v32
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include
export HIP_CLANG_PATH=/path/to/llvm-project/build/bin
make -j
```

The executable lands in `build/$(ARCH)/mla_decode.exe`. `make asm` dumps the device assembly for every kernel next to the objects, and `make clean` removes the whole `build/` tree.

## Run and Validate

```bash
./build/gfx950/mla_decode.exe -dtype fp8  -h_q 128 -b 128 -s 1024 --verify
./build/gfx950/mla_decode.exe -dtype bf16 -h_q 128 -b 128 -s 1024 --varlen --verify
```

`-dtype fp8` always lands on `mxfp8_16mx8_32nx1`; for `bf16`, `-h_q 16 / 40 / 96 / 128` cover `32mx1_16nx4`, `16mx4_64nx1`, `32mx3_32nx1` and `32mx4_32nx1` respectively, so those four values sweep every dispatch branch.

| Option | Default | Description |
| --- | --- | --- |
| `-dtype` | `fp8` | Input precision: `fp8` (MXFP8 split) or `bf16`. |
| `-h_q` | `128` | Number of query heads. Also selects the wave layout. |
| `-b` | `128` | Batch size. |
| `-s` | `1024` | KV length per batch (the maximum when `--varlen` is set). |
| `-s_q` | `1` | Query tokens per batch (`> 1` for MTP / speculative decode). |
| `--varlen` | off | Randomize each batch's KV length in `[min(5, s), s]` instead of using `s`. |
| `--verify` | off | Compare GPU `O` and `LSE` against the CPU reference. |

The harness fills random Q/KV tensors, builds a shuffled page table (page size 1, so `kv_indices` holds one page index per KV row), runs the three-kernel pipeline, optionally checks against `mla_decode_attention_ref_bf16()` / `mla_decode_attention_ref_fp8()` in [common/host.cc](common/host.cc), and reports TFLOPs and effective TB/s. The reference runs on `std::thread` workers rather than OpenMP; set `MLA_DECODE_NUM_THREADS` to override the worker count.

## Integration Notes

- The caller owns the page table. `kv_indptr` / `kv_indices` are CSR over the batch, and `stride_kv_page` is the row stride of the latent KV tensor.
- In production aiter owns the metadata and reduce stages; `opus_mla_decode_metadata_kargs` and `opus_mla_decode_reduce_kargs` exist here only so the harness can run standalone, and are not ABI.
- The harness uses `softmax_scale = 1 / sqrt(D_QK)`; integrations pass the scale through the kargs.
