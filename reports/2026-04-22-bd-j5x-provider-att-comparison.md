# bd-j5x Provider ATT Comparison

This extends the bd-j5x native HIP high-AI WMMA investigation with comparable
hipBLASLt and IREE profiles for `gemm/tests/ai_very_high_large`.

Shape: `M=11520, N=3840, K=3840`, BF16 inputs/outputs with f32 accumulation.
Arithmetic intensity is about `1645.7 FLOP/byte`, so this is a compute-bound
surrogate for the aligned high-AI balance point. It dispatches the native
`gemm_wmma_128x128` kernel, not the boundary kernel used by
`ai_very_high_extreme`.

## Summary

| Provider | Main ATT signal | Interpretation |
| --- | --- | --- |
| native_hip | Large exposed `s_waitcnt vmcnt(*)` stalls, then LDS and WMMA stalls | Global-load latency is still the most visible limiter. Simple scheduling and padding variants did not improve it in the bd-j5x run. |
| hipBLASLt | `s_waitcnt vmcnt(5)`, LDS loads, then `lgkmcnt` | Tensile hides more global latency than native_hip, but the traced kernel is still limited by LDS-fed operand movement. WMMA issue stalls are low. |
| IREE | VALU data movement, WMMA issue stalls, one large `vmcnt(7)` wait | IREE shifts the bottleneck toward register packing/swizzling and compute-pipeline pressure. It has less explicit LDS/global wait than native_hip and was the fastest fresh timing run. |

The native_hip next useful target is not another isolated shared-memory padding
or launch-bound tweak. The comparison points at a larger software-pipeline
change: reduce exposed global-load wait while preserving the current aligned
128x128 balance point and `SMEM_PAD=8`.

## Timing Context

Fresh direct timing collected in this run:

| Provider | Time | TFLOPS |
| --- | ---: | ---: |
| native_hip | 4401.28 us | 77.19 |
| hipBLASLt | 4506.65 us | 75.39 |
| IREE | 4090.94 us | 83.05 |

The earlier bd-j5x report measured the same shape at native_hip `80.93` TFLOPS,
hipBLASLt `80.65` TFLOPS, and IREE `82.78` TFLOPS. Use the ATT comparison below
for bottleneck shape, not exact absolute timing; the single-run timings vary
enough that the ranking between native_hip and hipBLASLt moves.

## Collection

Kernel names were identified with `gemm/profiling/profile.py --skip-pmc`.
The IREE profiling wrapper now uses the same VMFB cache layout as the normal
IREE provider runner: `$KERNELGEN_CACHE_DIR/vmfb/iree/<gpu-target>/O3`.

Example hipBLASLt ATT command:

```bash
$THEROCK_PATH/bin/rocprofv3 --att \
  --att-library-path $THEROCK_PATH/lib \
  --att-gpu-index 0 --att-simd-select 0 \
  --kernel-include-regex 'Cijk_Ailk_Bljk.*MT96x96x32' \
  --kernel-trace --stats --att-buffer-size 0x6000000 \
  -d $ATT_DIR/hipblaslt-large -- \
  $BUILD_DIR/gemm/kernel_providers/hipblaslt/hipblaslt_gemm_bench \
  --config gemm/tests/ai_very_high_large/config.json \
  --warmup 0 --timed 1
```

Example IREE ATT command:

```bash
$THEROCK_PATH/bin/rocprofv3 --att \
  --att-library-path $THEROCK_PATH/lib \
  --att-gpu-index 0 --att-simd-select 0 \
  --kernel-include-regex main_dispatch_0_matmul \
  --kernel-trace --stats --att-buffer-size 0x6000000 \
  -d $ATT_DIR/iree-large -- \
  $BUILD_DIR/gemm/kernel_providers/iree/iree_gemm_bench \
  --config gemm/tests/ai_very_high_large/config.json \
  --vmfb $KERNELGEN_CACHE_DIR/vmfb/iree/gfx1100/O3/ai_very_high_large.vmfb \
  --warmup 0 --timed 1
```

Raw ATT UI directories were used only as temporary analysis artifacts and are
not checked in.

## ATT Summary

| Provider | Kernel | Wave files | Wave duration min/avg/max | Total stall | Total idle |
| --- | --- | ---: | ---: | ---: | ---: |
| native_hip | `gemm_wmma_128x128` | 112 | 565373 / 966664 / 2093490 | 40.62M | 46.56M |
| hipBLASLt | `Cijk_Ailk_Bljk...MT96x96x32...` | 100 | 227866 / 367700 / 894344 | 7.81M | 24.09M |
| IREE | `main_dispatch_0_matmul_like_11520x3840x3840_bf16` | 57 | 403544 / 877433 / 1438642 | 27.78M | 11.75M |

The absolute stall totals are not directly comparable across providers because
the sampled wave count, traced instruction count, and code shape differ. The
useful comparison is the dominant stall class within each provider.

## native_hip

Native data is from `reports/2026-04-22-bd-j5x-att-wmma.md`.

| Class | Stall |
| --- | ---: |
| `waitcnt:vmcnt(3)` | 17.10M |
| LDS | 4.70M |
| WMMA | 3.79M |
| `waitcnt:vmcnt(10)` | 3.36M |
| `waitcnt:vmcnt(2)` | 2.71M |
| `waitcnt:lgkmcnt(0)` | 1.77M |
| `waitcnt:lgkmcnt(6)` | 1.68M |

The top waitcnt sources were scalar B `global_load_u16` rows and A
`global_load_b128` rows. That means the aligned native kernel is still exposing
global-load latency before operands are available to the LDS/WMMA pipeline.

## hipBLASLt

Kernel:
`Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT96x96x32_MI16x16x1...WS32_WG32_4_1`

| Class | Stall | Share |
| --- | ---: | ---: |
| `waitcnt:vmcnt(5)` | 3.18M | 40.6% |
| LDS | 2.69M | 34.4% |
| `waitcnt:lgkmcnt(0)` | 1.13M | 14.5% |
| `waitcnt:lgkmcnt(4)` | 0.43M | 5.4% |
| `waitcnt:vmcnt(0)` | 0.18M | 2.3% |
| VALU | 0.15M | 1.9% |
| WMMA | 0.01M | 0.2% |

| Line | Class | Stall | Instruction |
| ---: | --- | ---: | --- |
| 675 | `waitcnt:vmcnt(5)` | 1.71M | `s_waitcnt vmcnt(5)` |
| 678 | `waitcnt:vmcnt(5)` | 1.24M | `s_waitcnt vmcnt(5)` |
| 712 | `waitcnt:lgkmcnt(0)` | 1.07M | `s_waitcnt lgkmcnt(0)` |
| 607 | `waitcnt:lgkmcnt(4)` | 0.42M | `s_waitcnt lgkmcnt(4)` |
| 525 | `waitcnt:vmcnt(0)` | 0.18M | `s_waitcnt vmcnt(0)` |
| 741 | LDS | 0.07M | `ds_load_u16_d16_hi v93, v80 offset:640` |

Top waitcnt source rows were all LDS loads, for example
`ds_load_u16 v108, v80 offset:3104` through
`ds_load_u16 v111, v80 offset:4256`. hipBLASLt still waits on global memory at
the top level, but once data reaches LDS the dominant visible pressure is LDS
operand feeding. WMMA stall attribution is small, so the traced Tensile kernel
looks more memory-feed limited than compute-issue limited.

## IREE

Kernel: `main_dispatch_0_matmul_like_11520x3840x3840_bf16`

| Class | Stall | Share |
| --- | ---: | ---: |
| VALU | 17.09M | 61.5% |
| `waitcnt:vmcnt(7)` | 3.58M | 12.9% |
| LDS | 2.50M | 9.0% |
| WMMA | 1.99M | 7.2% |
| `waitcnt:lgkmcnt(5)` | 0.45M | 1.6% |
| `waitcnt:lgkmcnt(3)` | 0.40M | 1.4% |
| `waitcnt:lgkmcnt(0)` | 0.33M | 1.2% |
| `waitcnt:lgkmcnt(7)` | 0.33M | 1.2% |

| Line | Class | Stall | Instruction |
| ---: | --- | ---: | --- |
| 779 | `waitcnt:vmcnt(7)` | 3.58M | `s_waitcnt vmcnt(7)` |
| 726 | VALU | 0.66M | `v_mov_b16_e64 v97.h, v129.h op_sel:[1,1]` |
| 655 | VALU | 0.62M | `v_mov_b16_e64 v93.h, v129.h op_sel:[1,1]` |
| 578 | WMMA | 0.59M | `v_wmma_bf16_16x16x16_bf16 ...` |
| 607 | VALU | 0.59M | `v_mov_b16_e64 v14.h, v129.h op_sel:[1,1]` |
| 551 | VALU | 0.59M | `v_mov_b16_e64 v22.h, v129.h op_sel:[1,1]` |

The repeated `v_mov_b16_e64 ... op_sel` rows point to register packing or
lane-half movement around the WMMA operands. IREE exposes less total LDS wait
than hipBLASLt as a share of its stall total, but pays a large VALU movement
cost and visible WMMA issue stalls. This is consistent with a kernel that is
closer to compute/register-pipeline pressure than raw global-memory latency.

## Native HIP Implications

1. Native_hip and hipBLASLt both expose global-memory waits, but hipBLASLt's
   largest VMEM wait is much smaller in this trace and the next pressure point
   is LDS operand delivery. Native_hip still needs a better global-to-LDS
   pipeline before LDS-only tuning is likely to matter.
1. IREE wins the fresh direct timing and shows a different profile: high VALU
   operand movement and WMMA stalls, lower explicit LDS/global wait share. For
   native_hip, shifting stall attribution from VMEM wait toward VALU/WMMA would
   likely be progress for this high-AI balance point.
1. hipBLASLt uses a `96x96x32` Tensile tile while native_hip uses the aligned
   `128x128` WMMA path. The tile difference suggests there may be another
   useful balance point between native_hip's current large tile and IREE's
   compiler-generated schedule, but this report does not prove a replacement
   tile. A new tile should be justified by reducing `vmcnt` stalls, not just by
   matching the tile size.
1. `ai_very_high_extreme` still dispatches the native boundary kernel because
   `M=150000` is not divisible by 128. This provider comparison explains the
   aligned high-AI kernel, not the boundary-path performance gap on extreme.

## Reproduction Notes

After collecting ATT, generate compact summaries with:

```bash
.venv/bin/python gemm/profiling/analyze_att.py \
  $ATT_DIR/hipblaslt-large/ui_output_agent_*_dispatch_* --top 20

.venv/bin/python gemm/profiling/analyze_att.py \
  $ATT_DIR/iree-large/ui_output_agent_*_dispatch_* --top 20
```
