# bd-j5x Native HIP Aligned WMMA ATT Investigation

Date: 2026-04-22
Worktree: `/home/mahesh/kernelGen/kernelGen-bd-j5x`
Branch: `users/MaheshRavishankar/bd-j5x-rebuildHighAiAttKernel`
PR: #10

## Summary

No kernel edit produced a defensible improvement for the reusable aligned
`gemm_wmma_128x128` high-AI path. The existing 8-warp, 128x128, `SMEM_PAD=8`
kernel remains the best balance point found in this run.

The fresh ATT trace for `ai_very_high_large` shows the same dominant bottleneck
family as the starting evidence: `vmcnt` waits on the global-load-to-LDS staging
path, followed by LDS waits/stalls and WMMA issue stalls. Attempts to reduce
global-load instruction count or change scheduling either moved the bottleneck to
LDS or regressed neighboring shapes. Because no code change improved the primary
large surrogate and adjacent aligned shapes together, the source kernel was left
unchanged. This branch adds the reusable ATT analyzer used for the investigation.

## Baseline Timings

Release build:

```bash
cmake --build /home/mahesh/kernelGen/build/bd-j5x-Release --target native_hip_gemm_bench
```

Initial native_hip baseline, default runner timing (`warmup=5`, `timed=20`):

| Test | Shape | Native HIP | hipBLASLt | IREE |
| --- | ---: | ---: | ---: | ---: |
| `ai_very_high_large` | 11520x3840x3840 | 4197.7 us, 80.93 TFLOPS | 4212.3 us, 80.65 TFLOPS | 4104.3 us, 82.78 TFLOPS |
| `ai_very_high_square` | 4096x4096x4096 | 1772.4 us, 77.54 TFLOPS | 1800.4 us, 76.34 TFLOPS | 1677.7 us, 81.92 TFLOPS |
| `ai_very_high_medium` | 3840x3840x2304 | 875.1 us, 77.65 TFLOPS | 897.6 us, 75.70 TFLOPS | 891.5 us, 76.22 TFLOPS |
| `small_f16` | 1024x1024x1024 | 43.0 us, 49.98 TFLOPS, PASS | 37.2 us, 57.79 TFLOPS, PASS | 106.5 us, 20.16 TFLOPS, PASS |

Final native_hip after reverting all kernel experiments (`warmup=5`,
`timed=20`; code identical to baseline):

| Test | Native HIP Final |
| --- | ---: |
| `ai_very_high_large` | 4290.5 us, 79.18 TFLOPS |
| `ai_very_high_square` | 1851.1 us, 74.25 TFLOPS |
| `ai_very_high_medium` | 910.2 us, 74.66 TFLOPS |
| `small_f16` | 43.0 us, 49.89 TFLOPS, PASS |

The standard rocprof summary for `ai_very_high_large` reported 4323.7 us,
78.58 TFLOPS, 16.9% timing variance, 120 VGPRs, 128 SGPRs, 20480 bytes LDS,
256-thread workgroups, and 100 LDS bank conflicts over 21600 waves
(effectively 0.0 conflicts/wave). Arithmetic intensity is 1645.7 FLOP/byte, so
the roofline classification is compute-bound.

## ATT Collection

Fresh traces were regenerated after the Release build. The primary command shape
was:

```bash
/home/mahesh/kernelGen/TheRock/bin/rocprofv3 --att \
  --att-library-path /home/mahesh/kernelGen/TheRock/lib \
  --att-gpu-index 0 --att-simd-select 0 \
  --kernel-include-regex gemm_wmma_128x128 \
  --kernel-exclude-regex boundary --kernel-trace --stats \
  --att-buffer-size 0x6000000 \
  -d /tmp/kernelgen-bd-j5x-att/fresh-baseline-large -- \
  /home/mahesh/kernelGen/build/bd-j5x-Release/gemm/kernel_providers/native_hip/native_hip_gemm_bench \
  --config /home/mahesh/kernelGen/kernelGen-bd-j5x/gemm/tests/ai_very_high_large/config.json \
  --warmup 0 --timed 1
```

`gemm/profiling/analyze_att.py` was used on the generated
`ui_output_agent_*_dispatch_2` directories.

### `ai_very_high_large`

ATT timing under rocprof: 5114.0 us.

| Metric | Value |
| --- | ---: |
| Wave files | 112 |
| Wave duration min/avg/max | 565373 / 966664 / 2093490 |
| Total latency / stall / idle | 61.84M / 40.62M / 46.56M |
| Top stall class | `waitcnt:vmcnt(3)`, 17.10M stall |
| Next stall classes | LDS 4.70M, WMMA 3.79M, `waitcnt:vmcnt(10)` 3.36M, `waitcnt:vmcnt(2)` 2.71M |
| Top LDS waits | `lgkmcnt(0)` 1.77M, `lgkmcnt(6)` 1.68M |
| Top waitcnt sources | scalar B `global_load_u16` rows and A `global_load_b128` rows |

The largest single stall row is `s_waitcnt vmcnt(3)`. The source dependency list
points at the global-load staging sequence, especially the A `global_load_b128`
loads and the scalar B load ladder. The profile does not show meaningful LDS
bank conflicts, so the LDS stalls appear to be scheduling/latency pressure rather
than a simple bank-conflict fix.

### `ai_very_high_square`

ATT timing under rocprof: 3481.3 us.

| Metric | Value |
| --- | ---: |
| Wave files | 42 |
| Wave duration min/avg/max | 831457 / 1484246 / 2650143 |
| Total latency / stall / idle | 41.48M / 33.27M / 20.91M |
| Top stall class | `waitcnt:vmcnt(3)`, 18.90M stall |
| Next stall classes | `waitcnt:vmcnt(10)` 3.74M, LDS 2.12M, `waitcnt:vmcnt(2)` 2.09M, `waitcnt:vmcnt(8)` 1.41M |
| WMMA issue stalls | 1.20M |
| Top waitcnt sources | scalar B `global_load_u16` rows and A `global_load_b128` rows |

The square trace agrees with the large trace: VMEM waitcnt pressure dominates,
with LDS and WMMA stalls secondary.

## Tried Strategies

All changes below were reverted.

| Strategy | Result | Decision |
| --- | --- | --- |
| `SMEM_PAD=0` | Large 7833.2 us / 43.37 TFLOPS; square 3348.3 us / 41.05; medium 1620.5 us / 41.93; small_f16 88.3 us PASS | Rejected. Removing padding roughly halves throughput despite lower LDS footprint. |
| `SMEM_PAD=4` | Large 14412.9 us / 23.57 TFLOPS; square 6017.1 us / 22.84; medium 2992.0 us / 22.71; small_f16 156.1 us PASS | Rejected. This stride is worse than no padding. |
| Vectorized B row loads with shared transpose scatter | Large 7176.7 us / 47.34 TFLOPS; square 3052.1 us / 45.03; medium 1479.3 us / 45.93; small_f16 79.5 us PASS | Rejected. Fewer global load instructions did not pay for the scattered LDS stores. |
| Load B before A | Large 4223.7 us / 80.44 TFLOPS; square 1833.7 us / 74.95; medium 863.7 us / 78.67; small_f16 43.2 us PASS | Rejected. Medium improved, but primary large and square regressed. |
| `__launch_bounds__(BLOCK_SIZE, 2)` | Large 4237.7 us / 80.17 TFLOPS; square 1829.5 us / 75.13; medium 904.6 us / 75.11; small_f16 43.1 us PASS | Rejected on this branch/run. |
| `__launch_bounds__(BLOCK_SIZE, 1)` | Large 4198.9 us / 80.91 TFLOPS; square 1802.3 us / 76.26; medium 852.8 us / 79.68; small_f16 43.0 us PASS | Rejected. Improves medium but does not materially improve large and regresses square. |
| Store alpha fast path for `alpha == 1.0f` | Large 4233.6 us / 80.25 TFLOPS; square 1838.6 us / 74.75; medium 839.5 us / 80.93; small_f16 43.2 us PASS | Rejected. Store-side work is not the large/square bottleneck. |

I did not repeat the earlier ragged-M split strategy. Fresh evidence did not show
a reason to trade the aligned high-AI path for M-split dispatch: `ai_very_high_large`,
`ai_very_high_square`, and `ai_very_high_medium` all dispatch directly to the
aligned 128x128 kernel and are already close to hipBLASLt. The extreme shape is
not aligned in M, so it is a boundary-kernel case rather than evidence for the
aligned path.

## Extreme Smoke/Timing

One-shot timing (`warmup=0`, `timed=1`) for `ai_very_high_extreme`
150000x16384x4096:

| Provider | Result |
| --- | ---: |
| native_hip | 535334.0 us, 37.61 TFLOPS |
| hipBLASLt | 285322.9 us, 70.56 TFLOPS |
| IREE | 265002.0 us, 75.97 TFLOPS |

`ai_very_high_extreme` has `M=150000`, which is not divisible by 128, so native_hip
uses `gemm_wmma_128x128_boundary`, not the aligned kernel profiled by ATT.

## Conclusion

The current aligned kernel is a reusable balance point because:

- `SMEM_PAD=8` is necessary for this LDS access pattern; smaller padding caused
  severe throughput losses.
- The 8-warp 32x64 warp-tile shape keeps VGPRs at 120 with no scratch and enough
  occupancy to hide part of the VMEM/LDS latency.
- Attempts to reduce VMEM pressure by vectorizing B loads created worse LDS
  behavior.
- Launch-bound and alpha-store changes did not improve the primary large
  surrogate and neighboring aligned shapes together.

The measured blocker is not a missing one-off dispatch rule; it is the staging
schedule around global loads, LDS handoff, and WMMA issue pressure in the reusable
aligned kernel. Further improvement likely needs a more substantial pipeline
change that preserves `SMEM_PAD=8` and avoids increasing LDS scatter or VGPR
pressure.
