# bd-1v2 Native HIP High-AI Software Pipeline Investigation

Date: 2026-04-22
Worktree: `/home/mahesh/kernelGen/kernelGen-bd-1v2`
Branch: `users/MaheshRavishankar/bd-1v2-softwarePipelineHighAi`
Build: `/home/mahesh/kernelGen/build/bd-1v2-Release`

## Summary

No reusable software-pipelined change was kept for
`gemm_wmma_128x128`. The best source state at the end of the run is the
baseline kernel from PR #10.

The pipeline attempts did reduce some sampled VMEM wait in ATT, but only by
raising VGPR pressure from 120 VGPRs to 144-168 VGPRs and shifting pressure to
WMMA/LDS waits or scratch/`vmcnt(0)` waits. The closest variants produced
noise-level movement on `ai_very_high_large` while regressing neighboring
aligned shapes, especially `ai_very_high_square` and `small_f16`. Because the
task explicitly called for a reusable strategy and no broad regressions, the
kernel source was restored and only this report should be committed.

## Baseline Provider Timings

Commands used the requested Release build directory and default runner timing
unless otherwise noted (`warmup=5`, `timed=20`). `small_f16` was verified.

| Test | native_hip | hipBLASLt | IREE |
| --- | ---: | ---: | ---: |
| `ai_very_high_large` | 4213.5 us, 80.63 TFLOPS | 4230.8 us, 80.30 TFLOPS | 4106.7 us, 82.73 TFLOPS |
| `ai_very_high_square` | 1800.2 us, 76.35 TFLOPS | 1773.5 us, 77.50 TFLOPS | 1833.3 us, 74.97 TFLOPS |
| `ai_very_high_medium` | 876.6 us, 77.51 TFLOPS | 887.2 us, 76.59 TFLOPS | 876.4 us, 77.53 TFLOPS |
| `small_f16` | 43.1 us, 49.78 TFLOPS, PASS | 37.4 us, 57.45 TFLOPS, PASS | 111.1 us, 19.32 TFLOPS, PASS |

Final native_hip timing after restoring the baseline source and rebuilding:

| Test | native_hip final |
| --- | ---: |
| `ai_very_high_large` | 4184.1 us, 81.20 TFLOPS |
| `ai_very_high_square` | 1790.4 us, 76.77 TFLOPS |
| `ai_very_high_medium` | 882.2 us, 77.02 TFLOPS |
| `small_f16` | 43.1 us, 49.79 TFLOPS, PASS |

Baseline `ai_very_high_large` profile after restore:

| Metric | Value |
| --- | ---: |
| Profile timing | 4253.4 us, 79.88 TFLOPS |
| Timing variance | 11.8% |
| VGPRs / SGPRs | 120 / 128 |
| LDS | 20480 bytes |
| Workgroup | 256 threads |
| VGPR-limited occupancy | 12 waves/SIMD |
| LDS bank conflicts | 100 total, 0.0/wave |

## Baseline ATT

Collected with:

```bash
/home/mahesh/kernelGen/TheRock/bin/rocprofv3 --att \
  --att-library-path /home/mahesh/kernelGen/TheRock/lib \
  --att-gpu-index 0 --att-simd-select 0 \
  --kernel-include-regex gemm_wmma_128x128 \
  --kernel-exclude-regex boundary --kernel-trace --stats \
  --att-buffer-size 0x6000000 \
  -d /tmp/kernelgen-bd-1v2/att/baseline-large -- \
  /home/mahesh/kernelGen/build/bd-1v2-Release/gemm/kernel_providers/native_hip/native_hip_gemm_bench \
  --config /home/mahesh/kernelGen/kernelGen-bd-1v2/gemm/tests/ai_very_high_large/config.json \
  --warmup 0 --timed 1
```

`analyze_att.py` summary for `ai_very_high_large`:

| Metric | Value |
| --- | ---: |
| ATT timing under rocprof | 5169.7 us |
| Wave files | 112 |
| Wave duration min/avg/max | 634569 / 966672 / 2068591 |
| Total latency / stall / idle | 63.05M / 44.47M / 45.35M |
| Occupancy samples | 7200 |

Top baseline stall classes:

| Class | Stall |
| --- | ---: |
| `waitcnt:vmcnt(3)` | 16.83M |
| LDS | 4.96M |
| `waitcnt:vmcnt(2)` | 4.47M |
| `waitcnt:vmcnt(10)` | 4.02M |
| WMMA | 3.79M |
| `waitcnt:vmcnt(4)` | 1.85M |
| `waitcnt:lgkmcnt(0)` | 1.80M |
| `waitcnt:lgkmcnt(6)` | 1.67M |

Top waitcnt sources were again the scalar B `global_load_u16` ladder and A
`global_load_b128` staging. This matched the bd-j5x reports: the exposed wait
is on global-to-LDS staging, not LDS bank conflicts.

## Pipeline Attempts

All source changes below were reverted.

| Strategy | Result | Decision |
| --- | --- | --- |
| Full-tile register prefetch using a helper struct/array | Large 7262.9 us / 46.78 TFLOPS; square 2836.1 / 48.46; medium 1269.3 / 53.53; `small_f16` 103.4 / 20.76 PASS | Rejected. ATT showed scratch traffic and massive `vmcnt(0)` waits. |
| Full-tile explicit scalar prefetch, no helper array | Large 5542.5 us / 61.30 TFLOPS; square 2285.8 / 60.13; medium 1158.4 / 58.66; `small_f16` 51.7 / 41.51 PASS | Rejected. Avoided the scratch cliff but still raised VGPRs to 168 and regressed all shapes. |
| B-only scalar prefetch for all 16 B rows | Large 4411.8 us / 77.01 TFLOPS; square 1847.0 / 74.41; medium 896.3 / 75.81; `small_f16` 44.8 / 47.90 PASS | Rejected. Reduced sampled VMEM wait but raised VGPRs to 168 and slowed every requested shape. |
| B-only scalar prefetch for 8 B rows | Large 4215.3 us / 80.60 TFLOPS; square 1845.6 / 74.47; medium 863.6 / 78.68; `small_f16` 45.2 / 47.55 PASS | Rejected. Medium improved, but square and small regressed; VGPRs were still 144. |
| B-only scalar prefetch for 4 B rows | Large 4208.2 us / 80.73 TFLOPS; square 1815.7 / 75.69; medium 876.1 / 77.56; `small_f16` 44.6 / 48.11 PASS | Rejected. Large/medium movement was noise-level and square/small still regressed; VGPRs were still 144. |

## After-Change ATT

The most informative after-change ATT was the all-B scalar prefetch variant:

| Metric | Baseline | B-only prefetch |
| --- | ---: | ---: |
| ATT timing under rocprof | 5169.7 us | 5262.6 us |
| Wave files | 112 | 112 |
| Wave duration min/avg/max | 634569 / 966672 / 2068591 | 417798 / 647386 / 1580650 |
| Total latency / stall / idle | 63.05M / 44.47M / 45.35M | 42.25M / 28.66M / 30.36M |
| Profile VGPRs | 120 | 168 |
| VGPR-limited occupancy | 12 waves/SIMD | 9 waves/SIMD |

Top B-only prefetch stall classes:

| Class | Stall |
| --- | ---: |
| `waitcnt:vmcnt(1)` | 7.89M |
| `waitcnt:vmcnt(0)` | 6.90M |
| WMMA | 4.67M |
| `waitcnt:lgkmcnt(0)` | 3.76M |
| LDS | 2.37M |
| VALU | 1.98M |

Interpretation: the staged B prefetch did hide part of the original
`vmcnt(3)`/`vmcnt(2)` pattern, but it did not improve end-to-end timing. The
register footprint cut the VGPR-limited occupancy from 12 to 9 waves/SIMD and
shifted the bottleneck into lower-count VMEM waits plus WMMA/LDS waits. Smaller
partial-B prefetches lowered VGPRs to 144 but still did not preserve the
neighboring aligned shapes.

The full-tile helper-struct variant was worse. Its ATT showed 111.51M stall
cycles, dominated by `waitcnt:vmcnt(0)` at 90.64M, with visible
`scratch_load_b128` and `scratch_store_b128` classes. That confirms the
register tile spilled and turned the intended software pipeline into extra
VMEM pressure.

## Extreme Smoke

One-shot native_hip timing for `ai_very_high_extreme` (`warmup=0`, `timed=1`):

| Test | native_hip |
| --- | ---: |
| `ai_very_high_extreme` | 533773.0 us, 37.72 TFLOPS |

Trace profiling reported:

```text
void gemm_wmma_128x128_boundary<__hip_bfloat16>(...)
```

This is expected because `M=150000` is not divisible by 128, so the test still
dispatches the boundary path rather than the aligned `gemm_wmma_128x128` path.

## Conclusion

The measured blocker is the cost of keeping enough next-tile data live to hide
global/LDS staging latency. Full-tile prefetch either spills or raises VGPRs to
168\. Partial B prefetch lowers that to 144 VGPRs but still regresses neighboring
aligned shapes and does not produce a defensible large-shape improvement.

No dispatch rule is justified by these measurements. A future attempt would
need a different schedule that reduces exposed VMEM wait without pushing the
aligned kernel above the current 120 VGPR balance point, or a larger tile/split
redesign that is evaluated as a new reusable high-AI path rather than a
shape-specific exception.
