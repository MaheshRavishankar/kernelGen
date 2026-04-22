# bd-huc ATT-Driven Aligned High-AI WMMA Tuning

Date: 2026-04-22
Worktree: `/home/mahesh/kernelGen/kernelGen-bd-huc`
Branch: `users/MaheshRavishankar/bd-huc-attProfileAlignedHighAi`
Build: `/home/mahesh/kernelGen/build/bd-huc-att-Release`

## Summary

No reusable source change was kept for `gemm_wmma_128x128`.

Fresh ATT traces for `ai_very_high_square` and `ai_very_high_large` identify the
same bottleneck family: exposed VMEM waitcnt stalls on the global-to-LDS staging
path, especially the scalar B `global_load_u16` ladder and A `global_load_b128`
staging. LDS and WMMA stalls are secondary. Standard rocprof counters show the
kernel remains at 120 VGPRs, 20480 bytes LDS, and effectively zero LDS bank
conflicts per wave, so the evidence does not support a simple padding or
dispatch-specialization fix.

Two ATT-supported variants were measured: `__launch_bounds__(BLOCK_SIZE, 2)` and
a small four-row B prefetch. Both helped `ai_very_high_large` in one direct pass
but regressed neighboring aligned shapes or `small_f16`, so both were reverted.
The final branch keeps the current 8-warp 128x128 `SMEM_PAD=8` balance-point
kernel and adds this report.

## Build Notes

The requested Release build directory was configured with only the providers
needed for this bead because this checkout does not have the Fusilli source at
the default path:

```bash
cmake -S . -B /home/mahesh/kernelGen/build/bd-huc-att-Release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=/usr/bin/ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/ccache \
  -DGPU_TARGETS=gfx1100 \
  -DKERNELGEN_ENABLE_GEMM=ON \
  -DKERNELGEN_ENABLE_HIPBLASLT=ON \
  -DKERNELGEN_ENABLE_NATIVE_HIP=ON \
  -DKERNELGEN_ENABLE_FUSILLI=OFF \
  -DKERNELGEN_ENABLE_IREE=OFF

cmake --build /home/mahesh/kernelGen/build/bd-huc-att-Release \
  --target native_hip_gemm_bench hipblaslt_gemm_bench
```

## Baseline Timings

Commands used default runner timing, `warmup=5` and `timed=20`. `--verify` was
passed for all runs; only `small_f16` has reference data and reported PASS.

Native HIP baseline pass 1:

| Test | Time | TFLOPS | Verify |
| --- | ---: | ---: | --- |
| `ai_very_high_square` | 1827.1 us | 75.22 | n/a |
| `ai_very_high_medium` | 835.2 us | 81.35 | n/a |
| `ai_very_high_large` | 4973.9 us | 68.30 | n/a |
| `small_f16` | 42.6 us | 50.43 | PASS |

Native HIP baseline pass 2:

| Test | Time | TFLOPS | Verify |
| --- | ---: | ---: | --- |
| `ai_very_high_square` | 1854.4 us | 74.11 | n/a |
| `ai_very_high_medium` | 824.6 us | 82.40 | n/a |
| `ai_very_high_large` | 4961.3 us | 68.48 | n/a |
| `small_f16` | 42.9 us | 50.01 | PASS |

hipBLASLt baseline pass 1:

| Test | Time | TFLOPS | Verify |
| --- | ---: | ---: | --- |
| `ai_very_high_square` | 1794.7 us | 76.58 | n/a |
| `ai_very_high_medium` | 6240.2 us | 10.89 | n/a |
| `ai_very_high_large` | 4273.4 us | 79.50 | n/a |
| `small_f16` | 37.1 us | 57.88 | PASS |

hipBLASLt baseline pass 2:

| Test | Time | TFLOPS | Verify |
| --- | ---: | ---: | --- |
| `ai_very_high_square` | 1783.6 us | 77.06 | n/a |
| `ai_very_high_medium` | 5384.5 us | 12.62 | n/a |
| `ai_very_high_large` | 4342.9 us | 78.23 | n/a |
| `small_f16` | 37.2 us | 57.77 | PASS |

The hipBLASLt medium result was slow in both repeated passes on this build. It
is recorded as measured, not used as evidence for a native HIP source change.

## Baseline Counter Profiles

Standard rocprof profiles were collected for the two ATT-target shapes.

| Test | Profile Time | TFLOPS | Variance | VGPR | SGPR | LDS | Waves | LDS Conflicts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ai_very_high_square` | 1716.7 us | 80.06 | 14.3% | 120 | 128 | 20480 B | 8192 | 0 |
| `ai_very_high_large` | 4305.5 us | 78.91 | 14.4% | 120 | 128 | 20480 B | 21600 | 100 total, 0.0/wave |

Resource-derived occupancy from `analyze.py`:

| Test | VGPR-limited | LDS-limited | Workgroup | Grid |
| --- | ---: | ---: | ---: | --- |
| `ai_very_high_square` | 12 waves/SIMD | 24 waves/CU | 256 threads | `[8192, 32, 1]` |
| `ai_very_high_large` | 12 waves/SIMD | 24 waves/CU | 256 threads | `[23040, 30, 1]` |

These counters rule out LDS bank conflicts as the primary issue. The kernel is
roofline compute-bound by arithmetic intensity, but ATT shows the compute pipe
is still fed through exposed VMEM/LDS waits.

## ATT Collection

ATT was collected with one timed dispatch to control trace size:

```bash
/home/mahesh/kernelGen/TheRock/bin/rocprofv3 --att \
  --att-library-path /home/mahesh/kernelGen/TheRock/lib \
  --att-gpu-index 0 --att-simd-select 0 \
  --kernel-include-regex gemm_wmma_128x128 \
  --kernel-exclude-regex boundary \
  --kernel-trace --stats --att-buffer-size 0x6000000 \
  -d /tmp/kernelgen-bd-huc-att/base-large -- \
  /home/mahesh/kernelGen/build/bd-huc-att-Release/gemm/kernel_providers/native_hip/native_hip_gemm_bench \
  --config /home/mahesh/kernelGen/kernelGen-bd-huc/gemm/tests/ai_very_high_large/config.json \
  --warmup 0 --timed 1
```

Summaries were generated with:

```bash
.venv/bin/python gemm/profiling/analyze_att.py \
  /tmp/kernelgen-bd-huc-att/base-square/ui_output_agent_19984_dispatch_2 --top 20

.venv/bin/python gemm/profiling/analyze_att.py \
  /tmp/kernelgen-bd-huc-att/base-large/ui_output_agent_63558_dispatch_2 --top 20
```

## ATT Bottlenecks

### `ai_very_high_square`

ATT timing under rocprof: 3107.4 us.

| Metric | Value |
| --- | ---: |
| Wave files | 44 |
| Wave duration min/avg/max | 757640 / 1182698 / 1757956 |
| Total latency / stall / idle | 33.31M / 26.01M / 18.78M |
| Occupancy samples | 2720 |

Top stall classes:

| Class | Stall |
| --- | ---: |
| `waitcnt:vmcnt(3)` | 13.09M |
| `waitcnt:vmcnt(10)` | 2.87M |
| `waitcnt:vmcnt(2)` | 2.22M |
| LDS | 1.94M |
| WMMA | 1.36M |
| `waitcnt:vmcnt(8)` | 1.04M |
| `waitcnt:lgkmcnt(0)` | 0.75M |
| `waitcnt:lgkmcnt(6)` | 0.69M |
| VALU | 0.50M |
| VMEM instructions | 0.08M |

Top waitcnt sources were scalar B `global_load_u16` rows and A
`global_load_b128` rows. Store-side VMEM is not the visible limiter.

### `ai_very_high_large`

ATT timing under rocprof: 5155.5 us.

| Metric | Value |
| --- | ---: |
| Wave files | 112 |
| Wave duration min/avg/max | 559638 / 982973 / 2231314 |
| Total latency / stall / idle | 65.08M / 44.67M / 45.15M |
| Occupancy samples | 7200 |

Top stall classes:

| Class | Stall |
| --- | ---: |
| `waitcnt:vmcnt(3)` | 19.34M |
| LDS | 4.93M |
| `waitcnt:vmcnt(10)` | 3.92M |
| WMMA | 3.81M |
| `waitcnt:vmcnt(2)` | 2.72M |
| `waitcnt:lgkmcnt(0)` | 1.86M |
| `waitcnt:lgkmcnt(6)` | 1.68M |
| VALU | 1.53M |
| `waitcnt:vmcnt(6)` | 1.33M |
| `waitcnt:vmcnt(8)` | 1.29M |

Top waitcnt sources again came from scalar B `global_load_u16` rows and the A
`global_load_b128` pair before LDS stores. The dominant `vmcnt(3)` wait is the
global-to-LDS staging handoff; `lgkmcnt` waits and LDS/WMMA stalls are secondary
after data reaches LDS. LDS bank conflict counters remain effectively zero, so
this is a latency/scheduling problem rather than a shared-memory conflict fix.

## Attempted Changes

All source variants below were reverted.

| Strategy | Timing Result | Decision |
| --- | --- | --- |
| `__launch_bounds__(BLOCK_SIZE, 2)` for the aligned kernel | Square 1867.9 us / 73.58 TFLOPS; medium 902.4 us / 75.30; large 4218.6 us / 80.53; `small_f16` 43.1 us / 49.82 PASS | Rejected. Fresh measurement did not justify reviving the old bd-huc launch-bound idea: large improved in this direct pass, but square and medium regressed. |
| Prefetch first four B rows before A staging, then store packed values to LDS | Square 1811.2 us / 75.88 TFLOPS; medium 864.0 us / 78.64; large 4219.8 us / 80.51; `small_f16` 44.7 us / 48.01 PASS | Rejected. It targeted the ATT B-load ladder and kept the profile at 120 VGPRs / 20480 B LDS, but it still regressed `small_f16` and did not produce a broad aligned-shape win. |

The B-prefetch large profile measured 4296.1 us, 79.08 TFLOPS, 15.3% variance,
120 VGPRs, 128 SGPRs, and 20480 bytes LDS. That is effectively the same resource
balance as baseline, so the direct large improvement is not enough to justify
the neighboring regressions.

## Final Timings

After reverting the rejected variants and rebuilding `native_hip_gemm_bench`:

| Test | Time | TFLOPS | Verify |
| --- | ---: | ---: | --- |
| `ai_very_high_square` | 1805.5 us | 76.12 | n/a |
| `ai_very_high_medium` | 866.8 us | 78.39 | n/a |
| `ai_very_high_large` | 4235.6 us | 80.21 | n/a |
| `small_f16` | 43.3 us | 49.64 | PASS |

## Conclusion

The aligned high-AI kernel remains limited by exposed global-load staging
latency, followed by LDS and WMMA feed pressure. The current `SMEM_PAD=8`,
8-warp, 128x128 kernel remains the best reusable balance point measured in this
run. No dispatch change is justified: all requested BF16 target shapes already
use the aligned kernel, and the rejected variants did not improve the target set
as a group.

Future work should focus on a schedule that reduces the dominant `vmcnt` waits
without increasing register pressure or regressing `small_f16`; simple launch
bound changes, padding changes, and small B prefetches are not sufficient based
on the measurements above.
