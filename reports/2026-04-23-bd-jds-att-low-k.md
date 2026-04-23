# bd-jds ATT-Driven Low-K Native HIP GEMM Strategy

Date: 2026-04-23
Worktree: `/home/mahesh/kernelGen/kernelGen-bd-jds`
Branch: `users/MaheshRavishankar/bd-jds-attProfileLowKStrategy`
Build: `/home/mahesh/kernelGen/build/bd-jds-att-Release`

## Summary

The existing boundary WMMA path is not the right balance point for non-skinny
`K <= 32` memory-bound BF16/F16 GEMMs. ATT shows that these shapes spend most of
their sampled time in VMEM/LDS waits and idle wave time, not useful WMMA issue.
For `K=10` and `K=20`, boundary WMMA also pads work to `BK=32`, uses a 20 KB
LDS tile, keeps 144 VGPRs live, and stores C through the WMMA lane layout.

This branch adds one reusable low-K streaming kernel:

- output tile: `16x128`
- workgroup: 256 threads
- per-thread output vector: 8 contiguous columns
- selector: `M > 32 && K <= 32`

`M <= 32` stays on the existing skinny-M WMMA kernel. `K` around 128 stays on
WMMA; the `ai_medium_extreme` evidence is a memory-bound but well-filled
boundary WMMA case, not the low-K launch/padding failure mode.

## Baseline Timings

Commands:

```bash
cmake -S . -B /home/mahesh/kernelGen/build/bd-jds-att-Release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1100 \
  -DKERNELGEN_ENABLE_GEMM=ON -DKERNELGEN_ENABLE_HIPBLASLT=ON \
  -DKERNELGEN_ENABLE_NATIVE_HIP=ON -DKERNELGEN_ENABLE_FUSILLI=OFF \
  -DKERNELGEN_ENABLE_IREE=OFF
cmake --build /home/mahesh/kernelGen/build/bd-jds-att-Release \
  --target native_hip_gemm_bench hipblaslt_gemm_bench
```

Fusilli was disabled for this build because the configured checkout path
`/home/mahesh/iree/fusilli` is absent. The target providers for this bead are
native HIP and hipBLASLt.

Default runner timing (`warmup=5`, `timed=20`):

| Test | Shape | Native HIP baseline | hipBLASLt baseline |
| --- | ---: | ---: | ---: |
| `ai_very_low_small_square` | 576x576x10 | 13.564 us, 0.49 TFLOPS, PASS | 7.520 us, 0.88 TFLOPS, PASS |
| `ai_very_low_small_wide` | 576x2304x10 | 21.326 us, 1.24 TFLOPS, PASS | 12.570 us, 2.11 TFLOPS, PASS |
| `ai_low_large_flat` | 21760x3840x20 | 1608.800 us, 2.08 TFLOPS | 592.320 us, 5.64 TFLOPS |
| `ai_medium_extreme` | 16800000x128x134 | 29423.200 us, 19.59 TFLOPS | 24811.790 us, 23.23 TFLOPS |

## Current Boundary WMMA ATT Evidence

ATT was collected with `warmup=0`, `timed=1`, `--att-simd-select 0`, and
`--kernel-include-regex gemm_wmma_128x128_boundary`. `analyze_att.py` was run on
the generated `ui_output_agent_*_dispatch_*` directories.

| Test | Sampled wave duration avg | Total stall | Total idle | Dominant stalls |
| --- | ---: | ---: | ---: | --- |
| `ai_low_large_flat` | 303898 | 10.98M | 48.57M | `vmcnt(0)` 5.13M, VMEM 3.12M, `vmcnt(1)` 1.97M |
| `ai_very_low_small_wide` | 35197 | 27.1K | 92.1K | `lgkmcnt(0)` 14.1K, `vmcnt(0)` 9.9K |
| `ai_medium_extreme` | 102501 | 265.7M | 248.7M | `vmcnt(0)` 197.4M, `vmcnt(1)` 40.8M, VALU 12.6M |

For `ai_low_large_flat`, the largest wait source was the global-load-to-LDS
staging sequence, followed by many `global_store_d16_hi_b16` C stores in the top
VMEM rows. Standard rocprof counters showed 144 VGPRs, 20 KB LDS, 40,800
waves, about 7,237 cycles/wave, and effectively zero LDS bank conflicts.
Extended gfx1100 PMCs reported zero for `MemUnitBusy`, `WRITE_SIZE`,
`FETCH_SIZE`, `GL2C_*`, and `MeanOccupancyPerActiveCU`, so memory-system
conclusions here use ATT wait sources plus derived traffic.

The low-K launch geometry was also weak:

| Test | Boundary WMMA workgroups | Boundary waves |
| --- | ---: | ---: |
| `ai_very_low_small_square` | 25 | 200 |
| `ai_very_low_small_wide` | 90 | 720 |
| `ai_low_large_flat` | 5100 | 40800 |

The small cases are launch/underfill dominated. The large-flat case has enough
workgroups, but still pays boundary WMMA padding, scattered stores, and exposed
VMEM waits.

## Strategy Decision

Extending the boundary WMMA path was not the best low-K fix. A smaller `BK=16`
boundary variant would reduce some K padding for `K=10`, but it would keep the
WMMA fragment store layout, boundary branch structure, and small-shape
underfill. It also would not address `K=20`, which still needs two WMMA K steps.

A scalar/vectorized streaming dot kernel is justified for `K <= 32` because it:

- avoids WMMA K padding for `K=10` and `K=20`
- reduces resources from 144 VGPRs / 20 KB LDS to 24 VGPRs / 9 KB LDS
- increases low-K launch parallelism
- writes contiguous 128-bit C vectors
- keeps the kernel count to one additional reusable low-K path

The selector intentionally does not include `K=134`. `ai_medium_extreme` has
131250 boundary WMMA workgroups and direct timing only about 19% slower than
hipBLASLt in this run. Its ATT is VMEM-dominated, but not launch/underfill or
K-padding dominated enough to justify scalar dot work at K around 128.

## Attempted Low-K Variants

All variants used one low-K streaming kernel and the same `M > 32 && K <= 32`
selector while testing tile constants.

| Variant | Square K10 | Wide K10 | Large-flat K20 | Decision |
| --- | ---: | ---: | ---: | --- |
| `BM16 BN128 VEC8`, 256 threads | 5.596 us, 1.19 TFLOPS | 7.742 us, 3.43 TFLOPS | 697.807 us, 4.79 TFLOPS | Selected. Best overall balance and best on two low-K targets. |
| `BM16 BN256 VEC16`, 256 threads | 8.092 us, 0.82 TFLOPS | 8.116 us, 3.27 TFLOPS | 668.583 us, 5.00 TFLOPS | Rejected. Better large-flat, but regressed both small K10 targets. |
| `BM16 BN128 VEC16`, 128 threads | 6.028 us, 1.10 TFLOPS | 8.320 us, 3.19 TFLOPS | 707.864 us, 4.72 TFLOPS | Rejected. Only improved small square. |

Old PR #6 was used only as historical context. Its `BN256/VEC16` balance point
was remeasured above and not used as the final selector.

## Final Timings

Default native HIP runner timing (`warmup=5`, `timed=20`):

| Test | Baseline native HIP | Final native HIP | hipBLASLt baseline |
| --- | ---: | ---: | ---: |
| `ai_very_low_small_square` | 13.564 us, 0.49 TFLOPS | 5.596 us, 1.19 TFLOPS, PASS | 7.520 us, 0.88 TFLOPS, PASS |
| `ai_very_low_small_wide` | 21.326 us, 1.24 TFLOPS | 7.742 us, 3.43 TFLOPS, PASS | 12.570 us, 2.11 TFLOPS, PASS |
| `ai_low_large_flat` | 1608.800 us, 2.08 TFLOPS | 697.807 us, 4.79 TFLOPS | 592.320 us, 5.64 TFLOPS |
| `ai_medium_extreme` | 29423.200 us, 19.59 TFLOPS | 29517.600 us, 19.52 TFLOPS | 24811.790 us, 23.23 TFLOPS |

The final low-K kernel beats hipBLASLt on the two small K10 tests and closes most
of the large-flat gap, but hipBLASLt remains faster on `ai_low_large_flat`.
`ai_medium_extreme` remains on boundary WMMA; the 0.3% timing difference is
within run noise and is not treated as a material regression.

## Final Profile/ATT Check

Standard rocprof on `ai_low_large_flat` identified the final main kernel as
`gemm_low_k_stream<__hip_bfloat16>`:

| Metric | Boundary baseline | Final low-K |
| --- | ---: | ---: |
| VGPRs | 144 | 24 |
| LDS bytes/WG | 20480 | 9216 |
| SQ_WAVES | 40800 | 326400 |
| Cycles/wave | 7237 | 244 |
| LDS bank conflicts/wave | ~0 | ~0 |

Final ATT for `ai_low_large_flat`:

| Metric | Value |
| --- | ---: |
| Wave files | 1698 |
| Wave duration min/avg/max | 4688 / 22809 / 48999 |
| Total latency / stall / idle | 34.48M / 18.62M / 4.26M |
| Top stalls | `vmcnt(0)` 14.10M, `lgkmcnt(1)` 2.56M, VALU 1.02M |

Final ATT for `ai_very_low_small_wide`:

| Metric | Value |
| --- | ---: |
| Wave files | 28 |
| Wave duration min/avg/max | 3388 / 8910 / 13007 |
| Total latency / stall / idle | 151.6K / 115.9K / 98.0K |
| Top stalls | SMEM 31.7K, `vmcnt(0)` 28.4K, `lgkmcnt(0)` 27.5K |

The final bottleneck is still memory feed and LDS handoff, but the work is now
spread across shorter waves with much lower resource pressure and contiguous C
stores. This matches the selector rationale: use streaming scalar/vectorized dot
for very low-K memory-bound GEMMs, keep WMMA for larger K.

## Validation

Commands run:

```bash
cmake --build /home/mahesh/kernelGen/build/bd-jds-att-Release \
  --target native_hip_gemm_bench hipblaslt_gemm_bench

.venv/bin/python gemm/kernel_providers/native_hip/run.py \
  --build-dir /home/mahesh/kernelGen/build/bd-jds-att-Release \
  --test gemm/tests/ai_very_low_small_square \
         gemm/tests/ai_very_low_small_wide \
         gemm/tests/ai_low_large_flat \
         gemm/tests/ai_medium_extreme \
  --verify -o /tmp/kernelgen-bd-jds-att/native_final.json

/home/mahesh/kernelGen/build/bd-jds-att-Release/gemm/kernel_providers/native_hip/native_hip_gemm_bench \
  --config gemm/tests/ai_very_low_small_square/config.json \
  --input-a gemm/tests/ai_very_low_small_square/input_a.npy \
  --input-b gemm/tests/ai_very_low_small_square/input_b.npy \
  --reference gemm/tests/ai_very_low_small_square/output_c.npy \
  --warmup 5 --timed 20 --use-dispatch

/home/mahesh/kernelGen/build/bd-jds-att-Release/gemm/kernel_providers/native_hip/native_hip_gemm_bench \
  --config gemm/tests/ai_medium_extreme/config.json \
  --warmup 1 --timed 3 --use-dispatch
```

Results:

- native HIP target test run: 4 passed, 0 failed
- reference correctness: PASS for `ai_very_low_small_square` and
  `ai_very_low_small_wide`
- C dispatch smoke: low-K verified PASS, K=134 boundary path launched
  successfully
