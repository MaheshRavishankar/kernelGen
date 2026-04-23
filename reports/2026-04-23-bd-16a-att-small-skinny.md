# bd-16a ATT Small/Skinny Native HIP GEMM Investigation

This report records fresh measurements for bead `bd-16a` on branch
`users/MaheshRavishankar/bd-16a-attProfileSmallSkinny`. Old PR #9 was treated
only as historical context; no conclusion below relies on it as completion
evidence.

Build directory:

```bash
/home/mahesh/kernelGen/build/bd-16a-att-Release
```

Raw result/analyzer JSON files were generated during the run and summarized
here. Raw ATT UI directories were temporary files under
`/tmp/kernelgen-bd-16a-att/` and are not checked in.

## Target Shapes

| Test | Shape | Native baseline path |
| --- | ---: | --- |
| `ai_high_small` | 576x576x1280 | `gemm_wmma_128x128_boundary` |
| `ai_low_small` | 32x576x2304 | `gemm_wmma_skinny` |
| `ai_low_skinny` | 16x512x1024 | `gemm_wmma_skinny` |

## Baselines

Default runner settings (`warmup=5`, `timed=20`), verification enabled.

| Test | native_hip | hipBLASLt |
| --- | ---: | ---: |
| `ai_high_small` | 80.85 us, 10.50 TFLOPS | 27.44 us, 30.95 TFLOPS |
| `ai_low_small` | 32.75 us, 2.59 TFLOPS | 14.52 us, 5.85 TFLOPS |
| `ai_low_skinny` | 17.11 us, 0.98 TFLOPS | 10.00 us, 1.68 TFLOPS |

hipBLASLt trace kernel names showed `MT64x96x32` for `ai_high_small` and
`MT64x16x64` for the two low-M shapes.

## ATT Bottleneck Model

ATT traces were collected for all three target native HIP shapes with
`rocprofv3 --att` and summarized with `gemm/profiling/analyze_att.py`.

| Test | Kernel | Wave duration avg | Dominant stalls |
| --- | --- | ---: | --- |
| `ai_high_small` | `128x128_boundary` | 285078 | `vmcnt(0/1)` 333.8k, `lgkmcnt` 27.3k |
| `ai_low_small` | `skinny` | 131987 | `vmcnt(0/1)` 137.2k, `lgkmcnt` 21.2k, LDS 2.6k |
| `ai_low_skinny` | `skinny` | 65404 | `vmcnt(0/1)` 63.9k, `lgkmcnt` 13.8k, LDS 1.7k |

Detailed interpretation:

- `vmcnt` dominates every trace. Top wait sources are global A `b128` loads and
  scalar B `u16` loads.
- `lgkmcnt` is secondary. Direct LDS stalls and PMC LDS bank conflicts are low:
  baseline high-small had 0 bank conflicts, low-small had 0.49 conflicts/wave,
  and final high-small had 0.15 conflicts/wave.
- WMMA and VALU are not primary blockers in ATT. WMMA issue stalls were zero in
  these traces; VALU stalls were below 1k.
- `ai_high_small` has launch underfill: the 128x128 boundary selector launches
  25 CTAs, below the 48 CUs on gfx1100, with 200 total waves. Its tile envelope
  covers 640x640 output elements for a 576x576 problem, about 19% output-tile
  compute waste.
- The skinny path launches 36 CTAs/288 waves for `ai_low_small` and 32 CTAs/256
  waves for `ai_low_skinny`. It has high OOB compute waste (75% and 87.5%), but
  the ATT model says exposed memory latency is still the stronger limit.

## Attempted Changes

| Change | Result | Decision |
| --- | ---: | --- |
| Change skinny from `BM=128, BN=16, 8 warps` to `BM=64, BN=16, 4 warps` | `ai_low_small` regressed to 39.82 us; `ai_low_skinny` regressed to 20.28 us | Rejected. Lower OOB work lost too much latency hiding. |
| Add small boundary `64x64`, 4 warps | `ai_high_small` improved to 67.27 us in the first verified run; skinny targets unchanged | Accepted. |
| Try small boundary `64x128` | `ai_high_small` regressed to 118.49 us | Rejected. |
| Try small boundary `64x96` | `ai_high_small` measured 71.66 us | Rejected. It matches the hipBLASLt tile dimensions, but this native implementation was slower than `64x64`. |

## Final Selector Rationale

The final change adds one reusable small-boundary kernel,
`gemm_wmma_64x64_boundary`, and leaves the existing skinny kernel in place.

Selector order:

1. `M <= 32`: keep `gemm_wmma_skinny`.
1. 128-aligned shapes: keep `gemm_wmma_128x128`.
1. `M <= 640 && N <= 640 && K >= 128`: use `gemm_wmma_64x64_boundary`.
1. Everything else: keep `gemm_wmma_128x128_boundary`.

This keeps total kernel count low and limits the new path to the measured
small, non-128-aligned boundary case. For `ai_high_small`, the new selector
changes the launch from 25 CTAs/200 waves to 81 CTAs/324 waves, reduces LDS per
CTA from 20480 to 10240 bytes, and removes output-tile OOB waste for 576x576.

Final ATT on `ai_high_small` with `64x64`:

| Metric | Baseline `128x128_boundary` | Final `64x64_boundary` |
| --- | ---: | ---: |
| Wave duration avg | 285078 | 260835 |
| Total stall | 362.4k | 348.1k |
| Total idle | 168.2k | 140.3k |
| `lgkmcnt` stall | 27.3k | 15.5k |
| Direct LDS stall | 0.2k | 0.4k |
| WMMA stall | 0 | 0 |
| SQ waves | 200 | 324 |

The remaining bottleneck is still global-load wait (`vmcnt`), so this is an
underfill/grid-parallelism improvement, not a full memory-scheduling fix.

## Final Timings

Default runner settings, verification enabled. Two native runs are shown because
the small kernels have visible run-to-run variance.

| Test | Baseline native | Final native run 1 | Final native run 2 | Final hipBLASLt |
| --- | ---: | ---: | ---: | ---: |
| `ai_high_small` | 80.85 us, 10.50 TFLOPS | 69.93 us, 12.14 TFLOPS | 66.88 us, 12.70 TFLOPS | 25.74 us, 33.00 TFLOPS |
| `ai_low_small` | 32.75 us, 2.59 TFLOPS | 32.64 us, 2.60 TFLOPS | 32.68 us, 2.60 TFLOPS | 14.60 us, 5.82 TFLOPS |
| `ai_low_skinny` | 17.11 us, 0.98 TFLOPS | 17.10 us, 0.98 TFLOPS | 17.06 us, 0.98 TFLOPS | 10.06 us, 1.67 TFLOPS |

## Validation

Commands run:

```bash
cmake -S . -B /home/mahesh/kernelGen/build/bd-16a-att-Release -G Ninja \
  -DCMAKE_INSTALL_PREFIX=/home/mahesh/kernelGen/install/bd-16a-att-Release \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=/usr/bin/ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/ccache \
  -DGPU_TARGETS=gfx1100 -DKERNELGEN_ENABLE_GEMM=ON \
  -DKERNELGEN_ENABLE_HIPBLASLT=ON -DKERNELGEN_ENABLE_NATIVE_HIP=ON \
  -DKERNELGEN_ENABLE_FUSILLI=OFF -DKERNELGEN_ENABLE_IREE=OFF
cmake --build /home/mahesh/kernelGen/build/bd-16a-att-Release \
  --target native_hip_gemm_bench hipblaslt_gemm_bench
python3 gemm/kernel_providers/native_hip/run.py --build-dir /home/mahesh/kernelGen/build/bd-16a-att-Release \
  --test gemm/tests/ai_high_small gemm/tests/ai_low_small gemm/tests/ai_low_skinny --verify
python3 gemm/kernel_providers/hipblaslt/run.py --build-dir /home/mahesh/kernelGen/build/bd-16a-att-Release \
  --test gemm/tests/ai_high_small gemm/tests/ai_low_small gemm/tests/ai_low_skinny --verify
python3 gemm/tests/run_all.py --provider native_hip --verify \
  --build-dir /home/mahesh/kernelGen/build/bd-16a-att-Release
```

Results:

- Target native HIP tests: 3 passed, 0 failed.
- Target hipBLASLt baselines: 3 passed, 0 failed.
- Full native HIP verification sweep: 17 passed, 0 failed.

No target regression was measured. The final change improves only
`ai_high_small`; the current skinny path remains the measured best native HIP
choice among the tested low-M variants.
