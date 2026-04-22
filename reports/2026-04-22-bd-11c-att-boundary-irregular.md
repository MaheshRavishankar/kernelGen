# bd-11c Native HIP Boundary WMMA ATT Tuning

Date: 2026-04-22
Worktree: `/home/mahesh/kernelGen/kernelGen-bd-11c`
Branch: `users/MaheshRavishankar/bd-11c-attProfileBoundaryIrregular`
Build: `/home/mahesh/kernelGen/build/bd-11c-att-Release`

## Summary

The final change extends the existing reusable
`gemm_wmma_128x128_boundary` kernel rather than adding another kernel variant.
Full in-bounds tiles now use unchecked A/B load and C store paths, and partial
final K tiles skip WMMA slices that would consume zero-filled operands.

This directly targets the ATT baseline bottleneck: the generic boundary kernel
was spending most recorded stall time in VMEM waitcnts and boundary-address
work even for tiles where M, N, and most K blocks were fully in bounds. The
change keeps one boundary kernel for arbitrary shapes while making the common
full-tile path cheaper.

No selector change is included. A measured M-tail split selector was rejected:
it slightly improved `ai_very_high_extreme`, but it regressed
`ai_high_medium` and did not improve the full target set.

## Baseline Timings

Command shape:

```bash
.venv/bin/python gemm/kernel_providers/native_hip/run.py \
  --build-dir /home/mahesh/kernelGen/build/bd-11c-att-Release \
  --test gemm/tests/ai_very_high_extreme gemm/tests/ai_high_large_k \
         gemm/tests/ai_high_medium gemm/tests/ai_high_small \
         gemm/tests/ai_medium_large gemm/tests/ai_medium_small \
  --verify -o /tmp/bd-11c-native-baseline.json

.venv/bin/python gemm/kernel_providers/hipblaslt/run.py \
  --build-dir /home/mahesh/kernelGen/build/bd-11c-att-Release \
  --test gemm/tests/ai_very_high_extreme gemm/tests/ai_high_large_k \
         gemm/tests/ai_high_medium gemm/tests/ai_high_small \
         gemm/tests/ai_medium_large gemm/tests/ai_medium_small \
  --verify -o /tmp/bd-11c-hipblaslt-baseline.json
```

| Test | Shape | Native baseline | hipBLASLt baseline |
| --- | ---: | ---: | ---: |
| `ai_very_high_extreme` | 150000x16384x4096 | 536179.0 us, 37.55 TFLOPS | 310442.9 us, 64.85 TFLOPS |
| `ai_high_large_k` | 4096x1024x150000 | 34654.5 us, 36.31 TFLOPS | 20385.0 us, 61.73 TFLOPS |
| `ai_high_medium` | 1285x2048x3840 | 465.0 us, 43.47 TFLOPS, PASS | 380.4 us, 53.14 TFLOPS, PASS |
| `ai_high_small` | 576x576x1280 | 80.6 us, 10.53 TFLOPS, PASS | 26.8 us, 31.73 TFLOPS, PASS |
| `ai_medium_large` | 7680x512x304 | 116.4 us, 20.54 TFLOPS, PASS | 50.7 us, 47.15 TFLOPS, PASS |
| `ai_medium_small` | 576x576x165 | 19.8 us, 5.54 TFLOPS, PASS | 9.8 us, 11.23 TFLOPS, PASS |

## Baseline ATT Bottlenecks

ATT traces were collected with one timed dispatch:

```bash
/home/mahesh/kernelGen/TheRock/bin/rocprofv3 --att \
  --att-library-path /home/mahesh/kernelGen/TheRock/lib \
  --att-gpu-index 0 --att-simd-select 0 \
  --kernel-include-regex gemm_wmma_128x128_boundary \
  --kernel-trace --stats --att-buffer-size 0x6000000 \
  -d /tmp/kernelgen-bd-11c-att/<case> -- \
  /home/mahesh/kernelGen/build/bd-11c-att-Release/gemm/kernel_providers/native_hip/native_hip_gemm_bench \
  --config <config> --warmup 0 --timed 1
```

`gemm/profiling/analyze_att.py` was run on each
`ui_output_agent_*_dispatch_2` directory.

| Case | ATT signal |
| --- | --- |
| `ai_high_large_k` | 264.4M total stall, 144.3M idle. Wave duration min/avg/max 42.7M/44.5M/47.5M. Top stalls: `vmcnt(0)` 191.9M, `vmcnt(1)` 34.8M, VALU 16.9M, `lgkmcnt(0)` 11.4M, `lgkmcnt(2)` 4.8M, WMMA 3.3M, LDS 0.8M. |
| `ai_very_high_extreme` | Trace warned that some waves were cut off, but the sampled bottleneck matched large-K: `vmcnt(0)` 900.1M, `vmcnt(1)` 218.9M, VALU 20.0M, `lgkmcnt(0)` 19.9M, `lgkmcnt(2)` 8.8M, WMMA 4.1M. |
| `ai_medium_small` | Only two wave files were sampled. Top stalls were still waitcnt-heavy: `vmcnt(0)` 40.5k, `lgkmcnt(0)` 8.0k, `vmcnt(1)` 6.1k. WMMA did not show meaningful stall. The logical launch is only 5x5 CTAs, so small-shape underutilization dominates more than the pipeline details. |

The waitcnt dependency sources identify both A `global_load_b128` and the B
scalar `global_load_u16` ladder feeding LDS. Direct LDS stall and LDS bank
conflicts were not the main issue: the baseline standard profile for
`ai_high_large_k` reported 144 VGPRs, 20 KB LDS, 2048 SQ waves, 2.99B busy
cycles, 1.46M cycles/wave, and 50 LDS bank conflicts, effectively 0.0 per wave.

Interpretation:

- `vmcnt`: dominant. The boundary path pays global-load wait time even on
  full tiles, with scalar B loads and guarded staging.
- `lgkmcnt` and LDS: secondary. LDS waits are visible, but bank conflicts are
  negligible.
- WMMA: secondary on large cases, not the first limiter.
- VALU/SALU: address and predicate overhead are material in the generic
  boundary path.
- Store path: not a primary ATT bottleneck, but full-tile stores still should
  avoid per-element bounds checks.
- Occupancy/resources: the final fast path increases register pressure, but
  large cases still have enough CTAs. Small 576x576 cases are limited by small
  grid size and boundary tile waste.

## Changes Tried

| Change | Result | Decision |
| --- | --- | --- |
| Boundary full-tile fast paths for A/B loads and C stores, plus K-tail WMMA guard | Kept. This removes per-element bounds checks for the common full-tile path and avoids a zero WMMA slice on partial K tails. |
| M-tail split selector using aligned kernel for full rows and boundary kernel for the final partial row tile | Rejected. Intermediate timing: extreme 456533.0 us / 44.10 TFLOPS, large-K 23663.5 us / 53.17, high-medium 444.1 us / 45.51 PASS, high-small 79.0 us / 10.75 PASS, medium-large 48.6 us / 49.18 PASS, medium-small 15.9 us / 6.90 PASS. The small extreme gain did not justify the high-medium regression. |
| Separate reusable K-boundary kernel | Not added. The current boundary extension recovered most K-tail loss without a new source file or selector branch. A new kernel would need to beat this final result across large-K and medium-large without adding regressions. |

## Final Timings

Final native HIP command:

```bash
.venv/bin/python gemm/kernel_providers/native_hip/run.py \
  --build-dir /home/mahesh/kernelGen/build/bd-11c-att-Release \
  --test gemm/tests/ai_very_high_extreme gemm/tests/ai_high_large_k \
         gemm/tests/ai_high_medium gemm/tests/ai_high_small \
         gemm/tests/ai_medium_large gemm/tests/ai_medium_small \
  --verify -o /tmp/bd-11c-native-final-post-format.json
```

| Test | Native baseline | Native final | Time change | Final vs hipBLASLt |
| --- | ---: | ---: | ---: | ---: |
| `ai_very_high_extreme` | 536179.0 us, 37.55 TFLOPS | 463939.0 us, 43.40 TFLOPS | -13.5% | 66.9% of hipBLASLt TFLOPS |
| `ai_high_large_k` | 34654.5 us, 36.31 TFLOPS | 23730.7 us, 53.02 TFLOPS | -31.5% | 85.9% |
| `ai_high_medium` | 465.0 us, 43.47 TFLOPS | 390.1 us, 51.81 TFLOPS, PASS | -16.1% | 97.5% |
| `ai_high_small` | 80.6 us, 10.53 TFLOPS | 79.2 us, 10.73 TFLOPS, PASS | -1.8% | 33.8% |
| `ai_medium_large` | 116.4 us, 20.54 TFLOPS | 47.8 us, 50.02 TFLOPS, PASS | -58.9% | 106.1% |
| `ai_medium_small` | 19.8 us, 5.54 TFLOPS | 15.8 us, 6.91 TFLOPS, PASS | -19.8% | 61.5% |

## Post-Change ATT Check

For `ai_high_large_k`, post-change ATT under rocprof reported 26359.6 us.
Compared with the baseline ATT analysis:

| Metric | Baseline | Final |
| --- | ---: | ---: |
| Total stall | 264.4M | 126.8M |
| Total idle | 144.3M | 106.1M |
| Wave duration avg | 44.5M | 22.3M |
| `vmcnt(0)` stall | 191.9M | 0.5M |
| `vmcnt(1)` stall | 34.8M | 42.4M |
| `lgkmcnt(0)` stall | 11.4M | 34.5M |
| WMMA stall | 3.3M | 13.1M |
| LDS direct stall | 0.8M | 0.8M |

The change reduced the dominant serialized `vmcnt(0)` wall and exposed more
LDS wait and WMMA issue pressure. The standard final profile for
`ai_high_large_k` reported 168 VGPRs, 20 KB LDS, 2048 SQ waves, 2.22B busy
cycles, 1.09M cycles/wave, and 100 LDS bank conflicts, still only 0.1 per wave.

## Rationale And Remaining Gaps

The final code keeps dispatch unchanged and improves the generic boundary path
used by all irregular tests. This is the best measured balance point from this
iteration: one kernel handles arbitrary M/N/K while full tiles avoid most of the
generic boundary overhead.

Remaining gaps:

- Register pressure increased from 144 to 168 VGPRs in the large-K profile.
  The target timings improved despite that, but future work should avoid
  further VGPR growth.
- `ai_high_small` and `ai_medium_small` remain far behind hipBLASLt. These have
  only 5x5 logical 128x128 CTAs and waste substantial work on boundary tiles;
  a reusable small/medium tile strategy is a better next target than more
  tuning inside this 128x128 boundary kernel.
- The B operand still uses scalar 16-bit global loads. ATT after the change
  shows the original `vmcnt(0)` wall is reduced, but B load efficiency remains
  a likely ceiling for further large-K gains.
