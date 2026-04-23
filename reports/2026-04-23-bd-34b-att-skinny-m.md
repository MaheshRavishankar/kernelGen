# bd-34b ATT-Driven Native HIP Skinny-M GEMM Tuning

Date: 2026-04-23
Worktree: `/home/mahesh/kernelGen/kernelGen-bd-34b`
Branch: `users/MaheshRavishankar/bd-34b-attProfileSkinnyM`
Build: `/home/mahesh/kernelGen/build/bd-34b-att-Release`

## Summary

The native HIP skinny-M kernel remains a single strategy selected for `M <= 32`.
No new kernel was added. The change keeps the existing 256-thread, BM=128,
BN=16 staging shape for B-load coverage and N-grid parallelism, but for `K > 32`
only the M warps that can produce output stage A and issue WMMA. For `K <= 32`
the kernel keeps all 8 compute warps because `ai_very_low_tiny` is launch/load
dominated and does not repay the extra branch/imbalance cost.

Primary result with `--warmup 20 --timed 200 --verify`:

| Test | Shape | Native before | Native after | Change |
| --- | ---: | ---: | ---: | ---: |
| `ai_very_low_tiny` | 4x384x5 | 4.765 us | 4.804 us | -0.8% |
| `ai_low_skinny` | 16x512x1024 | 16.445 us | 14.115 us | +14.2% |
| `ai_low_small` | 32x576x2304 | 31.732 us | 28.180 us | +11.2% |

The tiny-shape regression is below 0.04 us and within the observed timing noise,
but it is recorded as a regression. Native HIP is still slower than hipBLASLt on
the two larger skinny targets.

## Baselines

Release build configured with native HIP and hipBLASLt enabled; IREE/Fusilli were
disabled for this bead-specific build because the task only changes native HIP
and compares against hipBLASLt.

Before code changes:

| Test | native_hip | hipBLASLt |
| --- | ---: | ---: |
| `ai_very_low_tiny` | 4.765 us, PASS | 6.190 us, PASS |
| `ai_low_skinny` | 16.445 us, PASS | 9.450 us, PASS |
| `ai_low_small` | 31.732 us, PASS | 13.840 us, PASS |

Final recheck:

| Test | native_hip | hipBLASLt |
| --- | ---: | ---: |
| `ai_very_low_tiny` | 4.804 us, PASS | 6.360 us, PASS |
| `ai_low_skinny` | 14.115 us, PASS | 9.920 us, PASS |
| `ai_low_small` | 28.180 us, PASS | 14.480 us, PASS |

## Baseline ATT Bottlenecks

ATT was collected with `rocprofv3 --att`, `--kernel-include-regex gemm_wmma_skinny`, `--att-gpu-index 0`, `--att-simd-select 0`, and analyzed with
`gemm/profiling/analyze_att.py`.

| Test | Wave avg cycles | Total stall | VMEM wait stall | LGKM wait stall | LDS stall | WMMA stall | VALU stall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ai_low_skinny` | 50,560 | 56,518 | 42,022 | 12,465 | 1,696 | 0 | 47 |
| `ai_low_small` | 101,036 | 112,445 | 89,526 | 19,632 | 2,594 | 0 | 45 |
| `ai_very_low_tiny` | 9,514 | 8,559 | 2,040 | 6,421 | 53 | 0 | 45 |

Findings before code changes:

- `vmcnt` waits dominate `ai_low_skinny` and `ai_low_small`. The top wait
  sources are B `global_load_u16` rows and A `global_load_b128` rows.
- `lgkmcnt` and LDS are secondary. Standard rocprof PMC showed low LDS bank
  conflicts: about 0.2 conflicts/wave for both larger shapes.
- WMMA and VALU stalls are not limiting, but WMMA hit count exposes OOB compute
  waste: 128 hits for M=16 and 288 hits for M=32 even though only 1 and 2 M-warps
  respectively can produce output.
- Store bandwidth is not a visible bottleneck in ATT. Store-side instructions do
  not appear among the top wait sources/stall rows, and output traffic is small
  relative to streamed B traffic.
- Occupancy/grid parallelism is constrained by skinny geometry: 24, 32, and 36
  N-tiles for tiny, skinny, and small respectively. The 256-thread block provides
  8 waves per tile, but block count still underfills 48 CUs. Splitting M would
  increase blocks only by duplicating B traffic, which ATT already identifies as
  the dominant pressure.
- BN=16 remains the best selector-side tradeoff for these shapes: it preserves
  N-grid parallelism. Larger BN would reduce blocks; smaller effective BN would
  waste WMMA columns or duplicate work.

## Tried Changes

All variants used the same single skinny kernel and unchanged host selector.

| Variant | `ai_very_low_tiny` | `ai_low_skinny` | `ai_low_small` | Decision |
| --- | ---: | ---: | ---: | --- |
| Active output M-warps for `K > 32`, 8 warps for `K <= 32` | 4.804 us | 14.115 us | 28.180 us | Kept |
| At least 2 compute warps for `K > 32` | 4.785 us | 14.613 us | 28.225 us | Rejected; slower skinny |
| 4 compute warps for `K > 32` | 4.795 us | 15.001 us | 29.074 us | Rejected; slower larger shapes |

ATT did not justify changing BN/BM tile shape or adding a separate tiny-M kernel.
The tiny shape is launch/underfill dominated; its final timing moved by less than
0.04 us while preserving the original 8-warp tiny-K behavior.

## Final ATT/PMC Check

Final ATT confirms the change reduced OOB work while leaving VMEM pressure as the
remaining bottleneck:

| Test | Wave avg before -> after | WMMA hits | LDS hits | Total stall before -> after | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `ai_low_skinny` | 50,560 -> 46,484 | 128 -> 64 | 704 -> 384 | 56,518 -> 52,745 | VMEM waits still dominate |
| `ai_low_small` | 101,036 -> 86,755 | 288 -> 144 | 1,584 -> 864 | 112,445 -> 94,229 | VMEM waits still dominate |
| `ai_very_low_tiny` | 9,514 -> 8,989 | 4 -> 4 | 38 -> 38 | 8,559 -> 8,226 | Tiny path effectively unchanged |

Standard rocprof PMC/resource checks:

| Test | SQ_BUSY_CYCLES before -> after | Cycles/wave before -> after | Resources |
| --- | ---: | ---: | --- |
| `ai_low_skinny` | 1,586,520 -> 1,368,764 | 6,197 -> 5,347 | 64 VGPR, 128 SGPR, 11,776 B LDS |
| `ai_low_small` | 3,789,547 -> 3,445,192 | 13,158 -> 11,962 | 64 VGPR, 128 SGPR, 11,776 B LDS |
| `ai_very_low_tiny` | 145,699 -> 148,179 | 759 -> 772 | 64 VGPR, 128 SGPR, 11,776 B LDS |

## Validation Commands

```bash
cmake -S . -B /home/mahesh/kernelGen/build/bd-34b-att-Release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=/usr/bin/ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/ccache \
  -DGPU_TARGETS=gfx1100 -DKERNELGEN_ENABLE_GEMM=ON \
  -DKERNELGEN_ENABLE_HIPBLASLT=ON -DKERNELGEN_ENABLE_NATIVE_HIP=ON \
  -DKERNELGEN_ENABLE_FUSILLI=OFF -DKERNELGEN_ENABLE_IREE=OFF

cmake --build /home/mahesh/kernelGen/build/bd-34b-att-Release \
  --target native_hip_gemm_bench hipblaslt_gemm_bench

/home/mahesh/kernelGen/kernelGen/.venv/bin/python \
  gemm/kernel_providers/native_hip/run.py \
  --build-dir /home/mahesh/kernelGen/build/bd-34b-att-Release \
  --test gemm/tests/ai_very_low_tiny gemm/tests/ai_low_skinny \
         gemm/tests/ai_low_small \
  --warmup 20 --timed 200 --verify

/home/mahesh/kernelGen/kernelGen/.venv/bin/python \
  gemm/kernel_providers/hipblaslt/run.py \
  --build-dir /home/mahesh/kernelGen/build/bd-34b-att-Release \
  --test gemm/tests/ai_very_low_tiny gemm/tests/ai_low_skinny \
         gemm/tests/ai_low_small \
  --warmup 20 --timed 200 --verify
```

Additional profiling was run with `gemm/profiling/profile.py`,
`gemm/profiling/analyze.py`, direct `rocprofv3 --att`, and
`gemm/profiling/analyze_att.py` for all three target tests.
