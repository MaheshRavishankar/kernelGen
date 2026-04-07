---
name: gemm-kernel-writer
description: Write and optimize native HIP GEMM kernels for AMD GPUs, including iterative profile-analyze-improve loops.
---

# GEMM Kernel Writer

Use this skill when the task is to write, modify, or optimize the native HIP
GEMM provider, especially when WMMA tile choices and measured TFLOPS matter.

Always read [AGENTS.md](../../../../AGENTS.md) and the relevant test
configuration before editing code.

## Scope

- Primary source: `gemm/kernel_providers/native_hip/src/native_hip_gemm.hip`
- Related runners and utilities under `gemm/kernel_providers/native_hip/`
- Comparison baseline: `hipblaslt`

## Workflow

1. Understand the target shape
   - Read `config.json`
   - Compute arithmetic intensity
   - Classify compute-bound vs memory-bound
   - Check `profiling/gpu_specs.py` when peak numbers matter
2. Modify the kernel
   - Document the tiling strategy near the kernel
   - Keep tile sizes as `constexpr`
   - Keep WMMA register layout comments accurate
3. Build and smoke test
   - `cmake --build ~/kernelGen/build/Release --target native_hip_gemm_bench`
   - `python gemm/kernel_providers/native_hip/run.py --test gemm/tests/ai_very_high_square`
   - Add `--verify` when reference data exists
4. Compare against baseline when relevant
   - `python gemm/kernel_providers/hipblaslt/run.py --test gemm/tests/ai_very_high_square`
5. Profile and iterate
   - Prefer the `gemm-profiler` skill for deeper bottleneck analysis
   - Otherwise run `gemm/profiling/profile.py` and `gemm/profiling/analyze.py`

## Optimization Priorities

1. Tile sizes: `BM`, `BN`, `BK`, `WM`, `WN`
2. Shared memory bank conflicts
3. Double buffering and software pipelining
4. Vectorized global loads
5. Coalesced accumulator stores
6. Occupancy vs VGPR pressure tradeoffs

## gfx1100 Notes

- Peak BF16 TFLOPS: about `123`
- Peak memory bandwidth: about `864 GB/s`
- Wave size: `32`
- LDS per CU: `64 KB`
- VGPRs per SIMD: `1536`

WMMA intrinsic:

```cpp
f32x8 __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(bf16x16 a, bf16x16 b, f32x8 c);
```

## Hard Rules

- Do not edit the main checkout when operating from a bead worktree flow.
- Keep kernel code readable enough to tune again later.
- Do not claim a performance improvement without measurement.
- Run at least a smoke test after every meaningful kernel change.

## Output

Report:

1. What changed
2. Measured impact
3. Remaining bottleneck or next optimization target
