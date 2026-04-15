______________________________________________________________________

## name: gemm-kernel-writer description: Write and optimize native HIP GEMM kernels for AMD GPUs, including iterative profile-analyze-improve loops.

# GEMM Kernel Writer

Use this skill when the task is to write, modify, or optimize the native HIP
GEMM provider, especially when WMMA tile choices and measured TFLOPS matter.

Always read [AGENTS.md](../../../../AGENTS.md) and the relevant test
configuration before editing code.

Use the repo venv for Python entrypoints, especially IREE and Fusilli tooling.
For sandboxed implementation runs, use the configured Codex default model unless
the user explicitly requests a model override, and instruct the agent to send a
short final completion message before exit.

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
1. Modify the kernel
   - Document the tiling strategy near the kernel
   - Keep tile sizes as `constexpr`
   - Keep WMMA register layout comments accurate
1. Build and smoke test
   - `cmake --build ~/kernelGen/build/Release --target native_hip_gemm_bench`
   - `.venv/bin/python gemm/kernel_providers/native_hip/run.py --test gemm/tests/ai_very_high_square`
   - Add `--verify` when reference data exists
1. Compare against baseline when relevant
   - `.venv/bin/python gemm/kernel_providers/hipblaslt/run.py --test gemm/tests/ai_very_high_square`
1. Profile and iterate
   - Prefer the `gemm-profiler` skill for deeper bottleneck analysis
   - Otherwise run `.venv/bin/python gemm/profiling/profile.py` and `.venv/bin/python gemm/profiling/analyze.py`

## Optimization Priorities

1. Tile sizes: `BM`, `BN`, `BK`, `WM`, `WN`
1. Shared memory bank conflicts
1. Double buffering and software pipelining
1. Vectorized global loads
1. Coalesced accumulator stores
1. Occupancy vs VGPR pressure tradeoffs

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
1. Measured impact
1. Remaining bottleneck or next optimization target
