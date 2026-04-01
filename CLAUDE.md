# kernelGen

Framework for benchmarking and comparing AMD GPU kernel providers (GEMM, attention, convolutions).

## Build

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cmake --preset Release
cmake --build ~/kernelGen/build/Release
```

## Project layout

```
kernelGen/
  cmake/                        # Find modules for TheRock/ROCm dependencies
  docs/                         # Design docs and plans
  scripts/                      # One-off helper scripts (test generation, shape selection)
    gemm/                       # GEMM-specific helper scripts
  profiling/                      # Generic rocprofv3 profiling infrastructure
    rocprof.py                  # rocprofv3 wrapper (trace, PMC collection, CSV parsing)
    gpu_specs.py                # GPU hardware specs (peak TFLOPS, bandwidth, CUs)
  gemm/                         # GEMM operation
    utils/                      # Shared C++ utilities (config parsing, verification, HIP helpers)
    kernel_providers/
      hipblaslt/                # hipBLAS-LT provider
        run.py                  # Benchmark runner (common CLI)
        src/bench.cpp           # Benchmark executable
        src/hipblaslt_gemm.cpp  # GEMM implementation
      iree/                     # IREE provider
        run.py                  # Benchmark runner (generates MLIR, compiles, runs)
        src/bench.cpp           # Benchmark executable (IREE runtime + HIP events)
    tests/                      # Test cases (config.json + optional .npy files)
      run_all.py                # Run all tests for a provider
      generate_test.py          # Generate .npy data from config.json
    roofline.py                 # Roofline analysis (--arch or --peak-tflops/--peak-bw)
    profiling/
      profile.py                # GEMM profiling (wraps profiling/rocprof.py)
      analyze.py                # GEMM bottleneck analysis (roofline + counter analysis)
```

## Running tests

```bash
# Run all GEMM tests with a specific provider
python gemm/tests/run_all.py --provider hipblaslt --verify -o results.json
python gemm/tests/run_all.py --provider iree --verify -o results.json

# Run specific test
python gemm/kernel_providers/hipblaslt/run.py --test gemm/tests/ai_high_medium --verify
python gemm/kernel_providers/iree/run.py --test gemm/tests/ai_high_medium --verify

# Roofline analysis
python gemm/roofline.py results.json --arch gfx1100

# Profile a GEMM kernel (rocprofv3 trace + PMC counters)
python gemm/profiling/profile.py --provider hipblaslt --test gemm/tests/ai_high_small -o profile.json

# Analyze bottlenecks from profile
python gemm/profiling/analyze.py profile.json --arch gfx1100
```

## Profiling agent

The `gemm-profiler` agent (`.claude/agents/gemm-profiler.md`) profiles GEMM kernels and analyzes bottlenecks using rocprofv3. It has persistent memory in `.claude/agent-memory/gemm-profiler/` where it stores learned patterns about counter interpretation and tool quirks.

## Test format

Each test is a directory containing:

- `config.json` — M, N, K, dtypes, transpose, alpha/beta
- `input_a.npy`, `input_b.npy`, `output_c.npy` — optional, for correctness verification
- Tests without .npy files use random GPU-initialized data (benchmark-only)

## Key conventions

- All BF16 GEMM tests use `compute_type: f32` for accumulation
- Build presets: Debug, RelWithDebInfo, Release (use Release for benchmarking)
- TheRock/ROCm path defaults to `~/kernelGen/TheRock`, override with `-DTHEROCK_PATH=<path>`
- GPU target defaults to `gfx1100`, override with `-DGPU_TARGETS=<target>`
- Python dependencies: numpy, ml_dtypes (for bfloat16 support), iree-base-compiler (for IREE provider)
- IREE source defaults to `~/kernelGen/iree/iree/`, override with `-DIREE_SOURCE_DIR=<path>`
- IREE .vmfb files cached in `~/.cache/kernelgen/vmfb/<gpu_target>/`
- All providers use HIP events on a HIP stream for timing (apples-to-apples)
