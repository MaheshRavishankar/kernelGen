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
  .beads/                       # Beads issue tracker (managed by `br` CLI)
  scripts/                      # One-off helper scripts (test generation, shape selection)
    kernelgen-sandbox.sh        # bwrap sandbox for running agents on beads
    setup-bwrap-apparmor.sh     # One-time AppArmor setup for bwrap (Ubuntu 24.04+)
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
      native_hip/               # Native HIP provider (hand-written WMMA kernels)
        run.py                  # Benchmark runner
        src/bench.cpp           # Benchmark executable
        src/native_hip_gemm.hip # WMMA GEMM kernel + host launch
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
python gemm/kernel_providers/native_hip/run.py --test gemm/tests/ai_very_high_square

# Roofline analysis
python gemm/roofline.py results.json --arch gfx1100

# Profile a GEMM kernel (rocprofv3 trace + PMC counters)
python gemm/profiling/profile.py --provider hipblaslt --test gemm/tests/ai_high_small -o profile.json

# Analyze bottlenecks from profile
python gemm/profiling/analyze.py profile.json --arch gfx1100
```

## Agents

Agents execute inside a **bubblewrap (bwrap) sandbox** via `scripts/kernelgen-sandbox.sh`. The sandbox enforces filesystem isolation: the main checkout is read-only (except `.beads/` and `.git/worktrees/`), the bead worktree and build directories are read-write, and SDKs are read-only. Claude runs with `--dangerously-skip-permissions` inside the sandbox, so the filesystem restrictions ARE the permission model.

```bash
# Launch a sandboxed agent for a bead
scripts/kernelgen-sandbox.sh <bead-id> -- -p "Your task prompt here"

# One-time AppArmor setup (Ubuntu 24.04+)
sudo scripts/setup-bwrap-apparmor.sh
```

The `gemm-profiler` agent (`.claude/agents/gemm-profiler.md`) profiles GEMM kernels and analyzes bottlenecks using rocprofv3. It has persistent memory in `.claude/agent-memory/gemm-profiler/` where it stores learned patterns about counter interpretation and tool quirks.

The `gemm-kernel-writer` agent (`.claude/agents/gemm-kernel-writer.md`) writes and optimizes native HIP GEMM kernels using WMMA intrinsics. It works in a profile-analyze-improve loop with the profiler agent and stores optimization findings in `.claude/agent-memory/gemm-kernel-writer/`.

## Test format

Each test is a directory containing:

- `config.json` — M, N, K, dtypes, transpose, alpha/beta
- `input_a.npy`, `input_b.npy`, `output_c.npy` — optional, for correctness verification
- Tests without .npy files use random GPU-initialized data (benchmark-only)
- `.npy` files must only be generated when total size of A+B+C is ≤ 50 MB (see `MAX_NPY_BYTES` in `scripts/gemm/create_benchmark_tests.py`). Larger tests are config-only (benchmark without committed reference data).

## Beads issue tracker

Issues are managed with the `br` CLI (a Rust binary). **Never edit `.beads/issues.jsonl` by hand** — always use `br` commands. The JSONL file is auto-exported from the SQLite DB after each mutation.

```bash
# Common commands
br list                              # List open issues
br show <id>                         # Show issue details
br ready                             # List unblocked issues
br blocked                           # List blocked issues
br create "Title" -d "Description" -t task -p 1  # Create issue
br close <id> -r "Reason"            # Close issue
br dep add <issue> <depends-on>      # Add dependency
br dep tree <id>                     # Show dependency tree
br update <id> -s in_progress        # Update status
br search "keyword"                  # Search issues
```

Add `--json` to any command for machine-readable output.

## Key conventions

- All BF16 GEMM tests use `compute_type: f32` for accumulation
- Build presets: Debug, RelWithDebInfo, Release (use Release for benchmarking)
- TheRock/ROCm path defaults to `~/kernelGen/TheRock`, override with `-DTHEROCK_PATH=<path>`
- GPU target defaults to `gfx1100`, override with `-DGPU_TARGETS=<target>`
- Python dependencies: numpy, ml_dtypes (for bfloat16 support), iree-base-compiler (for IREE provider)
- IREE source defaults to `~/kernelGen/iree/iree/`, override with `-DIREE_SOURCE_DIR=<path>`
- IREE .vmfb files cached in `~/.cache/kernelgen/vmfb/<gpu_target>/`
- All providers use HIP events on a HIP stream for timing (apples-to-apples)
