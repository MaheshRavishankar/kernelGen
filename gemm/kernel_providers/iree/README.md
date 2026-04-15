# IREE Kernel Provider

Benchmarks GEMM kernels compiled by [IREE](https://iree.dev) targeting AMD HIP/ROCm.

## How it works

1. `run.py` generates MLIR (`linalg.matmul`) from the test config
1. `iree-compile --iree-opt-level=O3` compiles the MLIR to a `.vmfb` module (cached in `~/.cache/kernelgen/vmfb/iree/<gpu_target>/O3/`)
1. `iree_gemm_bench` loads the `.vmfb` via the IREE runtime C API and dispatches on a custom HIP stream
1. Kernel time is measured with HIP events — identical methodology to hipBLASLt

## Timing methodology

The bench executable creates its own HIP stream and passes it to the IREE HIP driver via `external_stream`. GPU work dispatched by IREE goes to this stream, and HIP events capture the kernel execution time. This ensures an apples-to-apples comparison with hipBLASLt.

## Dependencies

- **IREE source** (runtime only, no compiler): defaults to `~/kernelGen/iree/iree/`, override with `-DIREE_SOURCE_DIR=<path>`
- **iree-compile**: install via `pip install iree-base-compiler` or from IREE release artifacts
- **TheRock/ROCm**: for HIP runtime and headers

Use the repo venv for Python entrypoints, especially IREE and Fusilli tooling.

## Running

```bash
# Single test
.venv/bin/python gemm/kernel_providers/iree/run.py --test gemm/tests/ai_high_medium --verify

# All tests
.venv/bin/python gemm/kernel_providers/iree/run.py --test-dir gemm/tests -o results.json

# Specify GPU target
.venv/bin/python gemm/kernel_providers/iree/run.py --test-dir gemm/tests --gpu-target gfx1100
```
