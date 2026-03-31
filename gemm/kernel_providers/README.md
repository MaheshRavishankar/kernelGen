# Kernel Providers

Each subdirectory is a self-contained kernel provider for the GEMM operation. Providers share common utilities from `gemm/utils/` but are otherwise independent.

## Structure

```
<provider>/
  CMakeLists.txt          # Builds bench executable (+ optional static library)
  run.py                  # Python benchmark runner (common CLI)
  src/
    bench.cpp             # Benchmark executable (loads config, runs GEMM, outputs JSON)
```

## Common CLI

All provider `run.py` scripts share the same interface:

```bash
python gemm/kernel_providers/<provider>/run.py \
  --test-dir gemm/tests \
  --warmup 5 --timed 20 \
  --verify \
  -o results.json
```

## Current providers

- **hipblaslt** — AMD hipBLAS-LT library. Uses `hipblasLtMatmul` with algorithm heuristics.
- **iree** — IREE compiler. Compiles `linalg.matmul` MLIR to GPU kernels, dispatches via IREE runtime with external HIP stream.

## Adding a new provider

1. Create `<name>/` with `CMakeLists.txt`, sources, and `run.py` following the common CLI.
1. Add `KERNELGEN_ENABLE_<NAME>` option in the top-level `CMakeLists.txt`.
1. Guard `add_subdirectory(<name>)` in `gemm/CMakeLists.txt`.
