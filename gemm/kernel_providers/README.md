# Kernel Providers

Each subdirectory is a self-contained kernel provider for the GEMM operation. Providers are independent — they share no code or state with each other.

## Structure

```
<provider>/
  CMakeLists.txt          # Builds a static library + bench executable
  run.py                  # Python benchmark runner (common CLI)
  include/                # Public headers
  src/
    <provider>_gemm.cpp   # GEMM implementation (library)
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

## Adding a new provider

1. Create `<name>/` with `CMakeLists.txt`, sources, and `run.py` following the common CLI.
1. Add `KERNELGEN_ENABLE_<NAME>` option in the top-level `CMakeLists.txt`.
1. Guard `add_subdirectory(<name>)` in `gemm/CMakeLists.txt`.
