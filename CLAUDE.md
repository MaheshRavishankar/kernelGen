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
  gemm/                         # GEMM operation
    kernel_providers/
      hipblaslt/                # hipBLAS-LT provider
        run.py                  # Benchmark runner (common CLI)
        src/bench.cpp           # Benchmark executable
        src/hipblaslt_gemm.cpp  # GEMM implementation
        include/hipblaslt_gemm.h
    tests/                      # Test cases (config.json + optional .npy files)
      run_all.py                # Run all tests for a provider
      generate_test.py          # Generate .npy data from config.json
    roofline.py                 # Roofline analysis (--arch or --peak-tflops/--peak-bw)
```

## Running tests

```bash
# Run all GEMM tests with verification
python gemm/tests/run_all.py --verify -o results.json

# Run specific test
python gemm/kernel_providers/hipblaslt/run.py --test gemm/tests/ai_high_medium --verify

# Roofline analysis
python gemm/roofline.py results.json --arch gfx1100
```

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
- Python dependencies: numpy, ml_dtypes (for bfloat16 support)
