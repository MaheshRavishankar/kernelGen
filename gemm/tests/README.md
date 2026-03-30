# GEMM Tests

Test cases for benchmarking GEMM kernel providers. Each test is a directory containing:

- `config.json` — problem description (M, N, K, dtypes, transpose, alpha/beta)
- `input_a.npy` — A matrix (optional, tracked via git-lfs)
- `input_b.npy` — B matrix (optional)
- `output_c.npy` — reference output C matrix (optional, for verification)

Tests without `.npy` files are benchmark-only — the bench binary initializes random data on the GPU.

## Running

```bash
# Run all tests
python gemm/tests/run_all.py --verify -o results.json

# Or directly via a provider's runner
python gemm/kernel_providers/hipblaslt/run.py --test-dir gemm/tests --verify
```

## Test organization

Tests are named by arithmetic intensity (AI) bin to cover the full roofline spectrum:

| Prefix | AI range | Bound |
|---|---|---|
| `ai_very_low_*` | < 10 | Memory |
| `ai_low_*` | 10–50 | Memory |
| `ai_medium_*` | 50–200 | Transitional |
| `ai_high_*` | 200–1000 | Compute |
| `ai_very_high_*` | > 1000 | Compute |

## Generating test data

```bash
# Generate .npy files for a single test
python gemm/tests/generate_test.py gemm/tests/<test_name>/config.json

# Regenerate all benchmark tests (configs + .npy for small shapes)
python scripts/gemm/create_benchmark_tests.py
```
