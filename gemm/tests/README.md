# GEMM Tests

Each test is a directory containing:

- `config.json` — problem description (M, N, K, dtype, transpose, etc.)
- `input_a.npy` — A matrix (numpy format, tracked via git-lfs)
- `input_b.npy` — B matrix
- `output_c.npy` — reference output C matrix

Use `scripts/generate_test.py` to create new test cases.
Use `scripts/run_test.py` to benchmark a provider against a test.
