# Scripts

One-off helper scripts for generating test data, selecting shapes, and other tasks that are not part of the regular benchmark workflow.

## gemm/

- `select_gemm_shapes.py` — Parses a CSV of GEMM shapes (e.g., from nightly benchmark reports), computes arithmetic intensity, and selects a representative subset across AI bins.
- `create_benchmark_tests.py` — Creates test directories with `config.json` for the selected shapes. Generates `.npy` reference data for small shapes; large shapes get config-only (random init at runtime).
