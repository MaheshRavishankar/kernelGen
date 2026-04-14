# bd-1ke GEMM benchmark summary

Worktree: `/home/mahesh/kernelGen/kernelGen-bd-1ke`
Branch: `users/MaheshRavishankar/bd-1ke-benchmarkExistingPerf`

Commands run from the worktree against the existing Release build:

```bash
python gemm/tests/run_all.py --provider hipblaslt --verify -o reports/bd-1ke-hipblaslt.json
python gemm/tests/run_all.py --provider iree --verify -o reports/bd-1ke-iree.json
python gemm/tests/run_all.py --provider native_hip --verify -o reports/bd-1ke-native_hip.json
```

Logs were captured in the matching `reports/bd-1ke-*.log` files.

## Overall outcome

- `hipblaslt`: 17 passed, 0 failed, 0 skipped.
- `iree`: 17 passed, 0 failed, 0 skipped.
- `native_hip`: 3 passed, 14 failed, 0 skipped.
- `hipblaslt` and `iree` verified all 10 tests that have checked-in reference tensors. The remaining 7 tests do not include `output_c.npy`, so the benchmark runner executes them without reference validation.
- `native_hip` did not verify any tests in this run. Its 3 successful tests do not have checked-in reference tensors, and the 14 failures are unsupported shapes or types in the current native provider. Direct spot checks returned `native_hip: M must be multiple of 128, N must be multiple of 128` for `ai_high_medium` and `native_hip: only BF16 data types supported` for `small_f16`.

## Measured highlights

### hipblaslt

- Average across the 17 reported tests: `35.86 TFLOPS`.
- Peak throughput: `ai_very_high_large` at `79.25 TFLOPS` (`4286.69 us`).
- Strongest small/low-arithmetic-intensity cases:
  - `small_f16`: `57.68 TFLOPS` vs IREE `20.31 TFLOPS`.
  - `ai_high_small`: `31.79 TFLOPS` vs IREE `10.14 TFLOPS`.
  - `ai_medium_small`: `11.23 TFLOPS` vs IREE `1.59 TFLOPS`.
  - `ai_low_skinny`: `1.83 TFLOPS` vs IREE `0.22 TFLOPS`.

### iree

- Average across the 17 reported tests: `30.96 TFLOPS`.
- Peak throughput: `ai_very_high_large` at `81.77 TFLOPS` (`4154.58 us`).
- Best measured wins over hipBLASLt:
  - `ai_low_large_flat`: `9.51 TFLOPS` vs `6.06 TFLOPS` (`1.57x`).
  - `ai_medium_extreme`: `35.71 TFLOPS` vs `23.12 TFLOPS` (`1.54x`).
  - `ai_very_high_large`: `81.77 TFLOPS` vs `79.25 TFLOPS` (`1.03x`).
  - `ai_very_high_square`: `80.49 TFLOPS` vs `78.25 TFLOPS` (`1.03x`).
  - `ai_very_high_extreme`: `66.72 TFLOPS` vs `65.36 TFLOPS` (`1.02x`).

### native_hip

- Average across the 3 reported tests: `78.49 TFLOPS`.
- Peak throughput: `ai_very_high_large` at `79.68 TFLOPS` (`4263.90 us`).
- Successful cases:
  - `ai_very_high_large`: `79.68 TFLOPS` vs hipBLASLt `79.25 TFLOPS` and IREE `81.77 TFLOPS`.
  - `ai_very_high_medium`: `79.12 TFLOPS` vs hipBLASLt `78.21 TFLOPS` and IREE `76.23 TFLOPS`.
  - `ai_very_high_square`: `76.68 TFLOPS` vs hipBLASLt `78.25 TFLOPS` and IREE `80.49 TFLOPS`.

## Provider comparison

- `hipblaslt` was faster than `iree` on 10 tests.
- `iree` was faster than `hipblaslt` on 6 tests.
- `ai_very_low_tiny` rounded to `0.00 TFLOPS` for both providers.

Largest observed `hipblaslt` wins:

- `ai_very_low_small_square`: `0.97` vs `0.09 TFLOPS` (`10.8x`).
- `ai_low_skinny`: `1.83` vs `0.22 TFLOPS` (`8.3x`).
- `ai_medium_small`: `11.23` vs `1.59 TFLOPS` (`7.1x`).
- `ai_low_small`: `6.31` vs `0.96 TFLOPS` (`6.6x`).
- `ai_very_low_small_wide`: `2.21` vs `0.35 TFLOPS` (`6.3x`).

Largest observed `iree` wins:

- `ai_low_large_flat`: `1.57x`.
- `ai_medium_extreme`: `1.54x`.
- `ai_very_high_large`: `1.03x`.
- `ai_very_high_square`: `1.03x`.
- `ai_very_high_extreme`: `1.02x`.

## Files produced

- `reports/bd-1ke-hipblaslt.json`
- `reports/bd-1ke-hipblaslt.log`
- `reports/bd-1ke-iree.json`
- `reports/bd-1ke-iree.log`
- `reports/bd-1ke-native_hip.json`
- `reports/bd-1ke-native_hip.log`
- `reports/bd-1ke-summary.md`
