# Replace HIP Event Timing with rocprof

## Context

Currently, kernel timing uses HIP events (`hipEventRecord`/`hipEventElapsedTime`) inside the C++ code. We want to use `rocprof --stats` instead, which gives per-kernel dispatch timing from the ROCm profiler. This provides more accurate GPU kernel timing and aligns with standard AMD profiling workflows.

## Approach

- **C++ code**: Remove HIP event timing. The bench executable just runs warmup + timed GEMM dispatches and handles verification. It no longer reports `kernel_time_us`.
- **run.py**: Wraps the bench executable with `rocprof --stats -- ./bench ...`, parses the generated stats CSV to extract kernel timing, and reports it in the output.

## Files to Modify

### 1. `gemm/kernel_providers/hipblaslt/src/hipblaslt_gemm.cpp`

- Remove `hipEvent_t` creation, `hipEventRecord`, `hipEventElapsedTime`
- Keep warmup loop (with stream sync after) and timed loop (with stream sync after)
- `GemmResult::kernel_time_us` no longer set by C++ (set to 0 or remove)

### 2. `gemm/kernel_providers/hipblaslt/src/bench.cpp`

- Remove `kernel_time_us` from JSON output (or output 0)
- Keep everything else (NPY loading, config parsing, verification, GPU memory management)

### 3. `gemm/kernel_providers/hipblaslt/include/hipblaslt_gemm.h`

- Remove `kernel_time_us` from `GemmResult` (or keep as unused field)

### 4. `gemm/kernel_providers/hipblaslt/run.py` (main changes)

- Wrap bench invocation with `rocprof --stats -o <tmpdir>/results.csv --`
- Parse `<tmpdir>/results.stats.csv` after execution
- Identify the GEMM kernel (highest total duration kernel, filtering out memcpy/memset)
- Extract average kernel time from the stats CSV
- Report timing in the same output format (kernel_time_us, TFLOPS)

## rocprof Stats CSV Format

`rocprof --stats` generates a `results.stats.csv` with columns:

```
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"some_gemm_kernel",25,12500000,500000,98.5
```

We parse this, find the kernel with the highest `Percentage` (the GEMM kernel), and use `AverageNs` converted to microseconds as the kernel time.

## Verification

1. Build: `cmake --preset Release && cmake --build --preset Release`
1. Run: `python gemm/kernel_providers/hipblaslt/run.py --test-dir gemm/tests --verify`
1. Confirm rocprof stats CSV is generated and parsed
1. Confirm TFLOPS output matches expected range for the test size
