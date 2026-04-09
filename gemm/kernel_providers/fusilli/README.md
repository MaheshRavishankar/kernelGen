# Fusilli Kernel Provider

Benchmarks GEMM kernels built through the Fusilli C++ graph frontend on AMD
HIP/ROCm.

## Setup

- **Fusilli source**: defaults to `~/iree/fusilli`, override with
  `-DFUSILLI_SOURCE_DIR=<path>`
- **IREE source**: defaults to `~/kernelGen/iree/iree/`, override with
  `-DIREE_SOURCE_DIR=<path>`
- **IREE compiler library**: Fusilli compiles through the IREE compiler C API.
  If `libIREECompiler.so` is not discoverable from Python site-packages or
  `LD_LIBRARY_PATH`, pass `--iree-compiler-lib <path>` to `run.py` or set
  `FUSILLI_EXTERNAL_IREE_COMPILER_LIB=<path>`. A default can also be baked into
  the benchmark with `-DIREE_COMPILER_LIB=<path>`.

The runner sets:

- `FUSILLI_CACHE_DIR` to `${KERNELGEN_CACHE_DIR:-~/.cache/kernelgen}/fusilli`
- `FUSILLI_EXTERNAL_IREE_COMPILER_LIB` when a compiler library path is supplied
  or found in the repo venv

This keeps Fusilli compile artifacts separate from the IREE provider cache
while preserving the shared kernelGen cache-root override.

## Execution Model

`fusilli_gemm_bench`:

1. Builds a `fusilli::Graph` with `Graph::tensor`, `Graph::matmul`,
   `Graph::validate`, and `Graph::compile`
1. Creates a Fusilli AMDGPU handle on an external HIP stream
1. Uses `libIREECompiler.so` through Fusilli's compiler C API during graph
   compilation
1. Allocates Fusilli buffers, dispatches the graph, and measures kernel time
   with HIP events on that stream

## Running

```bash
.venv/bin/python gemm/kernel_providers/fusilli/run.py --test gemm/tests/ai_high_medium --verify
.venv/bin/python gemm/kernel_providers/fusilli/run.py --test-dir gemm/tests -o results.json
.venv/bin/python gemm/tests/run_all.py --provider fusilli --verify -o results.json
```
