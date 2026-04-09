# Fusilli Kernel Provider

Benchmarks GEMM kernels built through the Fusilli C++ graph frontend on AMD
HIP/ROCm.

## Setup

- **Fusilli source**: defaults to `~/iree/fusilli`, override with
  `-DFUSILLI_SOURCE_DIR=<path>`
- **IREE source**: defaults to `~/kernelGen/iree/iree/`, override with
  `-DIREE_SOURCE_DIR=<path>`
- **Python tooling**: use the repo venv for `run.py`, especially for
  `iree-compile`

The runner sets:

- `FUSILLI_COMPILE_BACKEND_USE_CLI=1` so Fusilli uses the `iree-compile` CLI
  instead of the compiler C API
- `FUSILLI_EXTERNAL_IREE_COMPILE` to the repo-venv `iree-compile` when present
- `FUSILLI_CACHE_DIR` to `${KERNELGEN_CACHE_DIR:-~/.cache/kernelgen}/fusilli`

This keeps Fusilli compile artifacts separate from the IREE provider cache
while preserving the shared kernelGen cache-root override.

## Execution Model

`fusilli_gemm_bench`:

1. Builds a `fusilli::Graph` with `Graph::tensor`, `Graph::matmul`,
   `Graph::validate`, and `Graph::compile`
1. Creates a Fusilli AMDGPU handle on an external HIP stream
1. Allocates Fusilli buffers, dispatches the graph, and measures kernel time
   with HIP events on that stream

## Running

```bash
.venv/bin/python gemm/kernel_providers/fusilli/run.py --test gemm/tests/ai_high_medium --verify
.venv/bin/python gemm/kernel_providers/fusilli/run.py --test-dir gemm/tests -o results.json
.venv/bin/python gemm/tests/run_all.py --provider fusilli --verify -o results.json
```
