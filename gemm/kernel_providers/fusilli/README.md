# Fusilli Kernel Provider

Benchmarks GEMM kernels through the Fusilli provider entrypoint while using the
same IREE VMFB/runtime execution path as the IREE provider.

## Current setup

Fusilli is modeled as a separate provider so benchmark outputs are labeled and
stored independently from `iree`. In the current implementation, `run.py`
reuses the shared IREE MLIR/VMFB runner and `fusilli_gemm_bench` reuses the
IREE benchmark executable source.

Compiled modules are cached under
`~/.cache/kernelgen/vmfb/fusilli/<gpu_target>/` so Fusilli runs do not reuse the
IREE provider cache.

## Assumptions

- `iree-compile` is the compiler frontend used for this first Fusilli wrapper.
- `IREE_SOURCE_DIR` still points to an IREE source checkout that can build the
  runtime.
- If a dedicated Fusilli compiler/frontend needs to be introduced later, this
  provider can keep its public CLI and swap only the compile path.

## Running

```bash
python gemm/kernel_providers/fusilli/run.py --test gemm/tests/ai_high_medium --verify
python gemm/kernel_providers/fusilli/run.py --test-dir gemm/tests -o results.json
python gemm/tests/run_all.py --provider fusilli --verify -o results.json
```
