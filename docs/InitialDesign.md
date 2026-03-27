# AMD GPU Kernel exploration.

## Motivation/Purpose

This project is meant to explore different kernel generation
technologies at AMD to understand space of optimal implementations for
different computations like gemms, attention (and various flavors) and
convolutions. The expectation for any of these operations is that no
single kernel will cover the entire space of shapes/configurations
for these operations, but a handful of kernels (order of ~20) would
cover the space reasonably well. The goal is two folds

1. Explore hip/ASM based implementations from CK/HipBLAS-LT/MIOpen,
   etc. and other kernel providers, and learn from them to synthesize
   efficient implementations.
1. Based on these implementations try to replicate the final code that
   they generate using MLIR/IREE.

The "outcomes" from this project are expected to be:

1. Having a general setup that can be used to measure the performance
   of these different kernel providers to be able to make an
   apples-to-apples comparison. The aim is to be able to replicate
   this setup "easily". The benchmarking setup is one of the things
   this project will focus on.

1. The HIP (and ASM) kernel implementations that will be derived from
   the exploration will be wrapped within an API similar to
   hipDNN/cuDNN so they could be used for benchmarking.

1. Based on these HIP/ASM kernels we will try to derive how to
   represent this in MLIR/IREE from the "bottom up" to light the way
   for what a codegeneration needs to do. Best case these lead to
   realization of how to use existing constructs in IREE/MLIR to
   generate efficient code. Worst case they just become MLIR snippets
   that can be dropped into certain use cases.

## Project layout

```
~/kernelGen/
  kernelGen/                    # Source tree (this repo)
  TheRock/                      # TheRock/ROCm installation (nightly or local build)
  build/<preset>/               # Build artifacts (out of source)
  install/<preset>/             # Install artifacts (out of source)
```

Source tree:

```
kernelGen/
  CMakeLists.txt                # Top-level. Guards operations/providers with options.
  CMakePresets.json             # Debug, RelWithDebInfo, Release presets.
  requirements.txt              # Python dependencies.
  cmake/                        # Find modules for TheRock dependencies.
  <operation>/
    CMakeLists.txt              # Guards providers with KERNELGEN_ENABLE_* options.
    kernel_providers/
      <provider>/
        run.py                  # Provider-specific benchmark runner (common CLI).
        CMakeLists.txt          # Builds a static library + bench executable.
        include/                # Provider-specific headers.
        src/                    # Provider-specific sources.
    tests/
      generate_test.py          # Generates npy input/output files from config.
      <test_name>/
        config.json             # Problem description (shapes, types, etc.).
        *.npy                   # Inputs/outputs (tracked via git-lfs).
```

### Design decisions

- **Modular CMake**: Every operation and provider is guarded by a
  `KERNELGEN_ENABLE_*` option. Each provider builds a self-contained
  static library. No directory pulls source files from another.

- **Per-provider runner scripts**: Each provider has its own `run.py`
  with a common CLI (`--test`, `--test-dir`, `--warmup`, `--timed`,
  `--verify`, `--build-dir`). Providers are fully independent with
  no shared state.

- **Test format**: Each test is a directory with a `config.json`
  and `.npy` files. Config supports per-operand types and a
  `compute_type` for accumulation.

- **TheRock path**: Defaults to `~/kernelGen/TheRock`, overridable
  at configure time with `-DTHEROCK_PATH=<path>`.

- **Out-of-source builds**: Presets place build/install directories
  at `~/kernelGen/{build,install}/<preset>`.

## Build and run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cmake --preset Release
cmake --build ~/kernelGen/build/Release
.venv/bin/python <operation>/tests/generate_test.py <operation>/tests/<test>/config.json
.venv/bin/python <operation>/kernel_providers/<provider>/run.py --test-dir <operation>/tests
```

## Adding a new provider

1. Create `<operation>/kernel_providers/<name>/` with `CMakeLists.txt`,
   sources, and a `run.py` following the common CLI.
1. Add a `KERNELGEN_ENABLE_<NAME>` option in the top-level CMakeLists.txt.
1. Guard `add_subdirectory(<name>)` in the operation's CMakeLists.txt.

## Adding a new operation

1. Create `<operation>/` with `CMakeLists.txt`, `kernel_providers/`,
   and `tests/`.
1. Add a `KERNELGEN_ENABLE_<OPERATION>` option and guarded
   `add_subdirectory(<operation>)` in the top-level CMakeLists.txt.
