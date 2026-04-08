#!/usr/bin/env python3
"""
Run IREE GEMM benchmark.

Generates MLIR for each test config, compiles to .vmfb with iree-compile,
then benchmarks with the iree_gemm_bench executable (HIP-event timing).

Usage:
  # Single test:
  python gemm/kernel_providers/iree/run.py --test gemm/tests/small_f16

  # All tests:
  python gemm/kernel_providers/iree/run.py --test-dir gemm/tests

  # With verification:
  python gemm/kernel_providers/iree/run.py --test-dir gemm/tests --verify
"""

from common import run_provider


def main():
    run_provider(
        provider_name="iree",
        benchmark_title="IREE",
        description="Run IREE GEMM benchmark",
    )


if __name__ == "__main__":
    main()
