#!/usr/bin/env python3
"""
Run Fusilli GEMM benchmark.

Fusilli is exposed as its own provider, but this runner currently reuses the
shared IREE MLIR->VMFB path so Fusilli and IREE can be compared side by side on
identical tests while preserving separate benchmark binaries and VMFB caches.

Usage:
  python gemm/kernel_providers/fusilli/run.py --test gemm/tests/small_f16
  python gemm/kernel_providers/fusilli/run.py --test-dir gemm/tests --verify
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "iree"))

from common import run_provider


def main():
    run_provider(
        provider_name="fusilli",
        benchmark_title="Fusilli",
        description="Run Fusilli GEMM benchmark",
    )


if __name__ == "__main__":
    main()
