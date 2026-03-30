#!/usr/bin/env python3
"""Run all GEMM tests in this directory.

Convenience wrapper around the provider-specific run.py scripts.

Usage:
  python gemm/tests/run_all.py
  python gemm/tests/run_all.py --verify
  python gemm/tests/run_all.py --provider hipblaslt --verify -o results.json
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROVIDERS = {
    "hipblaslt": "gemm/kernel_providers/hipblaslt/run.py",
}


def main():
    parser = argparse.ArgumentParser(description="Run all GEMM tests")
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS.keys()),
        default="hipblaslt",
        help="Kernel provider (default: hipblaslt)",
    )
    parser.add_argument("--build-dir", default=None, help="Build directory")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timed", type=int, default=20)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--output", "-o", default=None, help="Write JSON results to file"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    test_dir = Path(__file__).resolve().parent
    run_script = repo_root / PROVIDERS[args.provider]

    cmd = [
        sys.executable,
        str(run_script),
        "--test-dir",
        str(test_dir),
        "--warmup",
        str(args.warmup),
        "--timed",
        str(args.timed),
    ]
    if args.build_dir:
        cmd += ["--build-dir", args.build_dir]
    if args.verify:
        cmd += ["--verify"]
    if args.output:
        cmd += ["--output", args.output]

    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
