#!/usr/bin/env python3
"""
Run hipBLAS-LT GEMM benchmark.

Usage:
  # Single test:
  python gemm/kernel_providers/hipblaslt/run.py --test gemm/tests/small_f16

  # Multiple tests:
  python gemm/kernel_providers/hipblaslt/run.py --test gemm/tests/small_f16 gemm/tests/large_f16

  # All tests in a directory:
  python gemm/kernel_providers/hipblaslt/run.py --test-dir gemm/tests
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def find_bench_binary(build_dir: Path) -> Path:
    binary = (
        build_dir / "gemm" / "kernel_providers" / "hipblaslt" / "hipblaslt_gemm_bench"
    )
    if not binary.exists():
        print(f"ERROR: Benchmark binary not found: {binary}", file=sys.stderr)
        print("  Build the project first with the appropriate preset.", file=sys.stderr)
        sys.exit(1)
    return binary


def run_one_test(bench: Path, test_dir: Path, warmup: int, timed: int, verify: bool):
    """Run a single test. Returns (config, output) or (config, None) on failure."""
    config_path = test_dir / "config.json"
    input_a = test_dir / "input_a.npy"
    input_b = test_dir / "input_b.npy"
    output_c = test_dir / "output_c.npy"

    for p in [config_path, input_a, input_b]:
        if not p.exists():
            print(f"SKIP: {test_dir.name} — {p.name} not found", file=sys.stderr)
            return None, None

    with open(config_path) as f:
        config = json.load(f)

    cmd = [
        str(bench),
        "--config",
        str(config_path),
        "--input-a",
        str(input_a),
        "--input-b",
        str(input_b),
        "--warmup",
        str(warmup),
        "--timed",
        str(timed),
    ]
    if verify and output_c.exists():
        cmd += ["--reference", str(output_c)]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FAIL: {test_dir.name} (exit code {result.returncode})", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return config, None

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"FAIL: {test_dir.name} — could not parse output", file=sys.stderr)
        return config, None

    return config, output


def print_result(test_name: str, config: dict, output: dict):
    M, N, K = config["M"], config["N"], config["K"]
    time_us = output.get("kernel_time_us", 0)
    flops = 2.0 * M * N * K
    tflops = (flops / (time_us * 1e-6)) / 1e12 if time_us > 0 else 0

    dtype_str = f"A={config['dtype_A']} B={config['dtype_B']} C={config['dtype_C']} compute={config['compute_type']}"
    verify_str = ""
    if "verified" in output:
        verify_str = " PASS" if output["verified"] else " FAIL"

    print(
        f"  {test_name:<20s}  {M:>6d}x{N:>6d}x{K:>6d}  {dtype_str:<40s}  {time_us:>10.1f} us  {tflops:>7.2f} TFLOPS{verify_str}"
    )

    return {
        "test": test_name,
        "M": M,
        "N": N,
        "K": K,
        "dtype_A": config["dtype_A"],
        "dtype_B": config["dtype_B"],
        "dtype_C": config["dtype_C"],
        "compute_type": config["compute_type"],
        "kernel_time_us": time_us,
        "tflops": round(tflops, 2),
        **({k: output[k] for k in ["verified"] if k in output}),
    }


def collect_test_dirs(test_args, test_dir_arg):
    """Gather test directories from --test and --test-dir arguments."""
    dirs = []
    if test_args:
        for t in test_args:
            p = Path(t).resolve()
            if (p / "config.json").exists():
                dirs.append(p)
            else:
                print(f"SKIP: {t} — no config.json found", file=sys.stderr)
    if test_dir_arg:
        parent = Path(test_dir_arg).resolve()
        for child in sorted(parent.iterdir()):
            if child.is_dir() and (child / "config.json").exists():
                dirs.append(child)
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Run hipBLAS-LT GEMM benchmark")
    parser.add_argument(
        "--test", nargs="+", default=None, help="Path(s) to test directories"
    )
    parser.add_argument(
        "--test-dir",
        default=None,
        help="Directory containing test subdirectories (runs all)",
    )
    parser.add_argument(
        "--build-dir",
        default=None,
        help="Build directory (default: ~/kernelGen/build/Release)",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--timed", type=int, default=20, help="Timed iterations")
    parser.add_argument(
        "--verify", action="store_true", help="Verify output against reference"
    )
    args = parser.parse_args()

    if not args.test and not args.test_dir:
        parser.error("Provide --test or --test-dir")

    if args.build_dir is None:
        args.build_dir = os.path.expanduser("~/kernelGen/build/Release")

    build_dir = Path(args.build_dir).resolve()
    bench = find_bench_binary(build_dir)
    test_dirs = collect_test_dirs(args.test, args.test_dir)

    if not test_dirs:
        print("No tests found.", file=sys.stderr)
        sys.exit(1)

    print(f"{'=' * 60}")
    print(f"hipblaslt GEMM benchmark — {len(test_dirs)} test(s)")
    print(f"{'=' * 60}")
    print(
        f"  {'Test':<20s}  {'Problem':>20s}  {'Types':<40s}  {'Time':>13s}  {'Perf':>13s}"
    )
    print(f"  {'-' * 20}  {'-' * 20}  {'-' * 40}  {'-' * 13}  {'-' * 13}")

    results = []
    failures = 0
    for test_dir in test_dirs:
        config, output = run_one_test(
            bench, test_dir, args.warmup, args.timed, args.verify
        )
        if config is None:
            continue
        if output is None:
            failures += 1
            continue
        r = print_result(test_dir.name, config, output)
        results.append(r)

    print(f"{'=' * 60}")
    print(
        f"{len(results)} passed, {failures} failed, {len(test_dirs) - len(results) - failures} skipped"
    )

    # Emit JSON array for scripting.
    if results:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
