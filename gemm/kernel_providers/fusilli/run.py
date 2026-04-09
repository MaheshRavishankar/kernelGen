#!/usr/bin/env python3
"""Run the Fusilli GEMM benchmark."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

GPU_TARGET = os.environ.get("KERNELGEN_GPU_TARGET", "gfx1100")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def repo_venv_candidates() -> list[Path]:
    root = repo_root()
    return [
        root / ".venv",
        root.parent / "kernelGen" / ".venv",
    ]


def find_bench_binary(build_dir: Path) -> Path:
    binary = build_dir / "gemm" / "kernel_providers" / "fusilli" / "fusilli_gemm_bench"
    if not binary.exists():
        print(f"ERROR: Benchmark binary not found: {binary}", file=sys.stderr)
        print("  Build the project first with the appropriate preset.", file=sys.stderr)
        sys.exit(1)
    return binary


def find_iree_compile(iree_compile_arg: str | None) -> Path:
    if iree_compile_arg:
        path = Path(iree_compile_arg)
        if path.exists():
            return path

    which = shutil.which("iree-compile")
    candidates = [
        *(venv / "bin" / "iree-compile" for venv in repo_venv_candidates()),
        Path(which) if which else None,
        Path.home() / "kernelGen" / "iree" / "build" / "tools" / "iree-compile",
        Path.home() / ".local" / "bin" / "iree-compile",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate

    print(
        "ERROR: iree-compile not found. Install it in the repo venv with "
        "'.venv/bin/pip install iree-base-compiler' or pass --iree-compile.",
        file=sys.stderr,
    )
    sys.exit(1)


def collect_test_dirs(test_args, test_dir_arg):
    dirs = []
    if test_args:
        for test in test_args:
            path = Path(test).resolve()
            if (path / "config.json").exists():
                dirs.append(path)
            else:
                print(f"SKIP: {test} — no config.json found", file=sys.stderr)
    if test_dir_arg:
        parent = Path(test_dir_arg).resolve()
        for child in sorted(parent.iterdir()):
            if child.is_dir() and (child / "config.json").exists():
                dirs.append(child)
    return dirs


def print_result(test_name: str, config: dict, output: dict):
    m, n, k = config["M"], config["N"], config["K"]
    time_us = output.get("kernel_time_us", 0)
    flops = 2.0 * m * n * k
    tflops = (flops / (time_us * 1e-6)) / 1e12 if time_us > 0 else 0
    dtype_str = (
        f"A={config['dtype_A']} B={config['dtype_B']} "
        f"C={config['dtype_C']} compute={config['compute_type']}"
    )
    verify_str = ""
    if "verified" in output:
        verify_str = " PASS" if output["verified"] else " FAIL"

    print(
        f"  {test_name:<20s}  {m:>6d}x{n:>6d}x{k:>6d}  "
        f"{dtype_str:<40s}  {time_us:>10.1f} us  {tflops:>7.2f} TFLOPS{verify_str}"
    )

    return {
        "test": test_name,
        "M": m,
        "N": n,
        "K": k,
        "dtype_A": config["dtype_A"],
        "dtype_B": config["dtype_B"],
        "dtype_C": config["dtype_C"],
        "compute_type": config["compute_type"],
        "kernel_time_us": time_us,
        "tflops": round(tflops, 2),
        **({key: output[key] for key in ["verified"] if key in output}),
    }


def fusilli_env(iree_compile: Path) -> dict[str, str]:
    env = os.environ.copy()
    cache_root = Path(
        env.get("KERNELGEN_CACHE_DIR", Path.home() / ".cache" / "kernelgen")
    )
    env["FUSILLI_COMPILE_BACKEND_USE_CLI"] = "1"
    env["FUSILLI_EXTERNAL_IREE_COMPILE"] = str(iree_compile)
    env["FUSILLI_CACHE_DIR"] = str(cache_root / "fusilli")
    env["KERNELGEN_GPU_TARGET"] = GPU_TARGET
    return env


def run_one_test(
    bench: Path,
    iree_compile: Path,
    test_dir: Path,
    warmup: int,
    timed: int,
    verify: bool,
):
    config_path = test_dir / "config.json"
    input_a = test_dir / "input_a.npy"
    input_b = test_dir / "input_b.npy"
    output_c = test_dir / "output_c.npy"

    if not config_path.exists():
        print(f"SKIP: {test_dir.name} — config.json not found", file=sys.stderr)
        return None, None

    with open(config_path) as f:
        config = json.load(f)

    has_inputs = input_a.exists() and input_b.exists()

    cmd = [
        str(bench),
        "--config",
        str(config_path),
        "--warmup",
        str(warmup),
        "--timed",
        str(timed),
    ]
    if has_inputs:
        cmd += ["--input-a", str(input_a), "--input-b", str(input_b)]
    if verify and has_inputs and output_c.exists():
        cmd += ["--reference", str(output_c)]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=fusilli_env(iree_compile),
    )

    if result.returncode != 0:
        print(f"FAIL: {test_dir.name} (exit code {result.returncode})", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return config, None

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"FAIL: {test_dir.name} — could not parse output", file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        return config, None

    return config, output


def main():
    parser = argparse.ArgumentParser(description="Run Fusilli GEMM benchmark")
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
    parser.add_argument(
        "--output", "-o", default=None, help="Write JSON results to file"
    )
    parser.add_argument(
        "--iree-compile",
        default=None,
        help="Path to iree-compile binary used by Fusilli's CLI backend",
    )
    args = parser.parse_args()

    if not args.test and not args.test_dir:
        parser.error("Provide --test or --test-dir")

    if args.build_dir is None:
        args.build_dir = os.path.expanduser("~/kernelGen/build/Release")

    build_dir = Path(args.build_dir).resolve()
    bench = find_bench_binary(build_dir)
    iree_compile = find_iree_compile(args.iree_compile)
    test_dirs = collect_test_dirs(args.test, args.test_dir)

    if not test_dirs:
        print("No tests found.", file=sys.stderr)
        sys.exit(1)

    print(f"{'=' * 60}")
    print(f"Fusilli GEMM benchmark — {len(test_dirs)} test(s)")
    print(f"{'=' * 60}")
    print(
        f"  {'Test':<20s}  {'Problem':>20s}  {'Types':<40s}  {'Time':>13s}  {'Perf':>13s}"
    )
    print(f"  {'-' * 20}  {'-' * 20}  {'-' * 40}  {'-' * 13}  {'-' * 13}")

    results = []
    failures = 0
    for test_dir in test_dirs:
        config, output = run_one_test(
            bench,
            iree_compile,
            test_dir,
            args.warmup,
            args.timed,
            args.verify,
        )
        if config is None:
            continue
        if output is None:
            failures += 1
            continue
        results.append(print_result(test_dir.name, config, output))

    print(f"{'=' * 60}")
    print(
        f"{len(results)} passed, {failures} failed, {len(test_dirs) - len(results) - failures} skipped"
    )

    if results:
        print(json.dumps(results, indent=2))
    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
