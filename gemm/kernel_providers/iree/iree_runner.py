"""IREE GEMM benchmark runner utilities."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MLIR_DTYPE_MAP = {"f16": "f16", "bf16": "bf16", "f32": "f32"}

GPU_TARGET = os.environ.get("KERNELGEN_GPU_TARGET", "gfx1100")
IREE_OPT_LEVEL = "O3"


def find_bench_binary(build_dir: Path, provider_name: str) -> Path:
    binary = (
        build_dir
        / "gemm"
        / "kernel_providers"
        / provider_name
        / f"{provider_name}_gemm_bench"
    )
    if not binary.exists():
        print(f"ERROR: Benchmark binary not found: {binary}", file=sys.stderr)
        print("  Build the project first with the appropriate preset.", file=sys.stderr)
        sys.exit(1)
    return binary


def find_iree_compile(iree_compile_arg: str | None) -> Path:
    """Locate iree-compile binary."""
    if iree_compile_arg:
        p = Path(iree_compile_arg)
        if p.exists():
            return p

    which = shutil.which("iree-compile")
    if which:
        return Path(which)

    for candidate in [
        Path.home() / "kernelGen" / "iree" / "build" / "tools" / "iree-compile",
        Path.home() / ".local" / "bin" / "iree-compile",
    ]:
        if candidate.exists():
            return candidate

    print(
        "ERROR: iree-compile not found. Install via "
        "'pip install iree-base-compiler' or set --iree-compile.",
        file=sys.stderr,
    )
    sys.exit(1)


def generate_mlir(config: dict) -> str:
    """Generate MLIR for a GEMM operation from config."""
    M, N, K = config["M"], config["N"], config["K"]
    dtype_A = MLIR_DTYPE_MAP[config.get("dtype_A", "f16")]
    dtype_B = MLIR_DTYPE_MAP[config.get("dtype_B", "f16")]
    dtype_C = MLIR_DTYPE_MAP[config.get("dtype_C", "f16")]
    transA = config.get("transA", False)
    transB = config.get("transB", False)

    if transA:
        a_shape = f"{K}x{M}"
        matmul_op = "linalg.matmul_transpose_a"
    else:
        a_shape = f"{M}x{K}"
        matmul_op = "linalg.matmul"

    if transB:
        b_shape = f"{N}x{K}"
        if transA:
            matmul_op = "linalg.matmul_transpose_a_transpose_b"
        else:
            matmul_op = "linalg.matmul_transpose_b"
    else:
        b_shape = f"{K}x{N}"

    return f"""\
func.func @main(%a: tensor<{a_shape}x{dtype_A}>, %b: tensor<{b_shape}x{dtype_B}>) -> tensor<{M}x{N}x{dtype_C}> {{
  %cst = arith.constant 0.000000e+00 : {dtype_C}
  %empty = tensor.empty() : tensor<{M}x{N}x{dtype_C}>
  %fill = linalg.fill ins(%cst : {dtype_C}) outs(%empty : tensor<{M}x{N}x{dtype_C}>) -> tensor<{M}x{N}x{dtype_C}>
  %result = {matmul_op} ins(%a, %b : tensor<{a_shape}x{dtype_A}>, tensor<{b_shape}x{dtype_B}>) outs(%fill : tensor<{M}x{N}x{dtype_C}>) -> tensor<{M}x{N}x{dtype_C}>
  return %result : tensor<{M}x{N}x{dtype_C}>
}}
"""


def compile_mlir(
    iree_compile: Path, mlir_text: str, vmfb_path: Path, gpu_target: str
) -> bool:
    """Compile MLIR to .vmfb. Returns True on success."""
    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(mlir_text)
        mlir_path = f.name

    try:
        cmd = [
            str(iree_compile),
            mlir_path,
            f"--iree-opt-level={IREE_OPT_LEVEL}",
            "--iree-hal-target-backends=rocm",
            f"--iree-hip-target={gpu_target}",
            "-o",
            str(vmfb_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  iree-compile failed:\n{result.stderr}", file=sys.stderr)
            return False
        return True
    finally:
        os.unlink(mlir_path)


def run_one_test(
    bench: Path,
    iree_compile: Path,
    test_dir: Path,
    warmup: int,
    timed: int,
    verify: bool,
    gpu_target: str,
    vmfb_cache_namespace: str,
):
    """Run a single test. Returns (config, output) or (config, None) on failure."""
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

    cache_dir = Path(
        os.environ.get("KERNELGEN_CACHE_DIR", Path.home() / ".cache" / "kernelgen")
    )
    vmfb_cache = cache_dir / "vmfb" / vmfb_cache_namespace / gpu_target / IREE_OPT_LEVEL
    vmfb_cache.mkdir(parents=True, exist_ok=True)
    vmfb_path = vmfb_cache / f"{test_dir.name}.vmfb"
    if not vmfb_path.exists():
        mlir_text = generate_mlir(config)
        print(f"  Compiling {test_dir.name}...", file=sys.stderr, end=" ")
        if not compile_mlir(iree_compile, mlir_text, vmfb_path, gpu_target):
            print("FAILED", file=sys.stderr)
            return config, None
        print("OK", file=sys.stderr)

    cmd = [
        str(bench),
        "--config",
        str(config_path),
        "--vmfb",
        str(vmfb_path),
        "--warmup",
        str(warmup),
        "--timed",
        str(timed),
    ]
    if has_inputs:
        cmd += ["--input-a", str(input_a), "--input-b", str(input_b)]
    if verify and has_inputs and output_c.exists():
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
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        return config, None

    return config, output


def print_result(test_name: str, config: dict, output: dict):
    M, N, K = config["M"], config["N"], config["K"]
    time_us = output.get("kernel_time_us", 0)
    flops = 2.0 * M * N * K
    tflops = (flops / (time_us * 1e-6)) / 1e12 if time_us > 0 else 0

    dtype_str = (
        f"A={config['dtype_A']} B={config['dtype_B']} "
        f"C={config['dtype_C']} compute={config['compute_type']}"
    )
    verify_str = ""
    if "verified" in output:
        verify_str = " PASS" if output["verified"] else " FAIL"

    print(
        f"  {test_name:<20s}  {M:>6d}x{N:>6d}x{K:>6d}  "
        f"{dtype_str:<40s}  {time_us:>10.1f} us  {tflops:>7.2f} TFLOPS{verify_str}"
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


def run_provider(provider_name: str, benchmark_title: str, description: str):
    parser = argparse.ArgumentParser(description=description)
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
        help="Path to iree-compile binary",
    )
    parser.add_argument(
        "--gpu-target",
        default=GPU_TARGET,
        help=f"GPU target for iree-compile (default: {GPU_TARGET})",
    )
    args = parser.parse_args()

    if not args.test and not args.test_dir:
        parser.error("Provide --test or --test-dir")

    if args.build_dir is None:
        args.build_dir = os.path.expanduser("~/kernelGen/build/Release")

    build_dir = Path(args.build_dir).resolve()
    bench = find_bench_binary(build_dir, provider_name)
    iree_compile = find_iree_compile(args.iree_compile)
    test_dirs = collect_test_dirs(args.test, args.test_dir)

    if not test_dirs:
        print("No tests found.", file=sys.stderr)
        sys.exit(1)

    print(f"{'=' * 60}")
    print(f"{benchmark_title} GEMM benchmark — {len(test_dirs)} test(s)")
    print(f"{'=' * 60}")
    print(
        f"  {'Test':<20s}  {'Problem':>20s}  {'Types':<40s}  "
        f"{'Time':>13s}  {'Perf':>13s}"
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
            args.gpu_target,
            provider_name,
        )
        if config is None:
            continue
        if output is None:
            failures += 1
            continue
        results.append(print_result(test_dir.name, config, output))

    print(f"{'=' * 60}")
    print(
        f"{len(results)} passed, {failures} failed, "
        f"{len(test_dirs) - len(results) - failures} skipped"
    )

    if results:
        print(json.dumps(results, indent=2))
    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
        print(f"Results written to {args.output}")
