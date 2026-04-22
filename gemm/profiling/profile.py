#!/usr/bin/env python3
"""
Profile a GEMM kernel using rocprofv3.

Thin GEMM-specific wrapper around profiling.rocprof — handles provider
binary discovery, IREE vmfb compilation, and GEMM config loading.

Usage:
  python gemm/profiling/profile.py --provider hipblaslt --test gemm/tests/ai_high_small
  python gemm/profiling/profile.py --provider iree --test gemm/tests/ai_high_small --gpu-target gfx1100
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from profiling.rocprof import (
    find_rocprof,
    run_kernel_trace,
    run_pmc_collection,
    identify_main_kernel,
    build_profile,
    PMC_GROUPS,
    PMC_GROUPS_EXTENDED,
)


def find_bench_binary(build_dir: Path, provider: str) -> Path:
    binaries = {
        "hipblaslt": build_dir
        / "gemm"
        / "kernel_providers"
        / "hipblaslt"
        / "hipblaslt_gemm_bench",
        "iree": build_dir / "gemm" / "kernel_providers" / "iree" / "iree_gemm_bench",
        "native_hip": build_dir
        / "gemm"
        / "kernel_providers"
        / "native_hip"
        / "native_hip_gemm_bench",
    }
    binary = binaries.get(provider)
    if not binary:
        print(f"ERROR: Unknown provider '{provider}'", file=sys.stderr)
        sys.exit(1)
    if not binary.exists():
        print(f"ERROR: Benchmark binary not found: {binary}", file=sys.stderr)
        sys.exit(1)
    return binary


def build_bench_cmd(
    bench: Path,
    test_dir: Path,
    provider: str,
    warmup: int,
    timed: int,
    vmfb_path: str | None,
) -> list[str]:
    config_path = test_dir / "config.json"
    cmd = [
        str(bench),
        "--config",
        str(config_path),
        "--warmup",
        str(warmup),
        "--timed",
        str(timed),
    ]

    input_a = test_dir / "input_a.npy"
    input_b = test_dir / "input_b.npy"
    if input_a.exists() and input_b.exists():
        cmd += ["--input-a", str(input_a), "--input-b", str(input_b)]

    if provider == "iree" and vmfb_path:
        cmd += ["--vmfb", str(vmfb_path)]
    return cmd


def compile_iree_vmfb(test_dir: Path, gpu_target: str) -> str:
    """Compile MLIR to vmfb for IREE provider. Returns vmfb path."""
    cache_dir = (
        Path(
            os.environ.get("KERNELGEN_CACHE_DIR", Path.home() / ".cache" / "kernelgen")
        )
        / "vmfb"
        / "iree"
        / gpu_target
        / "O3"
    )

    # Keep profile.py aligned with gemm/kernel_providers/iree/iree_runner.py.
    vmfb_path = cache_dir / f"{test_dir.name}.vmfb"
    if vmfb_path.exists():
        return str(vmfb_path)

    iree_run = (
        Path(__file__).resolve().parent.parent / "kernel_providers" / "iree" / "run.py"
    )
    if not iree_run.exists():
        print("ERROR: IREE run.py not found for vmfb compilation", file=sys.stderr)
        sys.exit(1)

    print(
        f"  Compiling VMFB for {test_dir.name} (gpu_target={gpu_target})...",
        file=sys.stderr,
    )
    cmd = [
        sys.executable,
        str(iree_run),
        "--test",
        str(test_dir),
        "--gpu-target",
        gpu_target,
        "--timed",
        "0",
        "--warmup",
        "0",
    ]
    iree_compile = shutil.which("iree-compile")
    if not iree_compile:
        venv_iree_compile = Path(sys.executable).resolve().parent / "iree-compile"
        if venv_iree_compile.exists():
            iree_compile = str(venv_iree_compile)
    if iree_compile:
        cmd += ["--iree-compile", iree_compile]

    subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if vmfb_path.exists():
        return str(vmfb_path)
    print("ERROR: Failed to compile VMFB", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Profile a GEMM kernel using rocprofv3"
    )
    parser.add_argument(
        "--provider", required=True, choices=["hipblaslt", "iree", "native_hip"]
    )
    parser.add_argument("--test", required=True, help="Path to test directory")
    parser.add_argument("--build-dir", default=None)
    parser.add_argument("--rocprof", default=None, help="Path to rocprofv3 binary")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timed", type=int, default=20)
    parser.add_argument("--gpu-target", default="gfx1100")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--skip-pmc", action="store_true")
    parser.add_argument(
        "--extended-pmc",
        action="store_true",
        help="Collect all PMC groups (7 passes, most return 0 on gfx1100)",
    )
    args = parser.parse_args()

    test_dir = Path(args.test).resolve()
    if not (test_dir / "config.json").exists():
        print(f"ERROR: No config.json in {test_dir}", file=sys.stderr)
        sys.exit(1)

    with open(test_dir / "config.json") as f:
        config = json.load(f)

    build_dir = Path(
        args.build_dir or os.path.expanduser("~/kernelGen/build/Release")
    ).resolve()
    rocprof = find_rocprof(args.rocprof)
    bench = find_bench_binary(build_dir, args.provider)

    vmfb_path = None
    if args.provider == "iree":
        vmfb_path = compile_iree_vmfb(test_dir, args.gpu_target)

    bench_cmd = build_bench_cmd(
        bench, test_dir, args.provider, args.warmup, args.timed, vmfb_path
    )

    M, N, K = config["M"], config["N"], config["K"]
    print(
        f"Profiling {args.provider} GEMM: {M}x{N}x{K} "
        f"({config['dtype_A']}/{config['dtype_B']}/{config['dtype_C']})",
        file=sys.stderr,
    )

    with tempfile.TemporaryDirectory(prefix="kernelgen_profile_") as tmpdir:
        tmpdir = Path(tmpdir)

        print("  Collecting kernel trace...", file=sys.stderr)
        trace = run_kernel_trace(rocprof, bench_cmd, tmpdir / "trace")

        counters = {}
        if not args.skip_pmc:
            groups = PMC_GROUPS_EXTENDED if args.extended_pmc else PMC_GROUPS
            print(
                f"  Collecting PMC counters ({len(groups)} pass(es))...",
                file=sys.stderr,
            )
            counters = run_pmc_collection(
                rocprof, bench_cmd, tmpdir / "pmc", extended=args.extended_pmc
            )

    main_kernel = identify_main_kernel(trace.get("stats", []))
    if main_kernel:
        print(f"  GEMM kernel: {main_kernel[:80]}...", file=sys.stderr)
    else:
        print("  WARNING: Could not identify GEMM kernel", file=sys.stderr)

    profile = build_profile(config, trace, counters, main_kernel, args.provider)

    output_json = json.dumps(profile, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json + "\n")
        print(f"  Profile written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
