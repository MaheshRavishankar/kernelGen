#!/usr/bin/env python3
"""Roofline analysis for GEMM benchmark results.

Compares measured kernel performance against the theoretical roofline
for a given GPU architecture.

Usage:
  # Using a known architecture:
  python gemm/roofline.py results.json --arch gfx1100

  # Using custom specs:
  python gemm/roofline.py results.json --peak-tflops 123 --peak-bw 864
"""

import argparse
import json

# Known GPU architectures: (peak BF16 TFLOPS, peak memory BW in GB/s).
KNOWN_ARCHS = {
    "gfx1100": (123.0, 864.0),  # Radeon PRO W7900 / RX 7900 XTX
    "gfx1101": (92.0, 576.0),  # Radeon RX 7900 GRE / PRO W7800
    "gfx942": (1307.4, 5300.0),  # MI300X
}


def arithmetic_intensity(M, N, K, dtype_bytes=2):
    """Compute arithmetic intensity in FLOP/byte."""
    flops = 2.0 * M * N * K
    bytes_moved = dtype_bytes * (M * K + K * N + M * N)
    return flops / bytes_moved


def roofline_peak(ai, peak_tflops, peak_bw_gbs):
    """Theoretical peak TFLOPS at a given arithmetic intensity."""
    bw_limited = ai * peak_bw_gbs / 1e3  # TFLOPS
    return min(bw_limited, peak_tflops)


def analyze_results(results_json, peak_tflops, peak_bw_gbs):
    """Analyze benchmark results against the roofline model."""
    ridge_point = (peak_tflops * 1e12) / (peak_bw_gbs * 1e9)

    with open(results_json) as f:
        results = json.load(f)

    header = (
        f"{'Test':<30s} {'M':>8s} {'N':>8s} {'K':>8s} "
        f"{'AI':>7s} {'Measured':>10s} {'Roofline':>10s} {'Efficiency':>10s}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for r in results:
        M, N, K = r["M"], r["N"], r["K"]
        name = r["test"]

        kernel_us = r.get("kernel_time_us")
        if kernel_us is None or kernel_us <= 0:
            print(f"{name:<30s} {'SKIPPED — no timing data':>60s}")
            continue

        flops = 2.0 * M * N * K
        measured_tflops = flops / (kernel_us * 1e-6) / 1e12

        ai = arithmetic_intensity(M, N, K)
        roof = roofline_peak(ai, peak_tflops, peak_bw_gbs)
        efficiency = measured_tflops / roof * 100 if roof > 0 else 0

        bound = "compute" if ai > ridge_point else "memory"

        print(
            f"{name:<30s} {M:>8d} {N:>8d} {K:>8d} "
            f"{ai:>7.1f} {measured_tflops:>9.2f}T {roof:>9.2f}T {efficiency:>8.1f}%"
        )
        rows.append(
            {
                "test": name,
                "M": M,
                "N": N,
                "K": K,
                "arithmetic_intensity": round(ai, 1),
                "measured_tflops": round(measured_tflops, 3),
                "roofline_tflops": round(roof, 3),
                "efficiency_pct": round(efficiency, 1),
                "bound": bound,
            }
        )

    print()
    print(
        f"Roofline model: peak {peak_tflops} TFLOPS BF16, "
        f"{peak_bw_gbs} GB/s BW, ridge point {ridge_point:.0f} FLOP/byte"
    )

    return rows


def main():
    parser = argparse.ArgumentParser(description="GEMM roofline analysis")
    parser.add_argument("results", help="Path to JSON results file from run.py")
    parser.add_argument(
        "--arch",
        choices=sorted(KNOWN_ARCHS.keys()),
        help=f"GPU architecture (known: {', '.join(sorted(KNOWN_ARCHS.keys()))})",
    )
    parser.add_argument(
        "--peak-tflops", type=float, help="Peak BF16 TFLOPS (overrides --arch)"
    )
    parser.add_argument(
        "--peak-bw", type=float, help="Peak memory bandwidth in GB/s (overrides --arch)"
    )
    parser.add_argument("--output", "-o", help="Write analysis to JSON file")
    args = parser.parse_args()

    if args.arch:
        peak_tflops, peak_bw_gbs = KNOWN_ARCHS[args.arch]
        if args.peak_tflops is not None:
            peak_tflops = args.peak_tflops
        if args.peak_bw is not None:
            peak_bw_gbs = args.peak_bw
    elif args.peak_tflops is not None and args.peak_bw is not None:
        peak_tflops = args.peak_tflops
        peak_bw_gbs = args.peak_bw
    else:
        parser.error(
            "Specify --arch for a known GPU, or both --peak-tflops and --peak-bw"
        )

    rows = analyze_results(args.results, peak_tflops, peak_bw_gbs)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)
            f.write("\n")
        print(f"\nAnalysis written to {args.output}")


if __name__ == "__main__":
    main()
