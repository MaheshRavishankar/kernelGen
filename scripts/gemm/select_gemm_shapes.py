#!/usr/bin/env python3
"""Select representative GEMM shapes covering different arithmetic intensities."""

import csv
import re


def parse_mm_shapes(args_str):
    """Extract M, N, K from aten::mm argument string."""
    # Match the two matrix shapes: [M, K], [K, N]
    match = re.search(r"\[\[(\d+),\s*(\d+)\],\s*\[(\d+),\s*(\d+)\]\]", args_str)
    if not match:
        return None
    M, K1, K2, N = (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
    )
    assert K1 == K2, f"K mismatch: {K1} vs {K2}"
    return M, N, K1


def arithmetic_intensity(M, N, K):
    """Compute arithmetic intensity = flops / bytes for bf16."""
    flops = 2 * M * N * K
    bytes_transferred = 2 * (M * K + K * N + M * N)  # 2 bytes per bf16 element
    return flops / bytes_transferred


def main():
    csv_path = (
        "/home/mahesh/kernelGen/kernelGen/rdna4_gemm_iree_hipblaslt_comparison.csv"
    )

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            args_str = row[0]
            if not args_str.startswith("aten::mm"):
                continue
            shapes = parse_mm_shapes(args_str)
            if shapes is None:
                continue
            M, N, K = shapes
            ai = arithmetic_intensity(M, N, K)
            total_flops = 2 * M * N * K
            try:
                hipblaslt_mean = float(row[1])
            except ValueError:
                hipblaslt_mean = float("nan")
            try:
                iree_mean = float(row[2])
            except ValueError:
                iree_mean = float("nan")
            try:
                ratio = float(row[3])
            except ValueError:
                ratio = float("nan")
            rows.append(
                {
                    "M": M,
                    "N": N,
                    "K": K,
                    "ai": ai,
                    "total_flops": total_flops,
                    "hipblaslt_us": hipblaslt_mean,
                    "iree_us": iree_mean,
                    "ratio": ratio,
                }
            )

    print(f"Total aten::mm shapes parsed: {len(rows)}")
    print()

    # Define bins by arithmetic intensity
    bins = [
        ("very_low", 0, 10),
        ("low", 10, 50),
        ("medium", 50, 200),
        ("high", 200, 1000),
        ("very_high", 1000, float("inf")),
    ]

    # Group into bins
    binned = {name: [] for name, _, _ in bins}
    for r in rows:
        for name, lo, hi in bins:
            if lo <= r["ai"] < hi:
                binned[name].append(r)
                break

    print("Bin distribution:")
    for name, _, _ in bins:
        print(f"  {name}: {len(binned[name])} shapes")
    print()

    # From each bin, pick 2-3 representatives varying in total problem size
    selected = []
    for name, _, _ in bins:
        shapes = binned[name]
        if not shapes:
            continue
        # Sort by total_flops
        shapes.sort(key=lambda r: r["total_flops"])
        n = len(shapes)
        if n == 1:
            picks = [0]
        elif n == 2:
            picks = [0, n - 1]
        else:
            # Pick smallest, median, largest
            picks = [0, n // 2, n - 1]
        for idx in picks:
            entry = shapes[idx].copy()
            entry["bin"] = name
            selected.append(entry)

    # Print results
    print(f"Selected {len(selected)} representative shapes:\n")
    print(
        f"{'Bin':<12} {'M':>7} {'N':>7} {'K':>7} {'ArithInt':>10} {'TotalGFLOPs':>13} {'hipblaslt_us':>13} {'iree_us':>10} {'ratio':>7}"
    )
    print("-" * 100)
    for s in selected:
        gflops = s["total_flops"] / 1e9
        print(
            f"{s['bin']:<12} {s['M']:>7} {s['N']:>7} {s['K']:>7} {s['ai']:>10.1f} {gflops:>13.3f} {s['hipblaslt_us']:>13.1f} {s['iree_us']:>10.1f} {s['ratio']:>7.2f}"
        )

    # Also print in a copy-paste friendly format for test config
    print("\n\nShapes for test configs (M, N, K):")
    for s in selected:
        print(
            f"  M={s['M']}, N={s['N']}, K={s['K']}  # bin={s['bin']}, AI={s['ai']:.1f}"
        )


if __name__ == "__main__":
    main()
