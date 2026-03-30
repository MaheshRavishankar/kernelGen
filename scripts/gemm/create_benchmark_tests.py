#!/usr/bin/env python3
"""Create benchmark test directories with config.json for representative GEMM shapes.

Small shapes also get .npy files generated for correctness verification.
Large shapes only get config.json — the bench binary initializes random data.
"""

import json
import os
import subprocess
import sys

# 15 representative shapes across arithmetic intensity bins.
# AI = 2*M*N*K / (2*(M*K + K*N + M*N))  [flops / bytes for bf16]
SHAPES = [
    # (name, M, N, K, description)
    # Very low AI (<10) — memory bound
    ("ai_very_low_tiny", 4, 384, 5, "Very low AI, tiny shape"),
    ("ai_very_low_small_square", 576, 576, 10, "Very low AI, small square-ish"),
    ("ai_very_low_small_wide", 576, 2304, 10, "Very low AI, small wide"),
    # Low AI (10-50) — memory bound
    ("ai_low_skinny", 16, 512, 1024, "Low AI, skinny M"),
    ("ai_low_small", 32, 576, 2304, "Low AI, small"),
    ("ai_low_large_flat", 21760, 3840, 20, "Low AI, large M*N but tiny K"),
    # Medium AI (50-200) — transitional
    ("ai_medium_small", 576, 576, 165, "Medium AI, small"),
    ("ai_medium_large", 7680, 512, 304, "Medium AI, larger"),
    ("ai_medium_extreme", 16800000, 128, 134, "Medium AI, extreme M"),
    # High AI (200-1000) — approaching compute bound
    ("ai_high_small", 576, 576, 1280, "High AI, small"),
    ("ai_high_medium", 1285, 2048, 3840, "High AI, medium"),
    ("ai_high_large_k", 4096, 1024, 150000, "High AI, very large K"),
    # Very high AI (>1000) — compute bound
    ("ai_very_high_medium", 3840, 3840, 2304, "Very high AI, medium"),
    ("ai_very_high_large", 11520, 3840, 3840, "Very high AI, large"),
    ("ai_very_high_extreme", 150000, 16384, 4096, "Very high AI, extreme"),
]

# Threshold: if total memory for A+B+C exceeds this, skip .npy generation.
MAX_NPY_BYTES = 50 * 1024 * 1024  # 50 MB


def compute_ai(M, N, K):
    flops = 2.0 * M * N * K
    bytes_moved = 2 * (M * K + K * N + M * N)  # bf16 = 2 bytes per element
    return flops / bytes_moved


def total_bytes(M, N, K):
    return 2 * (M * K + K * N + M * N)


def main():
    test_root = os.path.dirname(os.path.abspath(__file__))
    generate_script = os.path.join(test_root, "generate_test.py")

    for name, M, N, K, desc in SHAPES:
        ai = compute_ai(M, N, K)
        test_dir = os.path.join(test_root, name)
        os.makedirs(test_dir, exist_ok=True)

        config = {
            "operation": "gemm",
            "M": M,
            "N": N,
            "K": K,
            "dtype_A": "bf16",
            "dtype_B": "bf16",
            "dtype_C": "bf16",
            "compute_type": "f32",
            "transA": False,
            "transB": False,
            "alpha": 1.0,
            "beta": 0.0,
            "description": f"{desc} (AI={ai:.1f})",
        }

        config_path = os.path.join(test_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

        nbytes = total_bytes(M, N, K)
        if nbytes <= MAX_NPY_BYTES:
            print(f"  {name}: M={M}, N={N}, K={K}, AI={ai:.1f} — generating .npy files")
            subprocess.run(
                [sys.executable, generate_script, config_path],
                check=True,
            )
        else:
            size_mb = nbytes / (1024 * 1024)
            print(
                f"  {name}: M={M}, N={N}, K={K}, AI={ai:.1f} — config only ({size_mb:.0f} MB, too large for .npy)"
            )

    print(f"\nCreated {len(SHAPES)} test directories in {test_root}")


if __name__ == "__main__":
    main()
