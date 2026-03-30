#!/usr/bin/env python3
"""Generate GEMM test data (npy files) from a config.json."""

import argparse
import json
import os

import ml_dtypes  # noqa: F401  (registers bfloat16 with numpy)
import numpy as np

DTYPE_MAP = {
    "f16": np.float16,
    "bf16": ml_dtypes.bfloat16,
    "f32": np.float32,
}


def get_numpy_dtype(dtype_str: str):
    return np.dtype(DTYPE_MAP[dtype_str])


def generate(config_path: str):
    with open(config_path) as f:
        cfg = json.load(f)

    M, N, K = cfg["M"], cfg["N"], cfg["K"]
    dtype_A = get_numpy_dtype(cfg["dtype_A"])
    dtype_B = get_numpy_dtype(cfg["dtype_B"])
    dtype_C = get_numpy_dtype(cfg["dtype_C"])
    transA = cfg.get("transA", False)
    transB = cfg.get("transB", False)
    alpha = cfg.get("alpha", 1.0)

    test_dir = os.path.dirname(os.path.abspath(config_path))

    # Generate random inputs in float32, then cast to operand types.
    rng = np.random.default_rng(42)
    A_f32 = rng.standard_normal((M, K)).astype(np.float32) * 0.01
    B_f32 = rng.standard_normal((K, N)).astype(np.float32) * 0.01

    A = A_f32.astype(dtype_A)
    B = B_f32.astype(dtype_B)

    # Compute reference in float32.
    A_compute = A.astype(np.float32)
    B_compute = B.astype(np.float32)
    if transA:
        A_compute = A_compute.T
    if transB:
        B_compute = B_compute.T
    C_f32 = alpha * (A_compute @ B_compute)
    C = C_f32.astype(dtype_C)

    np.save(os.path.join(test_dir, "input_a.npy"), A)
    np.save(os.path.join(test_dir, "input_b.npy"), B)
    np.save(os.path.join(test_dir, "output_c.npy"), C)

    print(f"Generated test data in {test_dir}")
    print(f"  A: {A.shape} {A.dtype}")
    print(f"  B: {B.shape} {B.dtype}")
    print(f"  C: {C.shape} {C.dtype}")


def main():
    parser = argparse.ArgumentParser(description="Generate GEMM test data")
    parser.add_argument("config", help="Path to config.json")
    args = parser.parse_args()
    generate(args.config)


if __name__ == "__main__":
    main()
