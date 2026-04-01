#!/usr/bin/env python3
"""
Analyze a GEMM profile JSON and identify performance bottlenecks.

GEMM-specific wrapper around profiling.rocprof — adds roofline analysis
(FLOPs/bytes for GEMM) and GEMM-aware report formatting.

Usage:
  python gemm/profiling/analyze.py profile.json
  python gemm/profiling/analyze.py profile.json --arch gfx1100
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from profiling.rocprof import analyze_counters, analyze_dispatch_info
from profiling.gpu_specs import GPU_SPECS


# ---------------------------------------------------------------------------
# GEMM-specific computations
# ---------------------------------------------------------------------------


def get_gemm_flops(config: dict) -> float:
    return 2.0 * config["M"] * config["N"] * config["K"]


def get_gemm_bytes(config: dict) -> float:
    dtype_sizes = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8}
    M, N, K = config["M"], config["N"], config["K"]
    sa = dtype_sizes.get(config["dtype_A"], 2)
    sb = dtype_sizes.get(config["dtype_B"], 2)
    sc = dtype_sizes.get(config["dtype_C"], 2)
    return M * K * sa + K * N * sb + M * N * sc


def analyze_timing(profile: dict, gpu: dict) -> dict:
    """GEMM roofline: achieved TFLOPS, arithmetic intensity, bound type."""
    stats = profile.get("trace", {}).get("stats", [])
    main_kernel = profile.get("main_kernel")
    config = profile["config"]

    gemm_stats = next((s for s in stats if s["kernel_name"] == main_kernel), None)
    if not gemm_stats:
        return {"error": "GEMM kernel stats not found"}

    avg_ns = gemm_stats["average_ns"]
    avg_us = avg_ns / 1000.0
    flops = get_gemm_flops(config)
    data_bytes = get_gemm_bytes(config)
    achieved_tflops = (flops / (avg_us * 1e-6)) / 1e12
    ai = flops / data_bytes

    dtype = config.get("dtype_A", "bf16")
    peak = gpu[
        f"peak_tflops_{'bf16' if dtype == 'bf16' else 'f16' if dtype == 'f16' else 'f32'}"
    ]
    ridge = peak * 1e12 / (gpu["peak_bandwidth_gbs"] * 1e9)

    return {
        "average_kernel_time_us": round(avg_us, 2),
        "achieved_tflops": round(achieved_tflops, 2),
        "peak_tflops": peak,
        "compute_efficiency_pct": round(achieved_tflops / peak * 100, 1),
        "arithmetic_intensity": round(ai, 1),
        "ridge_point": round(ridge, 1),
        "roofline_bound": "compute" if ai > ridge else "memory",
        "min_ns": gemm_stats["min_ns"],
        "max_ns": gemm_stats["max_ns"],
        "stddev_ns": round(gemm_stats["stddev"], 1),
        "timing_variance_pct": round(gemm_stats["stddev"] / avg_ns * 100, 1)
        if avg_ns > 0
        else 0,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(
    timing: dict,
    counter_analysis: dict,
    dispatch_info: dict,
    config: dict,
    provider: str,
) -> str:
    lines = []
    M, N, K = config["M"], config["N"], config["K"]
    lines.append(f"{'=' * 70}")
    lines.append(f"GEMM Bottleneck Analysis — {provider}")
    lines.append(f"{'=' * 70}")
    lines.append(
        f"  Problem: {M}x{N}x{K}  "
        f"({config['dtype_A']}/{config['dtype_B']}/{config['dtype_C']}  "
        f"compute={config['compute_type']})"
    )

    if "error" not in timing:
        lines.append("\n--- Timing ---")
        lines.append(f"  Kernel time:        {timing['average_kernel_time_us']:.1f} us")
        lines.append(
            f"  Achieved:           {timing['achieved_tflops']:.2f} TFLOPS "
            f"({timing['compute_efficiency_pct']:.1f}% of {timing['peak_tflops']} peak)"
        )
        lines.append(
            f"  Arithmetic intensity: {timing['arithmetic_intensity']:.1f} FLOP/byte "
            f"(ridge point: {timing['ridge_point']:.1f})"
        )
        lines.append(f"  Roofline bound:     {timing['roofline_bound']}")
        lines.append(
            f"  Timing variance:    {timing['timing_variance_pct']:.1f}% "
            f"(min={timing['min_ns']}ns, max={timing['max_ns']}ns)"
        )

    if dispatch_info and "vgpr_count" in dispatch_info:
        lines.append("\n--- Kernel Resources ---")
        lines.append(
            f"  VGPRs: {dispatch_info['vgpr_count']}  "
            f"SGPRs: {dispatch_info['sgpr_count']}  "
            f"LDS: {dispatch_info['lds_bytes']} bytes"
        )
        lines.append(
            f"  Workgroup: {dispatch_info['workgroup_size']}  "
            f"Grid: {dispatch_info['grid_size']}  "
            f"({dispatch_info['total_workgroups']} workgroups)"
        )
        lines.append(
            f"  VGPR-limited: {dispatch_info['vgpr_limited_waves_per_simd']} waves/SIMD  "
            f"LDS-limited: {dispatch_info['lds_limited_waves_per_cu']} waves/CU"
        )

    metrics = counter_analysis.get("metrics", {})
    has_pmc = any(
        k in metrics for k in ["sq_waves", "sq_busy_cycles", "lds_bank_conflicts"]
    )
    if has_pmc:
        lines.append("\n--- PMC Counters ---")
        if "sq_waves" in metrics:
            lines.append(f"  SQ_WAVES: {metrics['sq_waves']:.0f}")
        if "sq_busy_cycles" in metrics:
            lines.append(f"  SQ_BUSY_CYCLES: {metrics['sq_busy_cycles']:.0f}")
        if "cycles_per_wave" in metrics:
            lines.append(f"  Cycles/wave: {metrics['cycles_per_wave']:.0f}")
        if "lds_bank_conflicts" in metrics:
            lines.append(
                f"  LDS bank conflicts: {metrics['lds_bank_conflicts']:.0f}"
                f" ({metrics.get('lds_conflicts_per_wave', 0):.1f}/wave)"
            )

    if "instruction_mix" in metrics:
        mix = metrics["instruction_mix"]
        lines.append("\n--- Instruction Mix ---")
        lines.append(
            f"  VALU: {mix['valu_pct']:.1f}%  SALU: {mix['salu_pct']:.1f}%  "
            f"SMEM: {mix['smem_pct']:.1f}%  LDS: {mix['lds_pct']:.1f}%"
        )

    if "occupancy_measured" in metrics:
        occ = metrics["occupancy_measured"]
        lines.append("\n--- Measured Occupancy ---")
        lines.append(
            f"  {occ['mean_waves_per_active_cu']:.1f} waves/CU ({occ['occupancy_pct']:.1f}%)"
        )

    if "l2_cache" in metrics:
        lines.append("\n--- L2 Cache ---")
        lines.append(f"  Hit rate: {metrics['l2_cache']['hit_rate_pct']:.1f}%")

    if "mem_unit_busy_pct" in metrics:
        lines.append(f"  Memory unit busy: {metrics['mem_unit_busy_pct']:.1f}%")

    if "cu_activity" in metrics:
        lines.append("\n--- CU Activity ---")
        lines.append(f"  Busy: {metrics['cu_activity']['busy_pct']:.1f}%")

    all_findings = counter_analysis.get("findings", []) + dispatch_info.get(
        "findings", []
    )
    if all_findings:
        lines.append(f"\n{'=' * 70}")
        lines.append("BOTTLENECKS")
        lines.append(f"{'=' * 70}")
        for f in all_findings:
            icon = "!!" if f["severity"] == "high" else " >"
            lines.append(f"  {icon} [{f['category']}] {f['message']}")
            lines.append(f"     -> {f['suggestion']}")
    else:
        lines.append("\n  No significant bottlenecks detected.")

    lines.append(f"{'=' * 70}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a GEMM profile for bottlenecks"
    )
    parser.add_argument("profile", help="Path to profile JSON from profile.py")
    parser.add_argument("--arch", default="gfx1100")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    with open(args.profile) as f:
        profile = json.load(f)

    if args.arch not in GPU_SPECS:
        print(
            f"ERROR: Unknown arch '{args.arch}'. Known: {', '.join(GPU_SPECS.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    gpu = GPU_SPECS[args.arch]
    config = profile["config"]
    provider = profile["provider"]

    timing = analyze_timing(profile, gpu)
    ca = analyze_counters(profile.get("counters", {}), gpu)
    dispatches = profile.get("trace", {}).get("dispatches", [])
    di = analyze_dispatch_info(dispatches, profile.get("main_kernel"), gpu)

    print(format_report(timing, ca, di, config, provider))

    if args.output:
        analysis = {
            "provider": provider,
            "config": config,
            "arch": args.arch,
            "timing": timing,
            "counter_analysis": ca,
            "dispatch_info": di,
        }
        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2)
            f.write("\n")
        print(f"\nStructured analysis written to {args.output}")


if __name__ == "__main__":
    main()
