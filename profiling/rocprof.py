#!/usr/bin/env python3
"""
Generic rocprofv3 profiling infrastructure.

Wraps rocprofv3 to collect kernel dispatch traces and hardware performance
counters (PMC) via multi-pass profiling. Parses CSV output into structured
dicts. Operation-agnostic — callers provide the benchmark command.

Typical usage:
    from profiling.rocprof import (
        find_rocprof, run_kernel_trace, run_pmc_collection,
        identify_main_kernel, PMC_GROUPS, PMC_GROUPS_EXTENDED,
    )
"""

import csv
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# PMC counter groups validated for gfx1100 via rocprofv3-avail pmc-check.
# Each group can be collected in a single hardware pass.
# rocprofv3 runs the application once per group (multi-pass).
#
# NOTE: On gfx1100 (RDNA3), most per-dispatch PMC counters return 0.
# The counters confirmed to report non-zero values per-dispatch:
#   SQ_WAVES, SQ_BUSY_CYCLES, LDSBankConflict
# Many SQ instruction counters, GL2C, memory bandwidth counters report 0
# in per-dispatch mode. This is a known rocprofiler-sdk limitation on RDNA3.
# ---------------------------------------------------------------------------
PMC_GROUPS = [
    # Pass 1: Wave counts + LDS conflicts (confirmed working on gfx1100)
    ["SQ_WAVES", "SQ_BUSY_CYCLES", "LDSBankConflict"],
]

# Extended counter groups — most return 0 on gfx1100 per-dispatch but may
# work on CDNA (gfx90a, gfx942) or future rocprofiler-sdk versions.
PMC_GROUPS_EXTENDED = [
    [
        "GPUBusy",
        "SQ_WAVES",
        "SQ_INSTS_VALU",
        "SQ_INSTS_SALU",
        "SQ_INSTS_LDS",
        "SQ_INSTS_SMEM",
    ],
    ["VALUInsts", "SALUInsts", "SFetchInsts", "LDSBankConflict", "ALUStalledByLDS"],
    ["FETCH_SIZE", "MemUnitBusy"],
    ["WRITE_SIZE"],
    ["GL2C_HIT", "GL2C_MISS", "GL2C_MC_RDREQ", "GL2C_MC_WRREQ"],
    ["SQ_BUSY_CYCLES", "SQ_WAVE_CYCLES", "SQ_WAIT_ANY"],
    ["MeanOccupancyPerActiveCU"],
]


# ---------------------------------------------------------------------------
# rocprofv3 discovery
# ---------------------------------------------------------------------------


def find_rocprof(rocprof_path: str | None = None) -> Path:
    """Locate the rocprofv3 binary."""
    if rocprof_path:
        p = Path(rocprof_path)
        if p.exists():
            return p
        print(f"ERROR: rocprofv3 not found at {p}", file=sys.stderr)
        sys.exit(1)
    # Default: TheRock install
    default = Path.home() / "kernelGen" / "TheRock" / "bin" / "rocprofv3"
    if default.exists():
        return default
    # Try PATH
    which = shutil.which("rocprofv3")
    if which:
        return Path(which)
    print("ERROR: rocprofv3 not found. Use --rocprof or add to PATH.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_rocprof_temps(cwd: Path | None = None) -> None:
    """Remove .rocprofv3/ temp directory that rocprofv3 drops in cwd.

    rocprofv3 PMC collection creates .rocprofv3/<pid>-*-counter_values.dat
    files in the process working directory. These are binary counter dumps
    not needed after CSV output is generated.
    """
    target = (cwd or Path.cwd()) / ".rocprofv3"
    if target.is_dir():
        shutil.rmtree(target)


# ---------------------------------------------------------------------------
# Trace + stats collection
# ---------------------------------------------------------------------------


def run_kernel_trace(rocprof: Path, bench_cmd: list[str], output_dir: Path) -> dict:
    """Run rocprofv3 with --kernel-trace --stats to get dispatch timing.

    Returns {"dispatches": [...], "stats": [...]}.
    """
    cmd = [
        str(rocprof),
        "--kernel-trace",
        "--stats",
        "-d",
        str(output_dir),
        "-f",
        "csv",
        "--",
    ] + bench_cmd

    result = subprocess.run(cmd, capture_output=True, text=True)
    cleanup_rocprof_temps()
    if result.returncode != 0:
        print(
            f"ERROR: rocprofv3 kernel trace failed:\n{result.stderr}", file=sys.stderr
        )
        return {}

    # rocprofv3 puts output under <hostname>/<pid>_*.csv
    trace_data: dict = {"dispatches": [], "stats": []}
    for csv_file in output_dir.rglob("*_kernel_trace.csv"):
        trace_data["dispatches"] = _parse_kernel_trace(csv_file)
    for csv_file in output_dir.rglob("*_kernel_stats.csv"):
        trace_data["stats"] = _parse_kernel_stats(csv_file)
    return trace_data


# ---------------------------------------------------------------------------
# PMC counter collection
# ---------------------------------------------------------------------------


def run_pmc_collection(
    rocprof: Path, bench_cmd: list[str], output_dir: Path, extended: bool = False
) -> dict:
    """Run rocprofv3 with multi-pass PMC counter collection.

    Returns {kernel_name: {"num_dispatches": N, "counters": {name: avg_value}}}.
    """
    groups = PMC_GROUPS_EXTENDED if extended else PMC_GROUPS
    pmc_args: list[str] = []
    for group in groups:
        pmc_args += ["--pmc", " ".join(group)]

    cmd = (
        [str(rocprof), "--kernel-trace"]
        + pmc_args
        + ["-d", str(output_dir), "-f", "csv", "--"]
        + bench_cmd
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    cleanup_rocprof_temps()
    if result.returncode != 0:
        print(
            f"ERROR: rocprofv3 PMC collection failed:\n{result.stderr}", file=sys.stderr
        )
        return {}

    # Merge counter data from all passes
    all_counters: dict[str, dict[str, float]] = {}  # dispatch_id -> {counter: value}
    kernel_names: dict[str, str] = {}  # dispatch_id -> kernel name

    for csv_file in sorted(output_dir.rglob("*_counter_collection.csv")):
        for row in _parse_counter_csv(csv_file):
            did = row["Dispatch_Id"]
            name = row["Kernel_Name"]
            counter = row["Counter_Name"]
            value = row["Counter_Value"]

            if did not in all_counters:
                all_counters[did] = {}
                kernel_names[did] = name
            all_counters[did][counter] = value

    return _merge_counters_by_kernel(all_counters, kernel_names)


# ---------------------------------------------------------------------------
# Kernel identification
# ---------------------------------------------------------------------------

# Kernels to ignore when identifying the "main" kernel from stats.
_SKIP_KERNELS = frozenset(
    {
        "__amd_rocclr_fillBufferAligned",
        "__amd_rocclr_fillBuffer",
    }
)


def identify_main_kernel(
    stats: list[dict], skip: frozenset[str] | None = None
) -> str | None:
    """Find the main kernel (highest time %, excluding fill/memset).

    Works for any operation — not GEMM-specific.
    """
    skip = skip or _SKIP_KERNELS
    candidates = [s for s in stats if s["kernel_name"] not in skip]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s["percentage"])["kernel_name"]


# ---------------------------------------------------------------------------
# Profile assembly
# ---------------------------------------------------------------------------


def build_profile(
    config: dict, trace: dict, counters: dict, main_kernel: str | None, provider: str
) -> dict:
    """Assemble the final profile dict.

    Generic — the caller supplies config/provider, this function just
    attaches trace and counter data.
    """
    profile: dict = {
        "provider": provider,
        "config": config,
        "main_kernel": main_kernel,
        "trace": trace,
    }

    if main_kernel and main_kernel in counters:
        profile["counters"] = counters[main_kernel]["counters"]
        profile["num_counter_dispatches"] = counters[main_kernel]["num_dispatches"]
    elif counters:
        profile["counters_all"] = counters

    return profile


# ---------------------------------------------------------------------------
# Analysis helpers (operation-agnostic)
# ---------------------------------------------------------------------------


def analyze_counters(counters: dict, gpu: dict) -> dict:
    """Analyze PMC counters to identify bottlenecks.

    Returns {"metrics": {...}, "findings": [...]}.
    On gfx1100: only SQ_WAVES, SQ_BUSY_CYCLES, LDSBankConflict report data.
    Extended counters are included when non-zero (CDNA or future SDK).
    """
    if not counters:
        return {"note": "No PMC counter data available"}

    findings: list[dict] = []
    metrics: dict = {}

    # --- Counters confirmed working on gfx1100 ---

    sq_waves = counters.get("SQ_WAVES", 0)
    if sq_waves > 0:
        metrics["sq_waves"] = round(sq_waves, 0)

    busy_cycles = counters.get("SQ_BUSY_CYCLES", 0)
    if busy_cycles > 0:
        metrics["sq_busy_cycles"] = round(busy_cycles, 0)
        if sq_waves > 0:
            metrics["cycles_per_wave"] = round(busy_cycles / sq_waves, 0)

    lds_conflicts = counters.get("LDSBankConflict", 0)
    if lds_conflicts > 0:
        metrics["lds_bank_conflicts"] = round(lds_conflicts, 0)
        if sq_waves > 0:
            cpw = lds_conflicts / sq_waves
            metrics["lds_conflicts_per_wave"] = round(cpw, 2)
            if cpw > 10:
                findings.append(
                    {
                        "severity": "high",
                        "category": "lds_conflicts",
                        "message": f"High LDS bank conflicts ({cpw:.1f}/wave, {lds_conflicts:.0f} total).",
                        "suggestion": "Pad LDS allocations or change access strides.",
                    }
                )
            elif cpw > 2:
                findings.append(
                    {
                        "severity": "medium",
                        "category": "lds_conflicts",
                        "message": f"Moderate LDS bank conflicts ({cpw:.1f}/wave).",
                        "suggestion": "Investigate LDS access patterns.",
                    }
                )

    # --- Extended counters (may be 0 on gfx1100) ---

    valu = counters.get("SQ_INSTS_VALU", 0)
    salu = counters.get("SQ_INSTS_SALU", 0)
    smem = counters.get("SQ_INSTS_SMEM", 0)
    lds = counters.get("SQ_INSTS_LDS", 0)
    total_insts = valu + salu + smem + lds
    if total_insts > 0:
        metrics["instruction_mix"] = {
            "valu_pct": round(valu / total_insts * 100, 1),
            "salu_pct": round(salu / total_insts * 100, 1),
            "smem_pct": round(smem / total_insts * 100, 1),
            "lds_pct": round(lds / total_insts * 100, 1),
        }

    occupancy = counters.get("MeanOccupancyPerActiveCU", 0)
    if occupancy > 0:
        max_waves = gpu["max_waves_per_cu"]
        metrics["occupancy_measured"] = {
            "mean_waves_per_active_cu": round(occupancy, 1),
            "occupancy_pct": round(occupancy / max_waves * 100, 1),
        }

    mem_busy = counters.get("MemUnitBusy", 0)
    if mem_busy > 0:
        metrics["mem_unit_busy_pct"] = round(mem_busy, 1)
        if mem_busy > 80:
            findings.append(
                {
                    "severity": "high",
                    "category": "memory_bound",
                    "message": f"Memory unit busy {mem_busy:.0f}%.",
                    "suggestion": "Reduce memory traffic via tiling or smaller data types.",
                }
            )

    l2_hit = counters.get("GL2C_HIT", 0)
    l2_miss = counters.get("GL2C_MISS", 0)
    if l2_hit + l2_miss > 0:
        metrics["l2_cache"] = {
            "hit_rate_pct": round(l2_hit / (l2_hit + l2_miss) * 100, 1),
            "hits": round(l2_hit, 0),
            "misses": round(l2_miss, 0),
        }

    wave_cycles = counters.get("SQ_WAVE_CYCLES", 0)
    if busy_cycles > 0 and wave_cycles > 0:
        metrics["cu_activity"] = {
            "busy_cycles": round(busy_cycles, 0),
            "wave_cycles": round(wave_cycles, 0),
            "busy_pct": round(busy_cycles / wave_cycles * 100, 1),
        }

    severity_order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: severity_order.get(f["severity"], 3))
    return {"metrics": metrics, "findings": findings}


def analyze_dispatch_info(
    dispatches: list[dict], main_kernel: str | None, gpu: dict
) -> dict:
    """Analyze kernel dispatch parameters (VGPRs, LDS, grid size).

    Returns resource info + occupancy-limit findings.
    """
    kernel_dispatches = [d for d in dispatches if d["kernel_name"] == main_kernel]
    if not kernel_dispatches:
        return {}

    d = kernel_dispatches[0]
    vgpr = d["vgpr_count"]
    sgpr = d["sgpr_count"]
    lds = d["lds_size"]
    wg_size = d["workgroup_size"]
    grid = d["grid_size"]

    total_workgroups = grid[0] * grid[1] * grid[2]
    threads_per_wg = wg_size[0] * wg_size[1] * wg_size[2]
    waves_per_wg = (threads_per_wg + gpu["wave_size"] - 1) // gpu["wave_size"]

    waves_per_simd_vgpr = (1536 // vgpr) if vgpr > 0 else 16
    waves_per_cu_lds = (
        ((gpu["lds_per_cu_bytes"] // lds) * waves_per_wg)
        if lds > 0
        else gpu["max_waves_per_cu"]
    )

    info: dict = {
        "vgpr_count": vgpr,
        "sgpr_count": sgpr,
        "lds_bytes": lds,
        "workgroup_size": wg_size,
        "grid_size": grid,
        "total_workgroups": total_workgroups,
        "threads_per_workgroup": threads_per_wg,
        "waves_per_workgroup": waves_per_wg,
        "vgpr_limited_waves_per_simd": min(waves_per_simd_vgpr, 16),
        "lds_limited_waves_per_cu": min(waves_per_cu_lds, gpu["max_waves_per_cu"]),
    }

    findings: list[dict] = []
    if waves_per_simd_vgpr < 4:
        findings.append(
            {
                "severity": "high",
                "category": "vgpr_pressure",
                "message": f"High VGPR usage ({vgpr}) limits to {waves_per_simd_vgpr} waves/SIMD.",
                "suggestion": "Reduce register pressure or tile size.",
            }
        )
    if lds > 0 and waves_per_cu_lds < gpu["max_waves_per_cu"] // 2:
        findings.append(
            {
                "severity": "medium",
                "category": "lds_pressure",
                "message": f"LDS usage ({lds} bytes/WG) limits to {waves_per_cu_lds} waves/CU.",
                "suggestion": "Consider smaller shared memory tiles.",
            }
        )
    if total_workgroups < gpu["num_cus"]:
        findings.append(
            {
                "severity": "high",
                "category": "grid_underutilization",
                "message": f"Only {total_workgroups} workgroups for {gpu['num_cus']} CUs.",
                "suggestion": "Not enough work to fill the GPU.",
            }
        )

    info["findings"] = findings
    return info


# ---------------------------------------------------------------------------
# Internal CSV parsers
# ---------------------------------------------------------------------------


def _parse_kernel_trace(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            duration_ns = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            rows.append(
                {
                    "kernel_name": row["Kernel_Name"],
                    "dispatch_id": int(row["Dispatch_Id"]),
                    "duration_ns": duration_ns,
                    "grid_size": [
                        int(row["Grid_Size_X"]),
                        int(row["Grid_Size_Y"]),
                        int(row["Grid_Size_Z"]),
                    ],
                    "workgroup_size": [
                        int(row["Workgroup_Size_X"]),
                        int(row["Workgroup_Size_Y"]),
                        int(row["Workgroup_Size_Z"]),
                    ],
                    "vgpr_count": int(row["VGPR_Count"]),
                    "sgpr_count": int(row["SGPR_Count"]),
                    "lds_size": int(row["LDS_Block_Size"]),
                    "scratch_size": int(row["Scratch_Size"]),
                }
            )
    return rows


def _parse_kernel_stats(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "kernel_name": row["Name"],
                    "calls": int(row["Calls"]),
                    "total_duration_ns": int(row["TotalDurationNs"]),
                    "average_ns": float(row["AverageNs"]),
                    "percentage": float(row["Percentage"]),
                    "min_ns": int(row["MinNs"]),
                    "max_ns": int(row["MaxNs"]),
                    "stddev": float(row["StdDev"]),
                }
            )
    return rows


def _parse_counter_csv(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "Dispatch_Id": row["Dispatch_Id"],
                    "Kernel_Name": row["Kernel_Name"],
                    "Counter_Name": row["Counter_Name"],
                    "Counter_Value": float(row["Counter_Value"]),
                }
            )
    return rows


def _merge_counters_by_kernel(all_counters: dict, kernel_names: dict) -> dict:
    by_kernel: dict[str, list[dict]] = {}
    for did, counters in all_counters.items():
        name = kernel_names[did]
        by_kernel.setdefault(name, []).append(counters)

    result = {}
    for name, dispatch_list in by_kernel.items():
        all_counter_names: set[str] = set()
        for d in dispatch_list:
            all_counter_names.update(d.keys())

        averaged = {}
        for cn in sorted(all_counter_names):
            values = [d[cn] for d in dispatch_list if cn in d]
            if values:
                averaged[cn] = sum(values) / len(values)

        result[name] = {"num_dispatches": len(dispatch_list), "counters": averaged}
    return result
