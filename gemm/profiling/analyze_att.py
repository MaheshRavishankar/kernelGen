#!/usr/bin/env python3
"""Analyze ROCprof Compute Viewer ATT ui_output JSON.

rocprofv3 --att emits directories named ui_output_agent_*_dispatch_* that are
normally consumed by ROCprof Compute Viewer. The JSON files in those directories
also contain enough structured data to drive command-line bottleneck analysis:

  * code.json: ISA/source rows with Hit, Latency, Stall, and Idle columns
  * se*_sm*_sl*_wv*.json: per-wave timelines and waitcnt dependencies
  * occupancy.json: dispatch and occupancy samples

This script intentionally uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


WAITCNT_RE = re.compile(r"\bs_waitcnt\s+([^ ]+)")


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def short_text(text: str, limit: int = 140) -> str:
    text = " ".join(text.strip().split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def op_class(isa: str) -> str:
    text = isa.strip()
    if not text:
        return "empty"
    if text.startswith(";"):
        return "comment"
    op = text.split()[0]
    if op.startswith("s_waitcnt"):
        match = WAITCNT_RE.search(text)
        return f"waitcnt:{match.group(1)}" if match else "waitcnt"
    if op.startswith("s_barrier"):
        return "barrier"
    if "wmma" in op:
        return "wmma"
    if "mfma" in op:
        return "mfma"
    if op.startswith("ds_"):
        return "lds"
    if op.startswith(("global_", "buffer_", "flat_")):
        return "vmem"
    if op.startswith(("s_load", "s_buffer")):
        return "smem"
    if op.startswith("v_"):
        return "valu"
    if op.startswith("s_"):
        return "salu"
    return op


def header_index(header: list[str], name: str, fallback: int) -> int:
    try:
        return header.index(name)
    except ValueError:
        return fallback


def row_value(row: list[Any], index: int) -> float:
    return number(row[index]) if index < len(row) else 0.0


def code_rows(ui_dir: Path) -> tuple[list[str], list[list[Any]]]:
    data = load_json(ui_dir / "code.json")
    return [x.strip() for x in data["header"].split(",")], data["code"]


def row_by_line(rows: list[list[Any]], line_index: int) -> dict[int, list[Any]]:
    result = {}
    for row in rows:
        line = int(row_value(row, line_index))
        result[line] = row
    return result


def wave_files(ui_dir: Path) -> list[Path]:
    return sorted(ui_dir.glob("se*_sm*_sl*_wv*.json"))


def occupancy_summary(ui_dir: Path) -> dict[str, Any]:
    path = ui_dir / "occupancy.json"
    if not path.exists():
        return {}
    data = load_json(path)
    sample_rows = []
    for key, value in data.items():
        if key.isdigit() and isinstance(value, list):
            sample_rows.extend(value)
    times = [number(row[0]) for row in sample_rows if row]
    return {
        "dispatches": data.get("dispatches", {}),
        "sample_count": len(sample_rows),
        "time_min": min(times) if times else 0,
        "time_max": max(times) if times else 0,
    }


def wave_summary(
    ui_dir: Path,
    rows: list[list[Any]],
    line_index: int,
    stall_index: int,
    latency_index: int,
) -> dict[str, Any]:
    line_map = row_by_line(rows, line_index)
    paths = wave_files(ui_dir)
    durations = []
    timeline_state_cycles: Counter[int] = Counter()
    waitcnt_count_by_kind: Counter[str] = Counter()
    waitcnt_source_count: Counter[int] = Counter()
    waitcnt_source_latency: Counter[int] = Counter()
    waitcnt_source_stall: Counter[int] = Counter()

    for path in paths:
        data = load_json(path)
        wave = data.get("wave", {})
        if "duration" in data:
            durations.append(number(data["duration"]))
        elif "begin" in wave and "end" in wave:
            durations.append(number(wave["end"]) - number(wave["begin"]))

        for state, cycles in wave.get("timeline", []):
            timeline_state_cycles[int(state)] += number(cycles)

        for wait_line, sources in wave.get("waitcnt", []):
            wait_row = line_map.get(int(wait_line))
            wait_kind = "waitcnt"
            if wait_row:
                wait_kind = op_class(str(wait_row[0]))
            waitcnt_count_by_kind[wait_kind] += 1

            for source in sources:
                if not source:
                    continue
                source_line = int(source[0])
                waitcnt_source_count[source_line] += 1
                source_row = line_map.get(source_line)
                if source_row:
                    waitcnt_source_latency[source_line] += row_value(
                        source_row, latency_index
                    )
                    waitcnt_source_stall[source_line] += row_value(
                        source_row, stall_index
                    )

    return {
        "num_wave_files": len(paths),
        "duration_min": min(durations) if durations else 0,
        "duration_avg": mean(durations) if durations else 0,
        "duration_max": max(durations) if durations else 0,
        "timeline_state_cycles": dict(timeline_state_cycles),
        "waitcnt_count_by_kind": dict(waitcnt_count_by_kind),
        "waitcnt_source_count": dict(waitcnt_source_count),
        "waitcnt_source_latency": dict(waitcnt_source_latency),
        "waitcnt_source_stall": dict(waitcnt_source_stall),
    }


def top_rows(
    rows: list[list[Any]],
    line_index: int,
    hit_index: int,
    latency_index: int,
    stall_index: int,
    idle_index: int,
    sort_index: int,
    top: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (row_value(row, sort_index), row_value(row, latency_index)),
        reverse=True,
    )
    result = []
    for row in ranked[:top]:
        result.append(
            {
                "line": int(row_value(row, line_index)),
                "isa": short_text(str(row[0])),
                "op_class": op_class(str(row[0])),
                "hit": row_value(row, hit_index),
                "latency": row_value(row, latency_index),
                "stall": row_value(row, stall_index),
                "idle": row_value(row, idle_index),
            }
        )
    return result


def analyze(ui_dir: Path, top: int) -> dict[str, Any]:
    header, rows = code_rows(ui_dir)
    line_i = header_index(header, "LineNumber", 2)
    hit_i = header_index(header, "Hit", 6)
    latency_i = header_index(header, "Latency", 7)
    stall_i = header_index(header, "Stall", 8)
    idle_i = header_index(header, "Idle", 9)
    line_map = row_by_line(rows, line_i)

    totals: dict[str, float] = {"hit": 0, "latency": 0, "stall": 0, "idle": 0}
    op_totals: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        cls = op_class(str(row[0]))
        hit = row_value(row, hit_i)
        latency = row_value(row, latency_i)
        stall = row_value(row, stall_i)
        idle = row_value(row, idle_i)
        totals["hit"] += hit
        totals["latency"] += latency
        totals["stall"] += stall
        totals["idle"] += idle
        op_totals[cls]["hit"] += hit
        op_totals[cls]["latency"] += latency
        op_totals[cls]["stall"] += stall
        op_totals[cls]["idle"] += idle

    waves = wave_summary(ui_dir, rows, line_i, stall_i, latency_i)
    source_rows = []
    for line_text, count in Counter(waves["waitcnt_source_count"]).most_common(top):
        line = int(line_text)
        row = line_map.get(line)
        if not row:
            continue
        source_rows.append(
            {
                "line": line,
                "isa": short_text(str(row[0])),
                "op_class": op_class(str(row[0])),
                "waitcnt_refs": count,
                "latency": row_value(row, latency_i),
                "stall": row_value(row, stall_i),
            }
        )

    kernel = ""
    for row in rows:
        if len(row) > 3 and row[3]:
            kernel = str(row[3])
            break
        if str(row[0]).startswith(";") and not kernel:
            kernel = str(row[0]).lstrip("; ")

    return {
        "ui_dir": str(ui_dir),
        "kernel": kernel,
        "header": header,
        "num_code_rows": len(rows),
        "totals": totals,
        "op_totals_by_stall": {
            key: dict(value)
            for key, value in sorted(
                op_totals.items(), key=lambda item: item[1]["stall"], reverse=True
            )
        },
        "top_stall_instructions": top_rows(
            rows, line_i, hit_i, latency_i, stall_i, idle_i, stall_i, top
        ),
        "top_latency_instructions": top_rows(
            rows, line_i, hit_i, latency_i, stall_i, idle_i, latency_i, top
        ),
        "top_idle_instructions": top_rows(
            rows, line_i, hit_i, latency_i, stall_i, idle_i, idle_i, top
        ),
        "top_waitcnt_source_instructions": source_rows,
        "wave": {
            "num_wave_files": waves["num_wave_files"],
            "duration_min": waves["duration_min"],
            "duration_avg": waves["duration_avg"],
            "duration_max": waves["duration_max"],
            "timeline_state_cycles": waves["timeline_state_cycles"],
            "waitcnt_count_by_kind": waves["waitcnt_count_by_kind"],
        },
        "occupancy": occupancy_summary(ui_dir),
    }


def print_text(summary: dict[str, Any], top: int) -> None:
    print(f"ATT UI directory: {summary['ui_dir']}")
    print(f"Kernel: {summary['kernel']}")
    totals = summary["totals"]
    print(
        "Totals: "
        f"hit={totals['hit']:.0f} latency={totals['latency']:.0f} "
        f"stall={totals['stall']:.0f} idle={totals['idle']:.0f}"
    )
    wave = summary["wave"]
    print(
        "Waves: "
        f"{wave['num_wave_files']} files, duration min/avg/max="
        f"{wave['duration_min']:.0f}/{wave['duration_avg']:.0f}/"
        f"{wave['duration_max']:.0f}"
    )
    print(f"Waitcnt counts: {wave['waitcnt_count_by_kind']}")
    print(f"Timeline state cycles: {wave['timeline_state_cycles']}")

    occ = summary["occupancy"]
    if occ:
        print(
            "Occupancy samples: "
            f"{occ['sample_count']} from {occ['time_min']:.0f} to {occ['time_max']:.0f}"
        )

    print("\nTop op classes by stall:")
    for cls, vals in list(summary["op_totals_by_stall"].items())[:top]:
        print(
            f"  {cls:20s} stall={vals.get('stall', 0):.0f} "
            f"latency={vals.get('latency', 0):.0f} "
            f"hit={vals.get('hit', 0):.0f} idle={vals.get('idle', 0):.0f}"
        )

    print("\nTop stall instructions:")
    for row in summary["top_stall_instructions"][:top]:
        print(
            f"  line={row['line']} class={row['op_class']:16s} "
            f"hit={row['hit']:.0f} lat={row['latency']:.0f} "
            f"stall={row['stall']:.0f} idle={row['idle']:.0f} :: {row['isa']}"
        )

    print("\nTop waitcnt source instructions:")
    for row in summary["top_waitcnt_source_instructions"][:top]:
        print(
            f"  refs={row['waitcnt_refs']} line={row['line']} "
            f"class={row['op_class']:12s} lat={row['latency']:.0f} "
            f"stall={row['stall']:.0f} :: {row['isa']}"
        )

    print("\nInterpretation hints:")
    waits = summary["wave"]["waitcnt_count_by_kind"]
    if any("vmcnt" in key for key in waits):
        print(
            "  - vmcnt waitcnt stalls point to VMEM/global-load latency or coalescing."
        )
    if any("lgkmcnt" in key for key in waits):
        print("  - lgkmcnt waitcnt stalls point to LDS/scalar-memory dependencies.")
    if "barrier" in summary["op_totals_by_stall"]:
        print("  - barrier latency points to synchronization or workgroup imbalance.")
    if summary["totals"]["idle"] > summary["totals"]["stall"]:
        print("  - idle dominates recorded stall; check occupancy and wave scheduling.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze ROCprof Compute Viewer ATT ui_output JSON"
    )
    parser.add_argument("ui_dir", type=Path)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--json", action="store_true", help="emit JSON summary")
    args = parser.parse_args()

    ui_dir = args.ui_dir.resolve()
    if not (ui_dir / "code.json").exists():
        raise SystemExit(f"ERROR: {ui_dir} does not contain code.json")

    summary = analyze(ui_dir, args.top)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_text(summary, args.top)


if __name__ == "__main__":
    main()
