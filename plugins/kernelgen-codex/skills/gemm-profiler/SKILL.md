______________________________________________________________________

## name: gemm-profiler description: Profile and analyze GEMM kernel performance with rocprofv3 and roofline-style bottleneck analysis.

# GEMM Profiler

Use this skill when the task is to profile a GEMM provider, interpret rocprof
results, or explain why a kernel is underperforming.

Always read [AGENTS.md](../../../../AGENTS.md) and the relevant profile inputs
before drawing conclusions.

Use the repo venv for Python entrypoints, especially IREE and Fusilli tooling.
For sandboxed Codex runs, instruct the agent to send a short final completion
message before exit so the wrapper does not linger after the work is done.

## Scope

- Profile collection: `gemm/profiling/profile.py`
- Analysis: `gemm/profiling/analyze.py`
- Generic rocprof utilities: `profiling/rocprof.py`

## Workflow

1. Collect a profile
   - `.venv/bin/python gemm/profiling/profile.py --provider <provider> --test <test_dir> -o /tmp/profile.json`
   - Use `--skip-pmc` for quick trace-only checks
1. Analyze it
   - `.venv/bin/python gemm/profiling/analyze.py /tmp/profile.json --arch gfx1100`
1. Clean temporary rocprof artifacts
   - `rm -rf .rocprofv3/ *.co`
1. For deep dives, use `rocprofv3` directly when needed

## Bottleneck Heuristics

1. Low compute efficiency: inspect instruction mix and occupancy
1. `MemUnitBusy > 80%`: likely memory-bound
1. Low occupancy: inspect VGPR and LDS usage
1. High `LDSBankConflict`: shared memory access issue
1. High `ALUStalledByLDS`: pipeline or LDS pressure issue
1. High `SQ_WAIT_ANY`: latent memory or insufficient parallelism

## gfx1100 Notes

- Peak BF16 TFLOPS: about `123`
- Peak memory bandwidth: about `864 GB/s`
- CUs: `48`
- Wave size: `32`

Important PMC constraint:

- `FETCH_SIZE` and `WRITE_SIZE` cannot be collected in the same pass on
  `gfx1100`

## Hard Rules

- Lead with measured results, not speculation.
- Distinguish trace evidence from inferred bottlenecks.
- Clean rocprof temporary files after profiling runs.
- If comparing providers, use the same test and comparable run conditions.

## Output

Present results as:

1. Summary
1. Key metrics
1. Ranked bottlenecks
1. Actionable recommendations
