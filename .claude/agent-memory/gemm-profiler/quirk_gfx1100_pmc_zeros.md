______________________________________________________________________

## name: gfx1100 per-dispatch PMC counters mostly return 0 description: Most rocprofv3 PMC counters return 0 in per-dispatch collection mode on gfx1100 (RDNA3). Only SQ_WAVES, SQ_BUSY_CYCLES, and LDSBankConflict report non-zero values. type: quirk

## gfx1100 Per-Dispatch PMC Counter Limitation

Discovered 2026-03-31 while testing rocprofv3 1.2.1 from TheRock on W7900 (gfx1100).

### Working counters (per-dispatch)

- `SQ_WAVES` — total waves dispatched (matches grid_size expectations)
- `SQ_BUSY_CYCLES` — cycles with at least one active wave (always non-zero for real kernels)
- `LDSBankConflict` — reports non-zero when LDS access patterns have conflicts

### Counters that return 0 (per-dispatch on gfx1100)

- All `SQ_INSTS_*` (VALU, SALU, LDS, SMEM) — instruction counts
- All `GL2C_*` (HIT, MISS, MC_RDREQ, MC_WRREQ) and `_sum` variants — L2 cache
- `GPUBusy`, `MemUnitBusy` — utilization percentages
- `FETCH_SIZE`, `WRITE_SIZE` — memory bandwidth
- `VALUInsts`, `SALUInsts`, `SFetchInsts` — derived instruction rates
- `ALUStalledByLDS` — ALU stall metric
- `SQ_WAVE_CYCLES`, `SQ_WAIT_ANY` — wave cycle counters
- `MeanOccupancyPerActiveCU` — occupancy metric

### Implication

On gfx1100, bottleneck analysis must rely primarily on:

1. Dispatch metadata (VGPRs, LDS size, grid dimensions) for occupancy estimation
1. Kernel timing + problem size for roofline analysis
1. The 3 working counters for wave-level and LDS insights

### Possible workarounds (untested)

- `rocprof-compute` (omniperf) may use different collection methodology
- Application-wide (non per-dispatch) collection might report more counters
- CDNA GPUs (gfx90a, gfx942) may have better per-dispatch counter support
