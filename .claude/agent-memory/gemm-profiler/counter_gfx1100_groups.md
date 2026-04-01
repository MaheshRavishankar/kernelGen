______________________________________________________________________

## name: gfx1100 PMC counter compatibility groups description: Validated counter groups that can be collected together in a single pass on gfx1100. FETCH_SIZE and WRITE_SIZE cannot share a pass. type: counter

## gfx1100 Counter Compatibility Groups

Validated 2026-03-31 via `rocprofv3-avail pmc-check` on W7900 (gfx1100).

### Compatible groups (single pass each)

- `GPUBusy SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_SMEM`
- `VALUInsts SALUInsts SFetchInsts LDSBankConflict ALUStalledByLDS`
- `FETCH_SIZE MemUnitBusy` (FETCH + MemUnitBusy OK together)
- `WRITE_SIZE` (must be alone or with compatible counters)
- `GL2C_HIT GL2C_MISS GL2C_MC_RDREQ GL2C_MC_WRREQ`
- `SQ_BUSY_CYCLES SQ_WAVE_CYCLES SQ_WAIT_ANY`
- `MeanOccupancyPerActiveCU`
- `SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_WAIT_INST_LDS`

### Known incompatible

- `FETCH_SIZE` + `WRITE_SIZE` — **cannot** be in same pass on gfx1100
- `WRITE_SIZE` + `MemUnitBusy` — untested but likely incompatible
- Requesting too many counters in a single `--pmc` group causes error code 38:
  "Request exceeds the capabilities of the hardware to collect"

### Validation command

```bash
~/kernelGen/TheRock/bin/rocprofv3-avail pmc-check COUNTER1 COUNTER2 ...
```
