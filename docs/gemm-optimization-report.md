# GEMM Kernel Optimization Report (bd-3p3)

## Summary

Optimized the native HIP WMMA GEMM kernel on gfx1100 (W7900) from 78.3 TFLOPS to
79.5 TFLOPS (+1.5%), achieving parity with hipblaslt (~79 TFLOPS). Both kernels
operate at 64-67% of AMD's rated 123 TFLOPS BF16 peak. The gap is likely due to a
combination of factors — memory system bottlenecks, occupancy limits, and pipeline
efficiency — rather than any single cause.

## Baseline (4-warp kernel, pre-optimization)

| Shape | Native HIP | hipblaslt | % of Peak |
|-------|-----------|-----------|-----------|
| 4096x4096x4096 | 78.3 TFLOPS | 78.9 TFLOPS | 63.7% |
| 3840x3840x2304 | 80.2 TFLOPS | 79.4 TFLOPS | 65.2% |
| 11520x3840x3840 | 79.4 TFLOPS | 79.3 TFLOPS | 64.6% |

Baseline kernel: 128x128 block tile, 4 warps (128 threads), 64x64 warp tile,
224 VGPRs, 6 waves/SIMD, 0 scratch.

## Optimization Iterations

### Iteration 1: 8-warp kernel with smaller warp tiles (KEPT)

**Change:** Restructured from 4 warps (128 threads) to 8 warps (256 threads) with
4x2 warp arrangement and 32x64 warp tiles.

**Rationale:** The profiler showed 224 VGPRs limiting occupancy to 6 waves/SIMD.
Halving the accumulators per warp (8 tiles vs 16) would reduce VGPRs and improve
occupancy for better latency hiding.

**Results:**

- VGPRs: 224 -> 120 (46% reduction)
- Scratch: 0 (no spilling)
- Occupancy: 6 -> 12 waves/SIMD (VGPR limit)
- LDS: unchanged at 20480 bytes (3 WGs/CU)
- Per-thread global load work halved (256 threads share the load)
- Performance: 78.3 -> 79.5 TFLOPS (+1.5%)

**Why only +1.5%:** The WMMA pipeline throughput (1 per 16 cycles per SIMD) is the
fundamental bottleneck, not memory latency hiding. Higher occupancy helps marginally
but doesn't increase WMMA throughput.

### Iteration 2a: Software pipelining with register prefetch (REVERTED)

**Change:** Prefetch next K-tile into registers while computing current tile from
shared memory. Added 16 VGPRs for A prefetch buffers + 16 for B = 32 total.

**Results on 4-warp kernel:** 78.3 -> 42.3 TFLOPS (-46%). VGPR: 256, scratch=80.
The compiler spilled 80 bytes/thread to scratch memory, destroying performance.

**Results on 8-warp kernel:** 79.5 -> 62.3 TFLOPS (-22%). VGPR: 128, scratch=48.
Even with lower base VGPRs, the long live ranges of prefetch buffers across the
compute section caused spilling.

**Why it failed:** The AMD compiler (amdclang 23.0) is very aggressive about
reducing VGPR count, even at the cost of spilling. Holding 16-32 VGPRs of prefetch
data across 16+ WMMA instructions and 24+ LDS reads creates long live ranges that
the register allocator can't accommodate without spills.

### Iteration 2b: Double-buffered shared memory (REVERTED)

**Change:** Two LDS buffers (ping-pong), loads target inactive buffer while
computing from active buffer. Only 1 \_\_syncthreads per K-step instead of 2.

**Results:** 79.5 -> 74.7 TFLOPS (-6%). VGPR: 224 (up from 120), LDS: 40960 bytes.

**Why it failed:** Two problems compounded:

1. LDS doubled to 40KB, reducing from 3 WGs/CU to 1 WG/CU (occupancy dropped from
   12 to 4 waves/SIMD)
1. Runtime buffer index (`smem_A[buf]`) introduced more complex address calculations,
   inflating VGPRs from 120 to 224

The occupancy loss outweighed the benefit of overlapping loads with compute and
halving sync count.

### Iteration 3: ISA analysis and clock investigation

**ISA inspection** of the final 8-warp kernel showed clean code:

- 16 v_wmma instructions per K-step (fully pipelined)
- 24 ds_read instructions (2 per bf16x16 fragment, 12 fragments)
- 18 global_load instructions (2 uint4 for A, 16 scalar for B)
- 2 s_barrier instructions
- 0 scratch instructions
- 22 s_waitcnt instructions

**Clock analysis:** The W7900's sclk shows idle at ~0-10 MHz and supported levels
up to 1760 MHz (max boost). Note that AMD's 123 TFLOPS BF16 rating implies an
effective clock of ~2.5 GHz (back-calculated from 48 CUs × ops_per_cycle), which
is higher than the actual hardware clock spec. With extended warmup (50 runs), both
our kernel and hipblaslt achieve ~83 TFLOPS. The ~65% efficiency relative to the
rated 123 TFLOPS peak likely reflects a combination of the gap between marketing
TFLOPS and achievable hardware clocks, memory latency, occupancy limits, and
pipeline stalls rather than any single factor.

## Final Performance

| Shape | 8-warp (new) | 4-warp (old) | hipblaslt | vs old | vs hipblaslt |
|-------|-------------|-------------|-----------|--------|--------------|
| 4096x4096x4096 | 78.0-82.6 | 78.0-80.7 | 78.9-83.0 | ~+1.5% | ~99.6% |
| 3840x3840x2304 | 80.9 | 80.2 | 79.4 | +0.9% | 101.9% |
| 11520x3840x3840 | 79.7 | 79.4 | 79.3 | +0.4% | 100.5% |

Performance varies by ~3% across runs due to GPU clock/thermal dynamics.

## What Worked and Why

1. **8-warp restructuring (+1.5%):** Halving accumulators per warp reduced VGPRs by
   46%, doubled occupancy. Main benefit is better latency hiding for global loads and
   LDS reads, plus halved per-thread global load work.

1. **Shared memory padding (pre-existing):** SMEM_PAD=8 gives essentially zero LDS
   bank conflicts (profiler shows \<0.1 conflicts/wave). This was already in the
   baseline and is critical for LDS throughput.

## What Didn't Work and Why

1. **Software pipelining (register prefetch):** The amdclang compiler on gfx1100
   aggressively reduces VGPR count. Long-lived prefetch registers spanning the
   compute section get spilled to scratch memory, negating any overlap benefit.
   **Lesson:** On RDNA3, register prefetch is only viable if the base VGPR count is
   very low and the prefetch buffer is tiny (\<8 VGPRs).

1. **Double-buffered LDS:** Doubling shared memory from 20KB to 40KB drops from 3
   WGs/CU to 1 WG/CU. Additionally, the runtime buffer index inflates VGPR usage.
   On RDNA3 with 64KB LDS per CU, double buffering is only viable for kernels using
   \<16KB LDS per buffer. **Lesson:** On gfx1100, LDS occupancy is a hard constraint;
   prefer single-buffer with maximum WG concurrency.

1. **launch_bounds tuning:** `__launch_bounds__(256, 3)` had no effect on compiler
   output — the compiler already chose 120 VGPRs without any hint.

## Remaining Bottlenecks

1. **WMMA throughput is the hard limit.** At 1 WMMA per 16 cycles per SIMD, the
   theoretical minimum for this workload matches the observed performance closely.
   No amount of kernel optimization can exceed this limit.

1. **Efficiency gap vs rated peak.** Both our kernel and hipblaslt achieve only
   64-67% of AMD's rated 123 TFLOPS BF16 peak. The rated peak implies an effective
   clock of ~2.5 GHz (back-calculated), which exceeds the W7900's actual max boost
   of 1760 MHz. The remaining gap is likely a combination of memory system latency,
   occupancy constraints, and pipeline efficiency.

1. **Global load serialization.** The two \_\_syncthreads per K-step create a bubble
   where all threads must complete their loads before any thread can compute. With
   single-buffer LDS, this is unavoidable without sacrificing occupancy.

## Ideas for Future Work

1. **Asynchronous copy to LDS** — Future ROCm versions may support async LDS fill
   (similar to CUDA's cp.async), which would enable pipelining without extra VGPRs.

1. **Assembly-level optimization** — Hand-tuned GCN assembly could interleave global
   loads with WMMA instructions more precisely than the compiler, potentially
   eliminating the need for separate prefetch buffers.

1. **Kernel fusion** — For real workloads, fusing GEMM with subsequent operations
   (bias add, activation, etc.) would improve overall throughput by eliminating
   memory round-trips.

1. **Non-square tile exploration** — Asymmetric block tiles (e.g., 256x64) might
   better match specific problem shapes while maintaining similar LDS/VGPR budgets.

1. **Clock frequency locking** — Running with forced high-performance clock (via
   `rocm-smi --setperflevel high`) and monitoring actual clocks under load (via
   `rocm-smi -c`) would help quantify how much of the efficiency gap is due to
   clock behavior vs other factors.
