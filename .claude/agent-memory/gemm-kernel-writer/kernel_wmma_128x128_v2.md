______________________________________________________________________

## name: WMMA 128x128 kernel v2 (8-warp) description: Optimized 8-warp GEMM kernel with 32x64 warp tiles, matching hipblaslt at ~80 TFLOPS type: project

## Kernel: gemm_wmma_128x128 (v2 — 8-warp)

**Location:** `gemm/kernel_providers/native_hip/src/native_hip_gemm.hip`

### Tiling Strategy

- Block tile: 128x128, K tile: 32 (BK=32, 2x WMMA_K)
- 8 warps (256 threads), arranged 4x2
- Each warp computes 32x64 output = 2x4 WMMA tiles
- Single-buffered LDS (20480 bytes), PAD=8

### Performance (2026-04-01)

| Shape | Native HIP (8-warp) | hipblaslt |
|-------|--------------------|-----------|
| 4096x4096x4096 | 78-83 TFLOPS | 79-83 TFLOPS |
| 3840x3840x2304 | 80.9 TFLOPS | 79.4 TFLOPS |
| 11520x3840x3840 | 79.7 TFLOPS | 79.3 TFLOPS |

### Resource Usage: 120 VGPRs, 0 scratch, 20480B LDS, 12 waves/SIMD

### Optimizations That FAILED (don't retry on gfx1100)

1. **Register prefetch:** amdclang spills to scratch. Only viable if \<8 VGPRs prefetch.
1. **Double-buffered LDS:** 40KB -> 1 WG/CU. Also inflates VGPRs 120->224.

### gfx1100 Insights

- WMMA: 1 per 16 cycles/SIMD. sclk sustains ~1.6 GHz (not 2.5 GHz boost).
- ~80 TFLOPS = ~97-100% of effective hardware throughput.
- bf16x16 fragment = 2x ds_read_b128. Keep WG LDS < 22KB for 3 WGs/CU.
