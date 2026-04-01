______________________________________________________________________

## name: WMMA 128x128 kernel v1 description: First native HIP GEMM kernel — 128x128 block tile, 4 warps, WMMA 16x16x16, single-buffered shared memory type: project

## Kernel: gemm_wmma_128x128 (v1)

**Location:** `gemm/kernel_providers/native_hip/src/native_hip_gemm.hip`

### Tiling Strategy

- Block tile: 128x128, K tile: 16
- 4 warps (128 threads), arranged 2x2
- Each warp computes 64x64 output = 4x4 WMMA tiles
- A in shared memory: row-major [128][16]
- B in shared memory: transposed as [128][16] (B^T)

### Performance (2026-04-01)

| Shape | Native HIP | hipblaslt | % of Peak |
|-------|-----------|-----------|-----------|
| 4096x4096x4096 | 66.8 TFLOPS | 78.4 TFLOPS | 54.3% |
| 3840x3840x2304 | 76.5 TFLOPS | 78.7 TFLOPS | 62.2% |

### Known Limitations

1. No shared memory padding → 4-way bank conflicts on fragment loads
1. Scattered C stores (non-coalesced) — each lane writes 8 strided elements
1. Single-buffered shared memory — no overlap of global loads and compute
1. B load: 16 separate scalar loads per thread (strided global reads, though each is coalesced across threads)
1. Estimated ~212 VGPRs/wave → only 1 block (4 waves) per CU

### Next Optimizations to Try (priority order)

1. Shared memory padding (stride 20 instead of 16) to eliminate bank conflicts
1. Double buffering to hide global memory latency
1. Coalesced C stores via shared memory reorder
1. Vectorized B loads (load 8 bf16 at once, transpose in registers)
