# GEMM Profiling Report — 2026-03-31

## Environment

| Component | Version / Details |
|-----------|-------------------|
| GPU | AMD Radeon PRO W7900 (gfx1100, RDNA3) |
| Peak BF16 TFLOPS | 123.0 |
| Peak memory BW | 864 GB/s |
| CUs | 48 |
| TheRock (ROCm) | 7.13.0 |
| hipblaslt | 1.2.2 (from TheRock 7.13.0) |
| IREE runtime | Built from source, `~/kernelGen/iree/iree/` (package-version 3.12.0.dev) |
| iree-base-compiler | 3.12.0rc20260330 (pip, used for MLIR→vmfb compilation) |
| Profiler | rocprofv3 (rocprofiler-sdk 1.2.1, from TheRock) |
| Benchmark config | 5 warmup, 20 timed iterations per test |

## Test suite

All tests use `transA=false, transB=false, alpha=1.0, beta=0.0`.

| Test | M×N×K | Dtypes | Compute | Description |
|------|-------|--------|---------|-------------|
| ai_high_large_k | 4096x1024x150000 | bf16/bf16/bf16 | f32 | High AI, very large K (AI=814.8) |
| ai_high_medium | 1285x2048x3840 | bf16/bf16/bf16 | f32 | High AI, medium (AI=654.9) |
| ai_high_small | 576x576x1280 | bf16/bf16/bf16 | f32 | High AI, small (AI=235.1) |
| ai_low_large_flat | 21760x3840x20 | bf16/bf16/bf16 | f32 | Low AI, large M\*N but tiny K (AI=19.9) |
| ai_low_skinny | 16x512x1024 | bf16/bf16/bf16 | f32 | Low AI, skinny M (AI=15.3) |
| ai_low_small | 32x576x2304 | bf16/bf16/bf16 | f32 | Low AI, small (AI=29.9) |
| ai_medium_extreme | 16800000x128x134 | bf16/bf16/bf16 | f32 | Medium AI, extreme M (AI=65.5) |
| ai_medium_large | 7680x512x304 | bf16/bf16/bf16 | f32 | Medium AI, larger (AI=186.1) |
| ai_medium_small | 576x576x165 | bf16/bf16/bf16 | f32 | Medium AI, small (AI=104.9) |
| ai_very_high_extreme | 150000x16384x4096 | bf16/bf16/bf16 | f32 | Very high AI, extreme (AI=3206.7) |
| ai_very_high_large | 11520x3840x3840 | bf16/bf16/bf16 | f32 | Very high AI, large (AI=1645.7) |
| ai_very_high_medium | 3840x3840x2304 | bf16/bf16/bf16 | f32 | Very high AI, medium (AI=1047.3) |
| ai_very_low_small_square | 576x576x10 | bf16/bf16/bf16 | f32 | Very low AI, small square (AI=9.7) |
| ai_very_low_small_wide | 576x2304x10 | bf16/bf16/bf16 | f32 | Very low AI, small wide (AI=9.8) |
| ai_very_low_tiny | 4x384x5 | bf16/bf16/bf16 | f32 | Very low AI, tiny shape (AI=2.2) |
| small_f16 | 1024x1024x1024 | f16/f16/f16 | f32 | Small square F16 smoke test |

Arithmetic intensity (AI) = 2*M*N\*K / (bytes_A + bytes_B + bytes_C). Ridge point for gfx1100 BF16 is 142.4 FLOP/byte.

## Results — hipblaslt

| Test | Time (us) | TFLOPS | % Peak | Bound | VGPRs | LDS (bytes) | Waves/SIMD (VGPR) | Waves/CU (LDS) | SQ_WAVES | SQ_BUSY_CYCLES | Cycles/wave | LDS conflicts/wave | Bottleneck |
|------|-----------|--------|--------|-------|-------|-------------|-------------------|----------------|----------|----------------|-------------|--------------------|-----------|
| ai_very_high_large | 4,356 | 78.0 | 63.4% | compute | 256 | 30,720 | 6 | 8 | 19,200 | 459,977,603 | 23,957 | 0.0 | LDS→8 waves/CU |
| ai_very_high_medium | 874 | 77.7 | 63.2% | compute | 256 | 30,720 | 6 | 8 | 6,400 | 92,630,618 | 14,474 | 0.0 | LDS→8 waves/CU |
| ai_high_medium | 297 | 68.0 | 55.3% | compute | 256 | 30,720 | 6 | 8 | 1,232 | 30,723,593 | 24,938 | 0.1 | LDS→8 waves/CU |
| ai_very_high_extreme | 298,808 | 67.4 | 54.8% | compute | 256 | 30,720 | 6 | 8 | 1,069,092 | 26,801,045,684 | 25,069 | 0.0 | LDS→8 waves/CU |
| ai_high_large_k | 19,320 | 65.1 | 53.0% | compute | 256 | 30,720 | 6 | 8 | 1,892 | 1,802,021,033 | 952,442 | 0.1 | LDS→8 waves/CU |
| small_f16 | 35 | 60.6 | 49.3% | compute | 256 | 28,672 | 6 | 8 | 704 | 4,366,085 | 6,202 | 0.1 | LDS→8 waves/CU |
| ai_medium_large | 49 | 49.2 | 40.0% | compute | 256 | 30,720 | 6 | 8 | 1,920 | 6,048,936 | 3,150 | 0.0 | LDS→8 waves/CU |
| ai_high_small | 23 | 37.0 | 30.1% | compute | 256 | 28,672 | 6 | 8 | 216 | 2,620,987 | 12,134 | 0.2 | LDS→8 waves/CU |
| ai_medium_extreme | 24,612 | 23.4 | 19.0% | memory | 168 | 16,384 | 9 | 16 | 3,150,000 | 2,333,795,766 | 741 | — | None |
| ai_medium_small | 8 | 13.9 | 11.3% | memory | 256 | 28,672 | 6 | 8 | 216 | 980,052 | 4,537 | 0.6 | LDS→8 waves/CU |
| ai_low_small | 12 | 7.2 | 5.8% | memory | 256 | 27,648 | 6 | 8 | 144 | 1,212,744 | 8,422 | 0.3 | LDS→8 waves/CU |
| ai_low_large_flat | 503 | 6.7 | 5.4% | memory | 168 | 16,384 | 9 | 16 | 108,800 | 60,237,259 | 554 | 0.0 | None |
| ai_very_low_small_wide | 10 | 2.7 | 2.2% | memory | 256 | 28,672 | 6 | 8 | 864 | 1,047,630 | 1,213 | 0.1 | LDS→8 waves/CU |
| ai_low_skinny | 7 | 2.3 | 1.9% | memory | 256 | 27,648 | 6 | 8 | 128 | 654,829 | 5,116 | — | LDS→8 waves/CU |
| ai_very_low_small_square | 5 | 1.3 | 1.1% | memory | 168 | 16,384 | 9 | 16 | 432 | 600,788 | 1,391 | 0.2 | None |
| ai_very_low_tiny | 5 | 0.0 | 0.0% | memory | 128 | 13,824 | 12 | 16 | 24 | 78,928 | 3,289 | 1.2 | None |

## Results — IREE

| Test | Time (us) | TFLOPS | % Peak | Bound | VGPRs | LDS (bytes) | Waves/SIMD (VGPR) | Waves/CU (LDS) | SQ_WAVES | SQ_BUSY_CYCLES | Cycles/wave | LDS conflicts/wave | Bottleneck |
|------|-----------|--------|--------|-------|-------|-------------|-------------------|----------------|----------|----------------|-------------|--------------------|-----------|
| ai_very_high_medium | 797 | 85.3 | 69.4% | compute | 240 | 17,920 | 6 | 12 | 3,600 | 93,587,858 | 25,997 | 0.1 | LDS→12 waves/CU |
| ai_very_high_large | 3,989 | 85.2 | 69.2% | compute | 240 | 17,920 | 6 | 12 | 10,800 | 457,859,733 | 42,394 | 0.0 | LDS→12 waves/CU |
| ai_medium_large | 34 | 71.3 | 58.0% | compute | 136 | 7,168 | 11 | 32 | 1,920 | 4,075,167 | 2,122 | 0.1 | None |
| ai_high_large_k | 18,522 | 67.9 | 55.2% | compute | 240 | 9,728 | 6 | 24 | 1,024 | 1,788,405,074 | 1,746,489 | 0.1 | None |
| ai_very_high_extreme | 298,533 | 67.4 | 54.8% | compute | 256 | 17,920 | 6 | 12 | 600,064 | 27,662,372,554 | 46,099 | 0.0 | LDS→12 waves/CU |
| ai_high_medium | 311 | 65.0 | 52.8% | compute | 248 | 17,920 | 6 | 12 | 704 | 35,196,646 | 49,995 | 0.1 | LDS→12 waves/CU |
| small_f16 | 43 | 50.1 | 40.7% | compute | 256 | 50,688 | 6 | **4** | 512 | 5,105,065 | 9,971 | 0.4 | LDS→**4 waves/CU** |
| ai_medium_extreme | 15,964 | 36.1 | 29.3% | memory | 152 | 7,168 | 10 | 32 | 1,050,000 | 1,375,088,822 | 1,310 | — | None |
| ai_high_small | 24 | 35.6 | 28.9% | compute | 256 | 50,688 | 6 | **4** | 180 | 2,939,667 | 16,331 | 0.2 | LDS→**4 waves/CU** |
| ai_medium_small | 8 | 14.1 | 11.4% | memory | 136 | 7,168 | 11 | 32 | 180 | 930,216 | 5,168 | 0.9 | None |
| ai_low_large_flat | 264 | 12.6 | 10.3% | memory | 144 | 13,312 | 10 | 16 | 40,800 | 37,021,322 | 907 | — | None |
| ai_very_low_small_wide | 4 | 7.4 | 6.0% | memory | 96 | 7,168 | 16 | 32 | 648 | 422,746 | 652 | 0.2 | None |
| ai_low_small | 19 | 4.5 | 3.6% | memory | 256 | 26,112 | 6 | 8 | 36 | 489,890 | 13,608 | — | LDS→8 waves/CU |
| ai_very_low_small_square | 3 | 2.4 | 1.9% | memory | 104 | 7,168 | 14 | 32 | 180 | 309,389 | 1,719 | 0.3 | None |
| ai_low_skinny | 9 | 1.9 | 1.5% | memory | 248 | 22,016 | 6 | 4 | 16 | 199,270 | 12,454 | — | LDS→4 waves/CU |
| ai_very_low_tiny | 2 | 0.0 | 0.0% | memory | 16 | 512 | 16 | 32 | 48 | 184,838 | 3,851 | 2.8 | LDS bank conflicts |

## Head-to-head comparison

Sorted by absolute TFLOPS difference (IREE - hipblaslt). Positive = IREE faster.

| Test | AI | Bound | hipblaslt TFLOPS | IREE TFLOPS | Delta | Winner |
|------|----|-------|-----------------|-------------|-------|--------|
| ai_medium_large | 186 | compute | 49.2 | **71.3** | +22.1 | IREE |
| ai_medium_extreme | 66 | memory | 23.4 | **36.1** | +12.7 | IREE |
| ai_very_high_medium | 1047 | compute | 77.7 | **85.3** | +7.6 | IREE |
| ai_very_high_large | 1646 | compute | 78.0 | **85.2** | +7.2 | IREE |
| ai_low_large_flat | 20 | memory | 6.7 | **12.6** | +5.9 | IREE |
| ai_very_low_small_wide | 10 | memory | 2.7 | **7.4** | +4.7 | IREE |
| ai_high_large_k | 815 | compute | 65.1 | **67.9** | +2.8 | IREE |
| ai_very_low_small_square | 10 | memory | 1.3 | **2.4** | +1.1 | IREE |
| ai_medium_small | 105 | memory | 13.9 | 14.1 | +0.2 | ~tie |
| ai_very_high_extreme | 3207 | compute | 67.4 | 67.4 | 0.0 | tie |
| ai_very_low_tiny | 2 | memory | 0.0 | 0.0 | 0.0 | tie |
| ai_low_skinny | 15 | memory | **2.3** | 1.9 | -0.4 | hipblaslt |
| ai_high_small | 235 | compute | **37.0** | 35.6 | -1.4 | hipblaslt |
| ai_low_small | 30 | memory | **7.2** | 4.5 | -2.7 | hipblaslt |
| ai_high_medium | 655 | compute | **68.0** | 65.0 | -3.0 | hipblaslt |
| small_f16 | 341 | compute | **60.6** | 50.1 | -10.5 | hipblaslt |

## Analysis

### Compute-bound tests (AI > 142 FLOP/byte)

Both providers are capped by VGPR pressure — all compute-bound kernels use 240-256 VGPRs, limiting occupancy to 6 waves/SIMD. The key differentiator is LDS usage:

- **IREE uses less LDS on large shapes.** For ai_very_high_large and ai_very_high_medium, IREE uses 17,920 bytes/WG (12 waves/CU) vs hipblaslt's 30,720 bytes/WG (8 waves/CU). This gives IREE ~50% more concurrent waves, which helps hide latency. Result: **85 vs 78 TFLOPS** (69% vs 63% peak).

- **IREE uses too much LDS on some small shapes.** For ai_high_small and small_f16, IREE allocates 50,688 bytes/WG — nearly the entire 64KB LDS per CU — limiting occupancy to only 4 waves/CU. hipblaslt uses 28KB for the same shapes (8 waves/CU). Result: hipblaslt wins small_f16 by 10 TFLOPS.

- **IREE's ai_medium_large tile choice is excellent.** Only 136 VGPRs and 7,168 bytes LDS → 11 waves/SIMD, 32 waves/CU. Much better occupancy than hipblaslt's 256 VGPRs / 30KB LDS. Result: **71 vs 49 TFLOPS**.

Best compute-bound result: **IREE at 85.3 TFLOPS (69.4% peak)** on ai_very_high_medium.

### Memory-bound tests (AI < 142 FLOP/byte)

- **IREE wins most memory-bound tests**, often by 2x (ai_low_large_flat: 12.6 vs 6.7, ai_very_low_small_wide: 7.4 vs 2.7). IREE tends to choose tiles with lower VGPR counts (96-152 vs 168-256), giving much higher occupancy which helps hide memory latency.

- **hipblaslt wins ai_low_small and ai_low_skinny.** These are very small problem sizes where IREE's compilation overhead or tile choice doesn't pay off.

### Universal bottleneck: VGPR pressure

Neither provider exceeds 70% peak. The ceiling is 6 waves/SIMD from 240-256 VGPRs. Both providers would benefit from kernels that use fewer VGPRs, at the cost of more register spilling or smaller tile sizes.

### LDS bank conflicts

LDS bank conflicts are negligible for both providers (< 1 per wave on all tests except IREE's ai_very_low_tiny at 2.8/wave, which is a trivially small problem).

### PMC counter limitations

On gfx1100 (RDNA3), only 3 per-dispatch PMC counters report non-zero values: SQ_WAVES, SQ_BUSY_CYCLES, LDSBankConflict. All instruction mix, cache, memory bandwidth, and utilization counters return 0. The analysis therefore relies primarily on dispatch metadata (VGPRs, LDS, grid size) and timing-based roofline analysis.
