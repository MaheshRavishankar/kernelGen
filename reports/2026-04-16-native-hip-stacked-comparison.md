# Native HIP stacked vs hipBLASLt comparison

- Branch: users/MaheshRavishankar/nativeHipStackedPerfComparison
- Native HIP results: /tmp/kernelgen-nativehip-stacked-comparison/native_hip.json
- hipBLASLt results: /tmp/kernelgen-nativehip-stacked-comparison/hipblaslt.json
- Build: /home/mahesh/kernelGen/build/nativeHipStackedPerfComparison-Release
- Tests compared: 17
- Native faster: 7; near parity (+/-1%): 1; hipBLASLt faster: 9
- Geomean native speedup vs hipBLASLt: 0.868x

## Per-test results

| test | shape | dtype | native us | native TFLOPS | hipBLASLt us | hipBLASLt TFLOPS | native speedup | verify |
|---|---:|---|---:|---:|---:|---:|---:|---|
| ai_high_large_k | 4096x1024x150000 | bf16 | 23038.60 | 54.62 | 19914.73 | 63.18 | 0.864x | |
| ai_high_medium | 1285x2048x3840 | bf16 | 458.18 | 44.11 | 352.49 | 57.34 | 0.769x | native=True hipblaslt=True |
| ai_high_small | 576x576x1280 | bf16 | 79.235 | 10.72 | 26.700 | 31.81 | 0.337x | native=True hipblaslt=True |
| ai_low_large_flat | 21760x3840x20 | bf16 | 634.25 | 5.27 | 607.07 | 5.51 | 0.957x | |
| ai_low_skinny | 16x512x1024 | bf16 | 14.972 | 1.12 | 9.100 | 1.84 | 0.608x | native=True hipblaslt=True |
| ai_low_small | 32x576x2304 | bf16 | 29.208 | 2.91 | 13.450 | 6.31 | 0.460x | native=True hipblaslt=True |
| ai_medium_extreme | 16800000x128x134 | bf16 | 17628.60 | 32.69 | 25375.04 | 22.71 | 1.439x | |
| ai_medium_large | 7680x512x304 | bf16 | 45.928 | 52.05 | 50.550 | 47.29 | 1.101x | native=True hipblaslt=True |
| ai_medium_small | 576x576x165 | bf16 | 15.840 | 6.91 | 9.830 | 11.14 | 0.621x | native=True hipblaslt=True |
| ai_very_high_extreme | 150000x16384x4096 | bf16 | 456537.00 | 44.10 | 310266.70 | 64.89 | 0.680x | |
| ai_very_high_large | 11520x3840x3840 | bf16 | 4284.11 | 79.30 | 4400.67 | 77.20 | 1.027x | |
| ai_very_high_medium | 3840x3840x2304 | bf16 | 888.71 | 76.46 | 926.65 | 73.33 | 1.043x | |
| ai_very_high_square | 4096x4096x4096 | bf16 | 1828.00 | 75.19 | 1815.78 | 75.69 | 0.993x | |
| ai_very_low_small_square | 576x576x10 | bf16 | 5.810 | 1.14 | 6.870 | 0.97 | 1.182x | native=True hipblaslt=True |
| ai_very_low_small_wide | 576x2304x10 | bf16 | 7.560 | 3.51 | 11.950 | 2.22 | 1.581x | native=True hipblaslt=True |
| ai_very_low_tiny | 4x384x5 | bf16 | 4.640 | 0.00 | 6.290 | 0.00 | 1.356x | native=True hipblaslt=True |
| small_f16 | 1024x1024x1024 | f16 | 44.670 | 48.07 | 37.290 | 57.59 | 0.835x | native=True hipblaslt=True |

## Largest native wins

- ai_very_low_small_wide: 1.581x (7.560 us native vs 11.950 us hipBLASLt)
- ai_medium_extreme: 1.439x (17628.60 us native vs 25375.04 us hipBLASLt)
- ai_very_low_tiny: 1.356x (4.640 us native vs 6.290 us hipBLASLt)
- ai_very_low_small_square: 1.182x (5.810 us native vs 6.870 us hipBLASLt)
- ai_medium_large: 1.101x (45.928 us native vs 50.550 us hipBLASLt)

## Largest native gaps

- ai_high_small: 0.337x (79.235 us native vs 26.700 us hipBLASLt)
- ai_low_small: 0.460x (29.208 us native vs 13.450 us hipBLASLt)
- ai_low_skinny: 0.608x (14.972 us native vs 9.100 us hipBLASLt)
- ai_medium_small: 0.621x (15.840 us native vs 9.830 us hipBLASLt)
- ai_very_high_extreme: 0.680x (456537.00 us native vs 310266.70 us hipBLASLt)
