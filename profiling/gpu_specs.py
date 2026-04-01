"""GPU hardware specifications for analysis."""

GPU_SPECS = {
    "gfx1100": {
        "peak_tflops_bf16": 123.0,
        "peak_tflops_f16": 123.0,
        "peak_tflops_f32": 61.5,
        "peak_bandwidth_gbs": 864.0,
        "num_cus": 48,
        "max_waves_per_cu": 32,  # 16 per SIMD, 2 SIMDs
        "lds_per_cu_bytes": 65536,
        "vgpr_per_cu": 1536,  # per SIMD, in units of 32-wide registers
        "wave_size": 32,
    },
}
