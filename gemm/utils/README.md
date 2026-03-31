# GEMM Utilities

Shared C++ utilities used by all GEMM kernel providers.

## Libraries

### `kernelgen_gemm_utils` (pure C++)

No HIP dependency. Contains:

- **Config parsing**: `dtypeSize`, `readFile`, `loadNpy`, JSON extractors, `parseGemmConfig`
- **Verification**: `verify` (compares output vs reference with tolerance), `printVerifyJson`
- **Data init**: `fillDeterministic` (host-side deterministic byte pattern)

### `kernelgen_gemm_hip_utils` (HIP-dependent)

Links `kernelgen_gemm_utils` + `hip::amdhip64`. Contains:

- `allocDeviceDeterministic` — `hipMalloc` + fill with deterministic pattern
- `allocDeviceFromNpy` — `hipMalloc` + load .npy file to device

## Usage

Providers link against `kernelgen_gemm_hip_utils` which transitively provides both libraries:

```cmake
target_link_libraries(my_bench PRIVATE kernelgen_gemm_hip_utils)
```
