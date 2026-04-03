#pragma once
#ifndef KERNELGEN_WMMA_UTILS_H
#define KERNELGEN_WMMA_UTILS_H

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// =============================================================================
// Shared vector types and __device__ helpers for WMMA GEMM kernels
// =============================================================================

// BF16 fragments are passed as int16 vectors to the WMMA built-in.
// Each lane holds 16 BF16 values (one row of a 16x16 tile).
using bf16x16 = short __attribute__((ext_vector_type(16)));

// F16 fragments for the F16 WMMA intrinsic.
using f16x16 = _Float16 __attribute__((ext_vector_type(16)));

// F32 accumulator: each lane holds 8 floats of the 16x16 output tile.
using f32x8 = float __attribute__((ext_vector_type(8)));

// =============================================================================
// WmmaTraits: select fragment type and WMMA intrinsic by element type
// =============================================================================

template <typename T> struct WmmaTraits;

template <> struct WmmaTraits<__hip_bfloat16> {
  using frag_type = bf16x16;
  static __device__ __forceinline__ f32x8 wmma(bf16x16 a, bf16x16 b, f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
  }
};

template <> struct WmmaTraits<_Float16> {
  using frag_type = f16x16;
  static __device__ __forceinline__ f32x8 wmma(f16x16 a, f16x16 b, f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  }
};

// Load/store 128 bits in a single global_load_b128 instruction.
template <typename T>
__device__ __forceinline__ void load_128(T *dst, const T *src) {
  *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
}

// Zero-fill 128 bits in a single instruction.
template <typename T> __device__ __forceinline__ void zero_128(T *dst) {
  *reinterpret_cast<uint4 *>(dst) = make_uint4(0, 0, 0, 0);
}

// WMMA hardware tile dimensions (fixed by the ISA).
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

#endif // KERNELGEN_WMMA_UTILS_H
