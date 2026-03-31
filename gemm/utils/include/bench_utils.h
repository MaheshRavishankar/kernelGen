#ifndef KERNELGEN_GEMM_BENCH_UTILS_H
#define KERNELGEN_GEMM_BENCH_UTILS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kernelgen {
namespace gemm {
namespace utils {

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

/// Returns the byte size of a dtype string ("f16", "bf16" → 2, "f32" → 4).
size_t dtypeSize(const std::string &dtype);

/// Read an entire file into a string.
std::string readFile(const std::string &path);

/// Load the raw data portion of a .npy file (skipping the header).
std::vector<char> loadNpy(const std::string &path);

/// Simple JSON extractors (no full parser needed).
std::string jsonString(const std::string &json, const std::string &key);
double jsonNumber(const std::string &json, const std::string &key,
                  double defaultVal = 0);
bool jsonBool(const std::string &json, const std::string &key,
              bool defaultVal = false);

// ---------------------------------------------------------------------------
// GEMM config parsed from JSON
// ---------------------------------------------------------------------------

struct GemmBenchConfig {
  int64_t M = 0, N = 0, K = 0;
  bool transA = false, transB = false;
  float alpha = 1.0f, beta = 0.0f;
  std::string dtype_A = "f16", dtype_B = "f16", dtype_C = "f16";
  std::string compute_type = "f32";
};

/// Parse a GEMM config from a JSON file.
GemmBenchConfig parseGemmConfig(const std::string &configPath);

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

struct VerifyResult {
  bool pass = false;
  double max_rel_err = 0;
  double max_abs_err = 0;
  size_t mismatches = 0;
  size_t num_elements = 0;
};

/// Compare output buffer against reference .npy data.
/// Both |actual| and |ref| are raw byte buffers in the given dtype.
VerifyResult verify(const char *actual, const char *ref, size_t sizeBytes,
                    const std::string &dtype, double relTol = 1e-2,
                    double absTol = 5e-2);

/// Print verification results as JSON fields to stdout.
void printVerifyJson(const VerifyResult &v);

// ---------------------------------------------------------------------------
// Data init
// ---------------------------------------------------------------------------

/// Fill a host buffer with a deterministic byte pattern.
void fillDeterministic(char *buf, size_t bytes);

} // namespace utils
} // namespace gemm
} // namespace kernelgen

#endif // KERNELGEN_GEMM_BENCH_UTILS_H
