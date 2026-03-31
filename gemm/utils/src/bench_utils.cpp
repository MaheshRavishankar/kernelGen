#include "bench_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace kernelgen {
namespace gemm {
namespace utils {

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

size_t dtypeSize(const std::string &dtype) {
  if (dtype == "f16" || dtype == "bf16")
    return 2;
  if (dtype == "f32")
    return 4;
  throw std::runtime_error("Unsupported dtype: " + dtype);
}

std::string readFile(const std::string &path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::vector<char> loadNpy(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot open: " + path);

  char magic[6];
  f.read(magic, 6);
  uint8_t major, minor;
  f.read(reinterpret_cast<char *>(&major), 1);
  f.read(reinterpret_cast<char *>(&minor), 1);

  uint32_t headerLen = 0;
  if (major == 1) {
    uint16_t hl;
    f.read(reinterpret_cast<char *>(&hl), 2);
    headerLen = hl;
  } else {
    f.read(reinterpret_cast<char *>(&headerLen), 4);
  }
  f.seekg(headerLen, std::ios::cur);

  std::vector<char> data((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
  return data;
}

std::string jsonString(const std::string &json, const std::string &key) {
  auto pos = json.find("\"" + key + "\"");
  if (pos == std::string::npos)
    return "";
  pos = json.find(':', pos);
  auto start = json.find('"', pos + 1);
  auto end = json.find('"', start + 1);
  return json.substr(start + 1, end - start - 1);
}

double jsonNumber(const std::string &json, const std::string &key,
                  double defaultVal) {
  auto pos = json.find("\"" + key + "\"");
  if (pos == std::string::npos)
    return defaultVal;
  pos = json.find(':', pos);
  std::string rest = json.substr(pos + 1);
  return std::stod(rest);
}

bool jsonBool(const std::string &json, const std::string &key,
              bool defaultVal) {
  auto pos = json.find("\"" + key + "\"");
  if (pos == std::string::npos)
    return defaultVal;
  pos = json.find(':', pos);
  auto rest = json.substr(pos + 1);
  return rest.find("true") < rest.find("false");
}

GemmBenchConfig parseGemmConfig(const std::string &configPath) {
  std::string json = readFile(configPath);
  GemmBenchConfig c;
  c.M = static_cast<int64_t>(jsonNumber(json, "M"));
  c.N = static_cast<int64_t>(jsonNumber(json, "N"));
  c.K = static_cast<int64_t>(jsonNumber(json, "K"));
  c.transA = jsonBool(json, "transA");
  c.transB = jsonBool(json, "transB");
  c.alpha = static_cast<float>(jsonNumber(json, "alpha", 1.0));
  c.beta = static_cast<float>(jsonNumber(json, "beta", 0.0));
  c.dtype_A = jsonString(json, "dtype_A");
  c.dtype_B = jsonString(json, "dtype_B");
  c.dtype_C = jsonString(json, "dtype_C");
  c.compute_type = jsonString(json, "compute_type");
  if (c.dtype_A.empty())
    c.dtype_A = "f16";
  if (c.dtype_B.empty())
    c.dtype_B = "f16";
  if (c.dtype_C.empty())
    c.dtype_C = "f16";
  if (c.compute_type.empty())
    c.compute_type = "f32";
  return c;
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

static float halfToFloat(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign << 31;
    } else {
      exp = 1;
      while (!(mant & 0x400)) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3ff;
      f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = (sign << 31) | 0x7f800000 | (mant << 13);
  } else {
    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float result;
  std::memcpy(&result, &f, 4);
  return result;
}

static float bf16ToFloat(uint16_t raw) {
  uint32_t f32 = static_cast<uint32_t>(raw) << 16;
  float result;
  std::memcpy(&result, &f32, 4);
  return result;
}

VerifyResult verify(const char *actual, const char *ref, size_t sizeBytes,
                    const std::string &dtype, double relTol, double absTol) {
  VerifyResult v;
  size_t elemSize = dtypeSize(dtype);
  v.num_elements = sizeBytes / elemSize;

  for (size_t i = 0; i < v.num_elements; ++i) {
    float a, e;
    if (dtype == "f16") {
      uint16_t aRaw, eRaw;
      std::memcpy(&aRaw, actual + i * 2, 2);
      std::memcpy(&eRaw, ref + i * 2, 2);
      a = halfToFloat(aRaw);
      e = halfToFloat(eRaw);
    } else if (dtype == "f32") {
      std::memcpy(&a, actual + i * 4, 4);
      std::memcpy(&e, ref + i * 4, 4);
    } else { // bf16
      uint16_t aRaw, eRaw;
      std::memcpy(&aRaw, actual + i * 2, 2);
      std::memcpy(&eRaw, ref + i * 2, 2);
      a = bf16ToFloat(aRaw);
      e = bf16ToFloat(eRaw);
    }

    double absErr = std::fabs(a - e);
    double denom = std::max(std::fabs(static_cast<double>(e)),
                            std::fabs(static_cast<double>(a)));
    double relErr = (denom > 0) ? absErr / denom : 0.0;

    v.max_abs_err = std::max(v.max_abs_err, absErr);
    v.max_rel_err = std::max(v.max_rel_err, relErr);

    if (absErr > absTol && relErr > relTol)
      v.mismatches++;
  }

  v.pass = (v.mismatches == 0);
  return v;
}

void printVerifyJson(const VerifyResult &v) {
  std::cout << ", \"verified\": " << (v.pass ? "true" : "false")
            << ", \"max_rel_err\": " << v.max_rel_err
            << ", \"max_abs_err\": " << v.max_abs_err
            << ", \"mismatches\": " << v.mismatches
            << ", \"num_elements\": " << v.num_elements;
}

// ---------------------------------------------------------------------------
// Data init
// ---------------------------------------------------------------------------

void fillDeterministic(char *buf, size_t bytes) {
  for (size_t i = 0; i < bytes; ++i)
    buf[i] = static_cast<char>((i * 7 + 13) & 0xFF);
}

} // namespace utils
} // namespace gemm
} // namespace kernelgen
