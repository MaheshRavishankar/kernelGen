#include "hipblaslt_gemm.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kernelgen::gemm::hipblaslt;

namespace {

size_t dtypeSize(const std::string &dtype) {
  if (dtype == "f16" || dtype == "bf16")
    return 2;
  if (dtype == "f32")
    return 4;
  throw std::runtime_error("Unsupported dtype: " + dtype);
}

// Minimal npy loader: reads shape and raw data from a .npy file.
std::vector<char> loadNpy(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot open: " + path);

  // Skip the npy header: magic(6) + version(2) + header_len(2 or 4).
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

  // Read remaining bytes.
  std::vector<char> data((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
  return data;
}

// Simple JSON string value extractor.
std::string jsonString(const std::string &json, const std::string &key) {
  auto pos = json.find("\"" + key + "\"");
  if (pos == std::string::npos)
    return "";
  pos = json.find(':', pos);
  auto start = json.find('"', pos + 1);
  auto end = json.find('"', start + 1);
  return json.substr(start + 1, end - start - 1);
}

// Simple JSON number extractor.
double jsonNumber(const std::string &json, const std::string &key,
                  double defaultVal = 0) {
  auto pos = json.find("\"" + key + "\"");
  if (pos == std::string::npos)
    return defaultVal;
  pos = json.find(':', pos);
  std::string rest = json.substr(pos + 1);
  return std::stod(rest);
}

// Simple JSON bool extractor.
bool jsonBool(const std::string &json, const std::string &key,
              bool defaultVal = false) {
  auto pos = json.find("\"" + key + "\"");
  if (pos == std::string::npos)
    return defaultVal;
  pos = json.find(':', pos);
  auto rest = json.substr(pos + 1);
  return rest.find("true") < rest.find("false");
}

std::string readFile(const std::string &path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

void printUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0
            << " --config <config.json> --input-a <a.npy> --input-b <b.npy>"
            << " [--warmup N] [--timed N] [--reference <c.npy>]\n";
}

} // namespace

int main(int argc, char **argv) {
  std::string configPath, inputAPath, inputBPath, refPath;
  int warmup = 5, timed = 20;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc)
      configPath = argv[++i];
    else if (arg == "--input-a" && i + 1 < argc)
      inputAPath = argv[++i];
    else if (arg == "--input-b" && i + 1 < argc)
      inputBPath = argv[++i];
    else if (arg == "--reference" && i + 1 < argc)
      refPath = argv[++i];
    else if (arg == "--warmup" && i + 1 < argc)
      warmup = std::stoi(argv[++i]);
    else if (arg == "--timed" && i + 1 < argc)
      timed = std::stoi(argv[++i]);
    else {
      printUsage(argv[0]);
      return 1;
    }
  }

  if (configPath.empty() || inputAPath.empty() || inputBPath.empty()) {
    printUsage(argv[0]);
    return 1;
  }

  std::string json = readFile(configPath);

  GemmConfig config;
  config.M = static_cast<int64_t>(jsonNumber(json, "M"));
  config.N = static_cast<int64_t>(jsonNumber(json, "N"));
  config.K = static_cast<int64_t>(jsonNumber(json, "K"));
  config.transA = jsonBool(json, "transA");
  config.transB = jsonBool(json, "transB");
  config.alpha = static_cast<float>(jsonNumber(json, "alpha", 1.0));
  config.beta = static_cast<float>(jsonNumber(json, "beta", 0.0));
  config.dtype_A = jsonString(json, "dtype_A");
  config.dtype_B = jsonString(json, "dtype_B");
  config.dtype_C = jsonString(json, "dtype_C");
  config.compute_type = jsonString(json, "compute_type");

  if (config.dtype_A.empty())
    config.dtype_A = "f16";
  if (config.dtype_B.empty())
    config.dtype_B = "f16";
  if (config.dtype_C.empty())
    config.dtype_C = "f16";
  if (config.compute_type.empty())
    config.compute_type = "f32";

  // Load inputs.
  auto dataA = loadNpy(inputAPath);
  auto dataB = loadNpy(inputBPath);

  size_t sizeA = config.M * config.K * dtypeSize(config.dtype_A);
  size_t sizeB = config.K * config.N * dtypeSize(config.dtype_B);
  size_t sizeC = config.M * config.N * dtypeSize(config.dtype_C);

  // Allocate and copy to device.
  void *dA, *dB, *dC;
  hipMalloc(&dA, sizeA);
  hipMalloc(&dB, sizeB);
  hipMalloc(&dC, sizeC);
  hipMemcpy(dA, dataA.data(), sizeA, hipMemcpyHostToDevice);
  hipMemcpy(dB, dataB.data(), sizeB, hipMemcpyHostToDevice);
  hipMemset(dC, 0, sizeC);

  GemmResult result = run(config, dA, dB, dC, warmup, timed);

  // Output JSON to stdout.
  std::cout << "{\"provider\": \"hipblaslt\""
            << ", \"kernel_time_us\": " << result.kernel_time_us
            << ", \"success\": " << (result.success ? "true" : "false");
  if (!result.error.empty())
    std::cout << ", \"error\": \"" << result.error << "\"";

  // Verify against reference with tolerance.
  if (!refPath.empty() && result.success) {
    auto refData = loadNpy(refPath);
    std::vector<char> hostC(sizeC);
    hipMemcpy(hostC.data(), dC, sizeC, hipMemcpyDeviceToHost);

    size_t elemSize = dtypeSize(config.dtype_C);
    size_t numElems = sizeC / elemSize;
    double maxRelErr = 0.0;
    double maxAbsErr = 0.0;
    size_t mismatches = 0;
    // Tolerance: element passes if absErr <= absTol OR relErr <= relTol.
    double relTol = 1e-2; // 1% relative tolerance
    double absTol =
        5e-2; // absolute tolerance (f16 has ~1e-3 precision at unit scale)

    for (size_t i = 0; i < numElems; ++i) {
      float actual, expected;
      if (config.dtype_C == "f16") {
        // f16 stored as uint16_t, convert via half-float bit manipulation.
        uint16_t aRaw, eRaw;
        std::memcpy(&aRaw, hostC.data() + i * 2, 2);
        std::memcpy(&eRaw, refData.data() + i * 2, 2);
        // IEEE 754 half to float conversion.
        auto halfToFloat = [](uint16_t h) -> float {
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
        };
        actual = halfToFloat(aRaw);
        expected = halfToFloat(eRaw);
      } else if (config.dtype_C == "f32") {
        std::memcpy(&actual, hostC.data() + i * 4, 4);
        std::memcpy(&expected, refData.data() + i * 4, 4);
      } else {
        // bf16: stored as uint16_t, upper 16 bits of float32.
        uint16_t aRaw, eRaw;
        std::memcpy(&aRaw, hostC.data() + i * 2, 2);
        std::memcpy(&eRaw, refData.data() + i * 2, 2);
        uint32_t af = static_cast<uint32_t>(aRaw) << 16;
        uint32_t ef = static_cast<uint32_t>(eRaw) << 16;
        std::memcpy(&actual, &af, 4);
        std::memcpy(&expected, &ef, 4);
      }

      double absErr = std::fabs(actual - expected);
      double denom = std::max(std::fabs(expected), std::fabs(actual));
      double relErr = (denom > 0) ? absErr / denom : 0.0;

      maxAbsErr = std::max(maxAbsErr, absErr);
      maxRelErr = std::max(maxRelErr, relErr);

      if (absErr > absTol && relErr > relTol)
        mismatches++;
    }

    bool pass = (mismatches == 0);
    std::cout << ", \"verified\": " << (pass ? "true" : "false")
              << ", \"max_rel_err\": " << maxRelErr
              << ", \"max_abs_err\": " << maxAbsErr
              << ", \"mismatches\": " << mismatches
              << ", \"num_elements\": " << numElems;
  }

  std::cout << "}" << std::endl;

  hipFree(dA);
  hipFree(dB);
  hipFree(dC);

  return result.success ? 0 : 1;
}
