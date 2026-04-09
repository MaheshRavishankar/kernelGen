/// Fusilli GEMM benchmark executable.
///
/// Builds a GEMM graph with the Fusilli frontend, JIT-compiles it on an
/// external HIP stream, and measures kernel time with HIP events.

#include "bench_utils.h"

#include <fusilli.h>

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime_api.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace kernelgen::gemm::utils;

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _err = (expr);                                                  \
    if (_err != hipSuccess) {                                                  \
      std::cerr << "HIP error code: " << static_cast<int>(_err) << "\n";       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

namespace {

using fusilli::Backend;
using fusilli::bf16;
using fusilli::Buffer;
using fusilli::DataType;
using fusilli::ErrorObject;
using fusilli::Graph;
using fusilli::half;
using fusilli::Handle;
using fusilli::MatmulAttr;
using fusilli::TensorAttr;
using VariantPack =
    std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>;

bool isOkOrPrint(const ErrorObject &error, const std::string &context) {
  if (fusilli::isError(error)) {
    std::cerr << "Fusilli error in " << context << ": " << error << "\n";
    return false;
  }
  return true;
}

std::string errorMessage(const ErrorObject &error) {
  std::ostringstream oss;
  oss << error;
  return oss.str();
}

template <typename T>
std::string errorMessage(const fusilli::ErrorOr<T> &errorOr) {
  return errorMessage(static_cast<ErrorObject>(errorOr));
}

template <typename T>
std::vector<T> bytesToVector(const std::vector<char> &bytes) {
  if (bytes.size() % sizeof(T) != 0) {
    throw std::runtime_error("Input size is not a multiple of element size");
  }
  std::vector<T> out(bytes.size() / sizeof(T));
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

DataType dtypeToFusilli(const std::string &dtype) {
  if (dtype == "f16")
    return DataType::Half;
  if (dtype == "bf16")
    return DataType::BFloat16;
  if (dtype == "f32")
    return DataType::Float;
  throw std::runtime_error("Unsupported dtype for Fusilli: " + dtype);
}

std::vector<int64_t> gemmDims(int64_t rows, int64_t cols) {
  return {rows, cols};
}

std::vector<int64_t> gemmStrides(int64_t rows, int64_t cols, bool transpose) {
  return transpose ? std::vector<int64_t>{1, rows}
                   : std::vector<int64_t>{cols, 1};
}

template <typename T>
std::shared_ptr<Buffer>
allocateBuffer(const Handle &handle, const std::shared_ptr<TensorAttr> &tensor,
               std::vector<T> data) {
  std::vector<iree_hal_dim_t> shape;
  for (auto dim : tensor->getPhysicalDim()) {
    shape.push_back(static_cast<iree_hal_dim_t>(dim));
  }

  auto bufferOr = Buffer::allocate(handle, shape, data);
  if (fusilli::isError(bufferOr)) {
    throw std::runtime_error(errorMessage(bufferOr));
  }
  return std::make_shared<Buffer>(std::move(*bufferOr));
}

template <typename T>
std::shared_ptr<Buffer>
allocateZeroBuffer(const Handle &handle, const std::shared_ptr<TensorAttr> &t) {
  return allocateBuffer(handle, t,
                        std::vector<T>(static_cast<size_t>(t->getVolume()),
                                       static_cast<T>(0.0f)));
}

template <typename T>
std::shared_ptr<Buffer>
makeInputBuffer(const Handle &handle, const std::shared_ptr<TensorAttr> &tensor,
                const std::string &inputPath, size_t expectedSize,
                bool randomInit) {
  std::vector<T> host;
  if (randomInit) {
    std::vector<char> bytes(expectedSize);
    fillDeterministic(bytes.data(), bytes.size());
    host = bytesToVector<T>(bytes);
  } else {
    host = bytesToVector<T>(loadNpy(inputPath));
  }
  return allocateBuffer(handle, tensor, std::move(host));
}

template <typename T>
std::vector<char> readOutputBytes(const Handle &handle,
                                  const std::shared_ptr<Buffer> &buffer) {
  std::vector<T> host;
  auto status = buffer->read(handle, host);
  if (fusilli::isError(status)) {
    throw std::runtime_error(errorMessage(status));
  }
  std::vector<char> bytes(host.size() * sizeof(T));
  std::memcpy(bytes.data(), host.data(), bytes.size());
  return bytes;
}

void printUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0 << " --config <config.json>"
            << " [--input-a <a.npy> --input-b <b.npy>]"
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

  if (configPath.empty()) {
    printUsage(argv[0]);
    return 1;
  }

  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;

  try {
    const bool randomInit = inputAPath.empty() || inputBPath.empty();
    const auto cfg = parseGemmConfig(configPath);

    if (cfg.alpha != 1.0f || cfg.beta != 0.0f) {
      throw std::runtime_error(
          "Fusilli GEMM benchmark supports only alpha=1 and beta=0");
    }
    if (cfg.compute_type != "f32") {
      throw std::runtime_error(
          "Fusilli GEMM benchmark expects compute_type=f32");
    }
    if (cfg.dtype_A != cfg.dtype_B || cfg.dtype_A != cfg.dtype_C) {
      throw std::runtime_error("Fusilli GEMM benchmark currently expects "
                               "matching input/output dtypes");
    }

    const size_t sizeA =
        static_cast<size_t>(cfg.M) * cfg.K * dtypeSize(cfg.dtype_A);
    const size_t sizeB =
        static_cast<size_t>(cfg.K) * cfg.N * dtypeSize(cfg.dtype_B);
    const size_t sizeC =
        static_cast<size_t>(cfg.M) * cfg.N * dtypeSize(cfg.dtype_C);

    HIP_CHECK(hipStreamCreate(&stream));

    auto handleOr = Handle::create(Backend::AMDGPU, /*deviceId=*/0,
                                   reinterpret_cast<uintptr_t>(stream));
    if (fusilli::isError(handleOr)) {
      throw std::runtime_error(errorMessage(handleOr));
    }
    Handle handle = std::move(*handleOr);

    auto graph = std::make_shared<Graph>();
    graph
        ->setName("kernelgen_fusilli_gemm_" + std::to_string(cfg.M) + "x" +
                  std::to_string(cfg.N) + "x" + std::to_string(cfg.K))
        .setIODataType(dtypeToFusilli(cfg.dtype_C))
        .setComputeDataType(dtypeToFusilli(cfg.compute_type));

    auto aTensor =
        graph->tensor(TensorAttr()
                          .setName("matrix_a")
                          .setDim(gemmDims(cfg.M, cfg.K))
                          .setStride(gemmStrides(cfg.M, cfg.K, cfg.transA)));
    auto bTensor =
        graph->tensor(TensorAttr()
                          .setName("matrix_b")
                          .setDim(gemmDims(cfg.K, cfg.N))
                          .setStride(gemmStrides(cfg.K, cfg.N, cfg.transB)));
    MatmulAttr matmulAttr;
    matmulAttr.setName("gemm_matmul");
    auto cTensor = graph->matmul(aTensor, bTensor, matmulAttr);
    cTensor->setOutput(true);

    if (!isOkOrPrint(graph->validate(), "graph validation") ||
        !isOkOrPrint(graph->compile(handle), "graph compile")) {
      std::cout << "{\"provider\": \"fusilli\", \"kernel_time_us\": 0"
                << ", \"success\": false"
                << ", \"error\": \"fusilli graph build failed\"}" << std::endl;
      if (stream)
        HIP_CHECK(hipStreamDestroy(stream));
      return 1;
    }

    std::shared_ptr<Buffer> aBuffer;
    std::shared_ptr<Buffer> bBuffer;
    std::shared_ptr<Buffer> cBuffer;
    std::shared_ptr<Buffer> workspace = nullptr;

    if (cfg.dtype_A == "f16") {
      aBuffer =
          makeInputBuffer<half>(handle, aTensor, inputAPath, sizeA, randomInit);
      bBuffer =
          makeInputBuffer<half>(handle, bTensor, inputBPath, sizeB, randomInit);
      cBuffer = allocateZeroBuffer<half>(handle, cTensor);
    } else if (cfg.dtype_A == "bf16") {
      aBuffer =
          makeInputBuffer<bf16>(handle, aTensor, inputAPath, sizeA, randomInit);
      bBuffer =
          makeInputBuffer<bf16>(handle, bTensor, inputBPath, sizeB, randomInit);
      cBuffer = allocateZeroBuffer<bf16>(handle, cTensor);
    } else if (cfg.dtype_A == "f32") {
      aBuffer = makeInputBuffer<float>(handle, aTensor, inputAPath, sizeA,
                                       randomInit);
      bBuffer = makeInputBuffer<float>(handle, bTensor, inputBPath, sizeB,
                                       randomInit);
      cBuffer = allocateZeroBuffer<float>(handle, cTensor);
    } else {
      throw std::runtime_error("Unsupported GEMM dtype");
    }

    auto workspaceSize = graph->getWorkspaceSize().value_or(0);
    if (workspaceSize > 0) {
      auto wsOr = Buffer::allocateRaw(handle, workspaceSize);
      if (fusilli::isError(wsOr)) {
        throw std::runtime_error(errorMessage(wsOr));
      }
      workspace = std::make_shared<Buffer>(std::move(*wsOr));
    }

    const VariantPack variantPack = {
        {aTensor, aBuffer},
        {bTensor, bBuffer},
        {cTensor, cBuffer},
    };

    for (int i = 0; i < warmup; ++i) {
      if (!isOkOrPrint(graph->execute(handle, variantPack, workspace),
                       "warmup execute")) {
        std::cout << "{\"provider\": \"fusilli\", \"kernel_time_us\": 0"
                  << ", \"success\": false"
                  << ", \"error\": \"warmup invocation failed\"}" << std::endl;
        if (stream)
          HIP_CHECK(hipStreamDestroy(stream));
        return 1;
      }
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < timed; ++i) {
      if (!isOkOrPrint(graph->execute(handle, variantPack, workspace),
                       "timed execute")) {
        std::cout << "{\"provider\": \"fusilli\", \"kernel_time_us\": 0"
                  << ", \"success\": false"
                  << ", \"error\": \"timed invocation failed\"}" << std::endl;
        if (start)
          HIP_CHECK(hipEventDestroy(start));
        if (stop)
          HIP_CHECK(hipEventDestroy(stop));
        if (stream)
          HIP_CHECK(hipStreamDestroy(stream));
        return 1;
      }
    }
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsedMs = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsedMs, start, stop));
    const double avgUs = (elapsedMs * 1000.0) / timed;

    std::cout << "{\"provider\": \"fusilli\""
              << ", \"kernel_time_us\": " << avgUs << ", \"success\": true";

    if (!refPath.empty() && !randomInit) {
      const auto refData = loadNpy(refPath);
      std::vector<char> hostC;
      if (cfg.dtype_C == "f16") {
        hostC = readOutputBytes<half>(handle, cBuffer);
      } else if (cfg.dtype_C == "bf16") {
        hostC = readOutputBytes<bf16>(handle, cBuffer);
      } else {
        hostC = readOutputBytes<float>(handle, cBuffer);
      }
      const auto verifyResult =
          verify(hostC.data(), refData.data(), sizeC, cfg.dtype_C);
      printVerifyJson(verifyResult);
    }

    std::cout << "}" << std::endl;

    if (start)
      HIP_CHECK(hipEventDestroy(start));
    if (stop)
      HIP_CHECK(hipEventDestroy(stop));
    if (stream)
      HIP_CHECK(hipStreamDestroy(stream));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Fusilli benchmark failed: " << e.what() << "\n";
    if (start)
      static_cast<void>(hipEventDestroy(start));
    if (stop)
      static_cast<void>(hipEventDestroy(stop));
    if (stream)
      static_cast<void>(hipStreamDestroy(stream));
    return 1;
  }
}
