/// IREE GEMM benchmark executable.
///
/// Loads a pre-compiled .vmfb module, dispatches the GEMM on an external
/// HIP stream, and measures kernel time with HIP events — identical
/// methodology to the hipBLAS-LT benchmark.

#include "bench_utils.h"
#include "hip_utils.h"

#include <iree/async/util/proactor_pool.h>
#include <iree/hal/api.h>
#include <iree/hal/drivers/hip/api.h>
#include <iree/io/file_contents.h>
#include <iree/modules/hal/module.h>
#include <iree/modules/hal/types.h>
#include <iree/vm/api.h>
#include <iree/vm/bytecode/module.h>

// HIP runtime API for stream/event timing.
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>

using namespace kernelgen::gemm::utils;

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _err = (expr);                                                  \
    if (_err != hipSuccess) {                                                  \
      std::cerr << "HIP error code: " << static_cast<int>(_err) << "\n";       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define IREE_CHECK(expr, msg)                                                  \
  do {                                                                         \
    iree_status_t _s = (expr);                                                 \
    if (!iree_status_is_ok(_s)) {                                              \
      iree_status_fprint(stderr, _s);                                          \
      iree_status_free(_s);                                                    \
      std::cerr << "IREE error: " << msg << "\n";                              \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

iree_hal_element_type_t dtypeToIree(const std::string &dtype) {
  if (dtype == "f16")
    return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
  if (dtype == "bf16")
    return IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
  if (dtype == "f32")
    return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
  throw std::runtime_error("Unsupported dtype for IREE: " + dtype);
}

void printUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0
            << " --config <config.json> --vmfb <module.vmfb>"
            << " [--input-a <a.npy> --input-b <b.npy>]"
            << " [--warmup N] [--timed N] [--reference <c.npy>]\n";
}

} // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  std::string configPath, vmfbPath, inputAPath, inputBPath, refPath;
  int warmup = 5, timed = 20;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc)
      configPath = argv[++i];
    else if (arg == "--vmfb" && i + 1 < argc)
      vmfbPath = argv[++i];
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

  if (configPath.empty() || vmfbPath.empty()) {
    printUsage(argv[0]);
    return 1;
  }

  bool randomInit = inputAPath.empty() || inputBPath.empty();

  // -- Parse config --------------------------------------------------------
  auto cfg = parseGemmConfig(configPath);

  size_t sizeA = cfg.M * cfg.K * dtypeSize(cfg.dtype_A);
  size_t sizeB = cfg.K * cfg.N * dtypeSize(cfg.dtype_B);
  size_t sizeC = cfg.M * cfg.N * dtypeSize(cfg.dtype_C);

  // -- Create HIP stream (for external-stream timing) ----------------------
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // -- IREE setup ----------------------------------------------------------
  iree_vm_instance_t *instance = nullptr;
  IREE_CHECK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                     iree_allocator_system(), &instance),
             "vm_instance_create");
  IREE_CHECK(iree_hal_module_register_all_types(instance),
             "hal_module_register_all_types");

  // Create HIP driver with external stream.
  iree_hal_hip_driver_options_t driver_options;
  iree_hal_hip_driver_options_initialize(&driver_options);

  iree_hal_hip_device_params_t device_params;
  iree_hal_hip_device_params_initialize(&device_params);
  device_params.external_stream = reinterpret_cast<uint64_t>(stream);
  device_params.allow_inline_execution = true;

  iree_hal_driver_t *driver = nullptr;
  IREE_CHECK(iree_hal_hip_driver_create(iree_make_cstring_view("hip"),
                                        &driver_options, &device_params,
                                        iree_allocator_system(), &driver),
             "hip_driver_create");

  // Create proactor pool (required for HIP device).
  iree_async_proactor_pool_t *proactor_pool = nullptr;
  IREE_CHECK(iree_async_proactor_pool_create(
                 /*node_count=*/1, /*node_ids=*/nullptr,
                 iree_async_proactor_pool_options_default(),
                 iree_allocator_system(), &proactor_pool),
             "proactor_pool_create");

  iree_hal_device_t *device = nullptr;
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;
  IREE_CHECK(iree_hal_driver_create_default_device(
                 driver, &create_params, iree_allocator_system(), &device),
             "create_default_device");
  iree_async_proactor_pool_release(proactor_pool);

  // Create VM context and register HAL module.
  iree_vm_context_t *context = nullptr;
  IREE_CHECK(iree_vm_context_create(instance, IREE_VM_CONTEXT_FLAG_NONE,
                                    iree_allocator_system(), &context),
             "vm_context_create");

  iree_hal_device_group_t *device_group = nullptr;
  IREE_CHECK(iree_hal_device_group_create_from_device(
                 device, iree_allocator_system(), &device_group),
             "device_group_create_from_device");

  iree_vm_module_t *hal_module = nullptr;
  IREE_CHECK(iree_hal_module_create(instance,
                                    iree_hal_module_device_policy_default(),
                                    device_group, IREE_HAL_MODULE_FLAG_NONE,
                                    iree_hal_module_debug_sink_null(),
                                    iree_allocator_system(), &hal_module),
             "hal_module_create");
  IREE_CHECK(iree_vm_context_register_modules(context, 1, &hal_module),
             "register hal module");
  iree_vm_module_release(hal_module);
  iree_hal_device_group_release(device_group);

  // Load .vmfb bytecode module.
  iree_io_file_contents_t *file_contents = nullptr;
  IREE_CHECK(
      iree_io_file_contents_read(iree_make_cstring_view(vmfbPath.c_str()),
                                 iree_allocator_system(), &file_contents),
      "read vmfb file");

  iree_vm_module_t *bytecode_module = nullptr;
  IREE_CHECK(iree_vm_bytecode_module_create(
                 instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
                 file_contents->const_buffer,
                 iree_io_file_contents_deallocator(file_contents),
                 iree_allocator_system(), &bytecode_module),
             "bytecode_module_create");
  IREE_CHECK(iree_vm_context_register_modules(context, 1, &bytecode_module),
             "register bytecode module");
  iree_vm_module_release(bytecode_module);

  // Resolve the async entry point.
  iree_vm_function_t function;
  iree_status_t resolve_status = iree_vm_context_resolve_function(
      context, iree_make_cstring_view("module.main$async"), &function);
  bool use_async = iree_status_is_ok(resolve_status);
  if (!use_async) {
    iree_status_free(resolve_status);
    IREE_CHECK(iree_vm_context_resolve_function(
                   context, iree_make_cstring_view("module.main"), &function),
               "resolve function module.main");
  }

  // -- Allocate device memory (same as hipBLASLt) ---------------------------
  void *dA = randomInit ? allocDeviceDeterministic(sizeA)
                        : allocDeviceFromNpy(inputAPath, sizeA);
  void *dB = randomInit ? allocDeviceDeterministic(sizeB)
                        : allocDeviceFromNpy(inputBPath, sizeB);

  // -- Import HIP device pointers into IREE HAL ----------------------------
  iree_hal_allocator_t *allocator = iree_hal_device_allocator(device);
  iree_hal_buffer_params_t buffer_params = {};
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  auto importDeviceBuffer = [&](void *dPtr,
                                size_t size) -> iree_hal_buffer_t * {
    iree_hal_external_buffer_t ext = {};
    ext.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION;
    ext.flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE;
    ext.size = size;
    ext.handle.device_allocation.ptr = reinterpret_cast<uint64_t>(dPtr);
    iree_hal_buffer_t *buf = nullptr;
    iree_status_t s = iree_hal_allocator_import_buffer(
        allocator, buffer_params, &ext, iree_hal_buffer_release_callback_null(),
        &buf);
    if (!iree_status_is_ok(s)) {
      iree_status_fprint(stderr, s);
      iree_status_free(s);
      return nullptr;
    }
    return buf;
  };

  iree_hal_buffer_t *rawA = importDeviceBuffer(dA, sizeA);
  iree_hal_buffer_t *rawB = importDeviceBuffer(dB, sizeB);
  if (!rawA || !rawB) {
    std::cerr << "Failed to import device buffers into IREE HAL\n";
    return 1;
  }

  iree_hal_dim_t shapeA[] = {static_cast<iree_hal_dim_t>(cfg.M),
                             static_cast<iree_hal_dim_t>(cfg.K)};
  iree_hal_dim_t shapeB[] = {static_cast<iree_hal_dim_t>(cfg.K),
                             static_cast<iree_hal_dim_t>(cfg.N)};

  iree_hal_buffer_view_t *bufA = nullptr;
  IREE_CHECK(iree_hal_buffer_view_create(rawA, 2, shapeA,
                                         dtypeToIree(cfg.dtype_A),
                                         IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                         iree_allocator_system(), &bufA),
             "create buffer view A");

  iree_hal_buffer_view_t *bufB = nullptr;
  IREE_CHECK(iree_hal_buffer_view_create(rawB, 2, shapeB,
                                         dtypeToIree(cfg.dtype_B),
                                         IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                         iree_allocator_system(), &bufB),
             "create buffer view B");

  // -- Helper: create input list and invoke --------------------------------
  auto invoke = [&]() -> iree_status_t {
    iree_vm_list_t *inputs = nullptr;
    iree_host_size_t capacity = use_async ? 4 : 2;
    IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             capacity, iree_allocator_system(),
                                             &inputs));

    iree_vm_ref_t refA = iree_hal_buffer_view_retain_ref(bufA);
    iree_vm_list_push_ref_move(inputs, &refA);
    iree_vm_ref_t refB = iree_hal_buffer_view_retain_ref(bufB);
    iree_vm_list_push_ref_move(inputs, &refB);

    if (use_async) {
      iree_hal_fence_t *wait_fence = nullptr;
      iree_hal_fence_create(0, iree_allocator_system(), &wait_fence);
      iree_vm_ref_t wait_ref = iree_hal_fence_move_ref(wait_fence);
      iree_vm_list_push_ref_move(inputs, &wait_ref);

      iree_hal_fence_t *signal_fence = nullptr;
      iree_hal_fence_create(0, iree_allocator_system(), &signal_fence);
      iree_vm_ref_t signal_ref = iree_hal_fence_move_ref(signal_fence);
      iree_vm_list_push_ref_move(inputs, &signal_ref);
    }

    iree_vm_list_t *outputs = nullptr;
    IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             1, iree_allocator_system(),
                                             &outputs));

    iree_status_t status = iree_vm_invoke(
        context, function, IREE_VM_INVOCATION_FLAG_NONE,
        /*policy=*/nullptr, inputs, outputs, iree_allocator_system());
    iree_vm_list_release(outputs);
    iree_vm_list_release(inputs);
    return status;
  };

  // -- Warmup --------------------------------------------------------------
  for (int i = 0; i < warmup; ++i) {
    iree_status_t s = invoke();
    if (!iree_status_is_ok(s)) {
      iree_status_fprint(stderr, s);
      iree_status_free(s);
      std::cout << "{\"provider\": \"iree\", \"kernel_time_us\": 0"
                << ", \"success\": false"
                << ", \"error\": \"warmup invocation failed\"}" << std::endl;
      return 1;
    }
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  // -- Timed runs with HIP events ------------------------------------------
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start, stream));
  for (int i = 0; i < timed; ++i) {
    iree_status_t s = invoke();
    if (!iree_status_is_ok(s)) {
      iree_status_fprint(stderr, s);
      iree_status_free(s);
      std::cout << "{\"provider\": \"iree\", \"kernel_time_us\": 0"
                << ", \"success\": false"
                << ", \"error\": \"timed invocation failed\"}" << std::endl;
      return 1;
    }
  }
  HIP_CHECK(hipEventRecord(stop, stream));
  HIP_CHECK(hipEventSynchronize(stop));

  float elapsed_ms = 0;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
  double avg_us = (elapsed_ms * 1000.0) / timed;

  // -- JSON output ---------------------------------------------------------
  std::cout << "{\"provider\": \"iree\""
            << ", \"kernel_time_us\": " << avg_us << ", \"success\": true";

  // -- Verification (optional) ---------------------------------------------
  if (!refPath.empty() && !randomInit) {
    iree_vm_list_t *inputs = nullptr;
    iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                        iree_allocator_system(), &inputs);
    iree_vm_ref_t refA2 = iree_hal_buffer_view_retain_ref(bufA);
    iree_vm_list_push_ref_move(inputs, &refA2);
    iree_vm_ref_t refB2 = iree_hal_buffer_view_retain_ref(bufB);
    iree_vm_list_push_ref_move(inputs, &refB2);

    iree_vm_list_t *outputs = nullptr;
    iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                        iree_allocator_system(), &outputs);

    iree_vm_function_t sync_fn;
    iree_status_t rs = iree_vm_context_resolve_function(
        context, iree_make_cstring_view("module.main"), &sync_fn);
    if (iree_status_is_ok(rs)) {
      rs = iree_vm_invoke(context, sync_fn, IREE_VM_INVOCATION_FLAG_NONE,
                          nullptr, inputs, outputs, iree_allocator_system());
    }

    if (iree_status_is_ok(rs) && iree_vm_list_size(outputs) > 0) {
      iree_vm_ref_t out_ref = iree_vm_ref_null();
      iree_vm_list_get_ref_assign(outputs, 0, &out_ref);
      iree_hal_buffer_view_t *out_view = iree_hal_buffer_view_deref(out_ref);

      if (out_view) {
        iree_hal_buffer_t *out_buf = iree_hal_buffer_view_buffer(out_view);
        std::vector<char> hostC(sizeC);
        iree_hal_device_transfer_d2h(device, out_buf, 0, hostC.data(), sizeC,
                                     IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                     iree_infinite_timeout());

        auto refData = loadNpy(refPath);
        auto v = verify(hostC.data(), refData.data(), sizeC, cfg.dtype_C);
        printVerifyJson(v);
      }
    } else {
      iree_status_free(rs);
    }
    iree_vm_list_release(inputs);
    iree_vm_list_release(outputs);
  }

  std::cout << "}" << std::endl;

  // -- Cleanup -------------------------------------------------------------
  iree_hal_buffer_view_release(bufA);
  iree_hal_buffer_view_release(bufB);
  iree_hal_buffer_release(rawA);
  iree_hal_buffer_release(rawB);
  iree_vm_context_release(context);
  iree_hal_device_release(device);
  iree_hal_driver_release(driver);
  iree_vm_instance_release(instance);
  hipFree(dA);
  hipFree(dB);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);

  return 0;
}
