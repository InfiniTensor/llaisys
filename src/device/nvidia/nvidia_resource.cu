#include "nvidia_resource.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <memory>
#include <unordered_map>

namespace llaisys::device::nvidia {

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(err));                   \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::fprintf(stderr, "[cuBLAS ERROR] %s:%d: status=%d\n", __FILE__, \
                         __LINE__, status);                                    \
        }                                                                      \
    } while (0)

Resource::Resource(int device_id)
    : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {}

Resource::~Resource() {
    if (_cublas_handle != nullptr) {
        CUBLAS_CHECK(cublasDestroy(_cublas_handle));
        _cublas_handle = nullptr;
    }
    _initialized = false;
}

void Resource::init() {
    if (_initialized) return;

    // Set device before creating handles
    CUDA_CHECK(cudaSetDevice(getDeviceId()));

    // Create cuBLAS handle
    CUBLAS_CHECK(cublasCreate(&_cublas_handle));

    _initialized = true;
}

cublasHandle_t Resource::cublasHandle() {
    if (!_initialized) {
        init();
    }
    return _cublas_handle;
}

namespace {
thread_local std::unordered_map<int, std::unique_ptr<Resource>> G_RESOURCES;
}

cublasHandle_t getCublasHandle(int device_id) {
    auto &resource = G_RESOURCES[device_id];
    if (resource == nullptr) {
        resource = std::make_unique<Resource>(device_id);
    }
    return resource->cublasHandle();
}

} // namespace llaisys::device::nvidia
