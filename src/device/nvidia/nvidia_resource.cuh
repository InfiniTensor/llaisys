#pragma once

#include "../device_resource.hpp"

// Forward declarations for CUDA types
typedef struct cublasContext *cublasHandle_t;

namespace llaisys::device::nvidia {

// NVIDIA-specific device resources
// Each device (GPU) has its own set of handles for library calls
class Resource : public llaisys::device::DeviceResource {
private:
    cublasHandle_t _cublas_handle = nullptr;
    bool _initialized = false;

public:
    Resource(int device_id);
    ~Resource();

    // Initialize resources (lazy initialization)
    void init();

    // Get cuBLAS handle ( initializes if needed)
    cublasHandle_t cublasHandle();

    // Prevent copying
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
};

cublasHandle_t getCublasHandle(int device_id);

} // namespace llaisys::device::nvidia
