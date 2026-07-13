#include "nvidia_resource.cuh"
#include <stdexcept>
#include <cuda_runtime.h>
#include <cstdio>

namespace llaisys::device::nvidia {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {
    cudaError_t err=cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device in Resource constructor");
    }

    // Create cuBLAS handle
    cublasStatus_t status=cublasCreate(&_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLAS handle in Resource constructor");
    }


}
Resource::~Resource() {
    // Destroy cuBLAS handle
    if (_cublas_handle) {
        cublasDestroy(_cublas_handle);
        _cublas_handle = nullptr;
    }
}
} // namespace llaisys::device::nvidia
