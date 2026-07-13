#include "nvidia_resource.cuh"
#include "cuda_utils.cuh"

#include <unordered_map>

namespace llaisys::device::nvidia {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id), _cublas_handle(nullptr) {
    CUDA_CHECK(cudaSetDevice(device_id));
    cublasStatus_t status = cublasCreate(&_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLAS handle");
    }
}

Resource::~Resource() {
    if (_cublas_handle != nullptr) {
        cublasDestroy(_cublas_handle);
        _cublas_handle = nullptr;
    }
}

cublasHandle_t Resource::cublasHandle() const {
    return _cublas_handle;
}

Resource &resource(int device_id) {
    static std::mutex resource_mutex;
    static std::unordered_map<int, std::unique_ptr<Resource>> resources;

    std::lock_guard<std::mutex> lock(resource_mutex);
    auto it = resources.find(device_id);
    if (it == resources.end()) {
        auto inserted = resources.emplace(device_id, std::make_unique<Resource>(device_id));
        it = inserted.first;
    }
    return *it->second;
}

} // namespace llaisys::device::nvidia
