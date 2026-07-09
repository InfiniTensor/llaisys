#include "nvidia_resource.cuh"
#include "cuda_utils.cuh"

#include <unordered_map>

namespace llaisys::device::nvidia {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {}
Resource::~Resource() = default;

cublasHandle_t get_cublas_handle(int device_id, void *stream) {
    thread_local std::unordered_map<int, cublasHandle_t> handles;
    auto iter = handles.find(device_id);
    if (iter == handles.end()) {
        CUDA_CHECK(cudaSetDevice(device_id));
        cublasHandle_t handle = nullptr;
        CUBLAS_CHECK(cublasCreate(&handle));
        iter = handles.emplace(device_id, handle).first;
    }
    CUBLAS_CHECK(cublasSetStream(iter->second, reinterpret_cast<cudaStream_t>(stream)));
    return iter->second;
}

} // namespace llaisys::device::nvidia
