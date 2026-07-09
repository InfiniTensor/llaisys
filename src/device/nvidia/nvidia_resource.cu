#include "nvidia_resource.cuh"

#include <cuda_runtime.h>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace llaisys::device::nvidia {

// 按设备 ID 存储 cuBLAS handles
static std::unordered_map<int, cublasHandle_t> g_cublas_handles;
static std::mutex g_cublas_mutex;

cublasHandle_t getCublasHandle() {
    int device_id = 0;
    cudaGetDevice(&device_id);

    std::lock_guard<std::mutex> lock(g_cublas_mutex);
    auto it = g_cublas_handles.find(device_id);
    if (it != g_cublas_handles.end()) {
        return it->second;
    }

    // 初始化该设备的 cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("getCublasHandle: cublasCreate failed");
    }
    // A100 SM80 开启 TF32 数学模式，对 f32 GEMM 自动使用 Tensor Core
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    g_cublas_handles[device_id] = handle;
    return handle;
}

Resource::Resource(int device_id)
    : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {
    // 切换到对应设备并确保 cuBLAS handle 已初始化
    cudaSetDevice(device_id);
    getCublasHandle();
}

Resource::~Resource() {
    // handle 会在进程退出时由 CUDA driver 清理，这里不主动销毁以避免重入问题
}

} // namespace llaisys::device::nvidia
