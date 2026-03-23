#include "swiglu_nvidia.hpp"

#include "../../../device/nvidia/cuda_utils.hpp"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    float g = llaisys::device::nvidia::ScalarOps<T>::load(gate + idx);
    float u = llaisys::device::nvidia::ScalarOps<T>::load(up + idx);
    float sigmoid = 1.0f / (1.0f + expf(-g));
    llaisys::device::nvidia::ScalarOps<T>::store(out + idx, u * g * sigmoid);
}

template <typename T>
void launch_swiglu(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel) {
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    swiglu_kernel<<<blocks, threads>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(gate),
                                       reinterpret_cast<const T *>(up), numel);
    llaisys::device::nvidia::cuda_check(cudaGetLastError());
}
} // namespace

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_swiglu<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return launch_swiglu<llaisys::bf16_t>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return launch_swiglu<llaisys::fp16_t>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
