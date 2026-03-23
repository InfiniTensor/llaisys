#include "add_nvidia.hpp"

#include "../../../device/nvidia/cuda_utils.hpp"

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float av = llaisys::device::nvidia::ScalarOps<T>::load(a + idx);
        float bv = llaisys::device::nvidia::ScalarOps<T>::load(b + idx);
        llaisys::device::nvidia::ScalarOps<T>::store(c + idx, av + bv);
    }
}

template <typename T>
void launch_add(T *c, const T *a, const T *b, size_t numel) {
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    add_kernel<<<blocks, threads>>>(c, a, b, numel);
    llaisys::device::nvidia::cuda_check(cudaGetLastError());
}
} // namespace

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_add(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a),
                          reinterpret_cast<const float *>(b), numel);
    case LLAISYS_DTYPE_BF16:
        return launch_add(reinterpret_cast<llaisys::bf16_t *>(c), reinterpret_cast<const llaisys::bf16_t *>(a),
                          reinterpret_cast<const llaisys::bf16_t *>(b), numel);
    case LLAISYS_DTYPE_F16:
        return launch_add(reinterpret_cast<llaisys::fp16_t *>(c), reinterpret_cast<const llaisys::fp16_t *>(a),
                          reinterpret_cast<const llaisys::fp16_t *>(b), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
