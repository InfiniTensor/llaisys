#include "linear_nvidia.hpp"

#include "../../../device/nvidia/cuda_utils.hpp"

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void linear_kernel(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t n, size_t k) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = m * n;
    if (idx >= total) {
        return;
    }
    size_t row = idx / n;
    size_t col = idx % n;
    float acc = bias ? llaisys::device::nvidia::ScalarOps<T>::load(bias + col) : 0.f;
    const T *w_row = weight + col * k;
    const T *in_row = in + row * k;
    for (size_t j = 0; j < k; ++j) {
        float a = llaisys::device::nvidia::ScalarOps<T>::load(in_row + j);
        float b = llaisys::device::nvidia::ScalarOps<T>::load(w_row + j);
        acc += a * b;
    }
    llaisys::device::nvidia::ScalarOps<T>::store(out + idx, acc);
}

template <typename T>
void launch_linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t m,
                   size_t n, size_t k) {
    const int threads = 256;
    const size_t total = m * n;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    linear_kernel<<<blocks, threads>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
                                       reinterpret_cast<const T *>(weight),
                                       bias ? reinterpret_cast<const T *>(bias) : nullptr, m, n, k);
    llaisys::device::nvidia::cuda_check(cudaGetLastError());
}
} // namespace

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t m, size_t n, size_t k) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_linear<float>(out, in, weight, bias, m, n, k);
    case LLAISYS_DTYPE_BF16:
        return launch_linear<llaisys::bf16_t>(out, in, weight, bias, m, n, k);
    case LLAISYS_DTYPE_F16:
        return launch_linear<llaisys::fp16_t>(out, in, weight, bias, m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
