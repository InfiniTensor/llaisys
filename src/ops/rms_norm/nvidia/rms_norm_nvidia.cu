#include "rms_norm_nvidia.hpp"

#include "../../../device/nvidia/cuda_utils.hpp"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const T *row_in = in + row * cols;
    T *row_out = out + row * cols;
    float sum_sq = 0.f;
    for (size_t j = 0; j < cols; ++j) {
        float v = llaisys::device::nvidia::ScalarOps<T>::load(row_in + j);
        sum_sq += v * v;
    }
    float mean = sum_sq / static_cast<float>(cols);
    float inv_rms = rsqrtf(mean + eps);
    for (size_t j = 0; j < cols; ++j) {
        float v = llaisys::device::nvidia::ScalarOps<T>::load(row_in + j);
        float w = llaisys::device::nvidia::ScalarOps<T>::load(weight + j);
        llaisys::device::nvidia::ScalarOps<T>::store(row_out + j, v * inv_rms * w);
    }
}

template <typename T>
void launch_rms(std::byte *out, const std::byte *in, const std::byte *weight, size_t rows, size_t cols, float eps) {
    const int threads = 256;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    rms_norm_kernel<<<blocks, threads>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
                                         reinterpret_cast<const T *>(weight), rows, cols, eps);
    llaisys::device::nvidia::cuda_check(cudaGetLastError());
}
} // namespace

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type,
              size_t rows, size_t cols, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_rms<float>(out, in, weight, rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return launch_rms<llaisys::bf16_t>(out, in, weight, rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return launch_rms<llaisys::fp16_t>(out, in, weight, rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
