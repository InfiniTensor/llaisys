#include "linear_nvidia.hpp"

#include "../../nvidia_utils.cuh"
#include "../../../core/llaisys_core.hpp"

namespace llaisys::ops::nvidia {
template <typename T>
__global__ void linear_kernel(T *out, const T *in, const T *W, const T *bias, size_t batch, size_t in_dim, size_t out_dim) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = batch * out_dim;
    if (idx >= total) {
        return;
    }
    size_t i = idx / out_dim;
    size_t j = idx - i * out_dim;
    const T *in_row = in + i * in_dim;
    const T *w_row = W + j * in_dim;
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        float acc = 0.0f;
        for (size_t k = 0; k < in_dim; ++k) {
            acc += detail::to_float(in_row[k]) * detail::to_float(w_row[k]);
        }
        if (bias) {
            acc += detail::to_float(bias[j]);
        }
        out[idx] = detail::from_float<T>(acc);
    } else {
        T acc = static_cast<T>(0);
        for (size_t k = 0; k < in_dim; ++k) {
            acc += in_row[k] * w_row[k];
        }
        if (bias) {
            acc += bias[j];
        }
        out[idx] = acc;
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *W, const std::byte *bias, llaisysDataType_t type, size_t batch, size_t in_dim, size_t out_dim) {
    size_t total = batch * out_dim;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    switch (type) {
    case LLAISYS_DTYPE_F32:
        linear_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(W),
            reinterpret_cast<const float *>(bias),
            batch, in_dim, out_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(W),
            reinterpret_cast<const llaisys::bf16_t *>(bias),
            batch, in_dim, out_dim);
        break;
    case LLAISYS_DTYPE_F16:
        linear_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(W),
            reinterpret_cast<const llaisys::fp16_t *>(bias),
            batch, in_dim, out_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    detail::checkCuda(cudaGetLastError());
    detail::checkCuda(cudaStreamSynchronize(stream));
}
}
