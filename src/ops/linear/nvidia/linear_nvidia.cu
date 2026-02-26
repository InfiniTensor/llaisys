#include "linear_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void linearKernel(T *out, const T *in, const T *weight, const T *bias,
                             size_t batch_size, size_t in_dim, size_t out_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * out_dim;

    for (size_t linear_idx = idx; linear_idx < total; linear_idx += blockDim.x * gridDim.x) {
        size_t i = linear_idx / out_dim;
        size_t j = linear_idx % out_dim;

        float sum = 0.0f;
        for (size_t k = 0; k < in_dim; ++k) {
            float x = to_float_cuda(in[i * in_dim + k]);
            float w = to_float_cuda(weight[j * in_dim + k]);
            sum += x * w;
        }

        if (bias) {
            sum += to_float_cuda(bias[j]);
        }

        out[linear_idx] = from_float_cuda<T>(sum);
    }
}

template <typename T>
void linear_(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *weight_bytes,
             const std::byte *bias_bytes, size_t batch_size, size_t in_dim, size_t out_dim) {
    auto out = reinterpret_cast<T *>(out_bytes);
    auto in = reinterpret_cast<const T *>(in_bytes);
    auto weight = reinterpret_cast<const T *>(weight_bytes);
    auto bias = reinterpret_cast<const T *>(bias_bytes);

    size_t total_outputs = batch_size * out_dim;
    const int blockSize = 256;
    const int numBlocks = (total_outputs + blockSize - 1) / blockSize;

    linearKernel<T><<<numBlocks, blockSize>>>(out, in, weight, bias, batch_size, in_dim, out_dim);
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_dim, size_t out_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_<float>(out, in, weight, bias, batch_size, in_dim, out_dim);
    case LLAISYS_DTYPE_BF16:
        return linear_<bf16_t_cuda>(out, in, weight, bias, batch_size, in_dim, out_dim);
    case LLAISYS_DTYPE_F16:
        return linear_<fp16_t_cuda>(out, in, weight, bias, batch_size, in_dim, out_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
