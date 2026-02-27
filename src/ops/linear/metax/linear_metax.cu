#include "linear_metax.cuh"

#include "../../../device/metax/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace llaisys::ops::metax {

// Improved linear kernel with more accurate accumulation
// Uses Kahan summation for better precision
template <typename T>
__global__ void linearKernel(T *out, const T *in, const T *weight, const T *bias,
                             size_t batch_size, size_t in_dim, size_t out_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * out_dim;

    for (size_t linear_idx = idx; linear_idx < total; linear_idx += blockDim.x * gridDim.x) {
        size_t i = linear_idx / out_dim;
        size_t j = linear_idx % out_dim;

        // Kahan summation for better precision
        float sum = 0.0f;
        float c = 0.0f;  // Compensation for lost low-order bits
        
        for (size_t k = 0; k < in_dim; ++k) {
            float x = to_float_metax(in[i * in_dim + k]);
            float w = to_float_metax(weight[j * in_dim + k]);
            float y = x * w - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }

        if (bias) {
            sum += to_float_metax(bias[j]);
        }

        out[linear_idx] = from_float_metax<T>(sum);
    }
}

// Alternative: blocked accumulation for very large dimensions
template <typename T>
__global__ void linearKernelBlocked(T *out, const T *in, const T *weight, const T *bias,
                                    size_t batch_size, size_t in_dim, size_t out_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * out_dim;
    
    const size_t BLOCK_SIZE = 256;  // Process in blocks for better precision

    for (size_t linear_idx = idx; linear_idx < total; linear_idx += blockDim.x * gridDim.x) {
        size_t i = linear_idx / out_dim;
        size_t j = linear_idx % out_dim;

        float sum = 0.0f;
        
        // Process in blocks
        for (size_t k_start = 0; k_start < in_dim; k_start += BLOCK_SIZE) {
            float block_sum = 0.0f;
            size_t k_end = min(k_start + BLOCK_SIZE, in_dim);
            
            for (size_t k = k_start; k < k_end; ++k) {
                float x = to_float_metax(in[i * in_dim + k]);
                float w = to_float_metax(weight[j * in_dim + k]);
                block_sum += x * w;
            }
            sum += block_sum;
        }

        if (bias) {
            sum += to_float_metax(bias[j]);
        }

        out[linear_idx] = from_float_metax<T>(sum);
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
    
    // Use blocked kernel for large dimensions (common in transformers)
    if (in_dim > 512) {
        linearKernelBlocked<T><<<numBlocks, blockSize>>>(out, in, weight, bias, batch_size, in_dim, out_dim);
    } else {
        linearKernel<T><<<numBlocks, blockSize>>>(out, in, weight, bias, batch_size, in_dim, out_dim);
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_dim, size_t out_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_<float>(out, in, weight, bias, batch_size, in_dim, out_dim);
    case LLAISYS_DTYPE_BF16:
        return linear_<bf16_t_metax>(out, in, weight, bias, batch_size, in_dim, out_dim);
    case LLAISYS_DTYPE_F16:
        return linear_<fp16_t_metax>(out, in, weight, bias, batch_size, in_dim, out_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
