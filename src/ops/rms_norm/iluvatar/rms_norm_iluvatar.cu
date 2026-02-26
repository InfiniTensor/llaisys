#include "rms_norm_iluvatar.cuh"

#include "../../../device/iluvatar/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace llaisys::ops::iluvatar {

template <typename T>
__global__ void rmsNormKernel(T *out, const T *in, const T *weight, float eps,
                              size_t num_rows, size_t row_size) {
    size_t row = blockIdx.x;
    if (row >= num_rows) return;
    
    // Compute RMS for this row
    float sum_squares = 0.0f;
    for (size_t i = threadIdx.x; i < row_size; i += blockDim.x) {
        float val = to_float_cuda(in[row * row_size + i]);
        sum_squares += val * val;
    }
    
    // Reduction within block
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum_squares;
    __syncthreads();
    
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float rms = sqrtf(shared_sum[0] / row_size + eps);
    
    // Normalize and scale
    for (size_t i = threadIdx.x; i < row_size; i += blockDim.x) {
        float x = to_float_cuda(in[row * row_size + i]);
        float w = to_float_cuda(weight[i]);
        out[row * row_size + i] = from_float_cuda<T>((x / rms) * w);
    }
}

template <typename T>
void rms_norm_(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *weight_bytes, 
               float eps, size_t num_rows, size_t row_size) {
    auto out = reinterpret_cast<T *>(out_bytes);
    auto in = reinterpret_cast<const T *>(in_bytes);
    auto weight = reinterpret_cast<const T *>(weight_bytes);
    
    const int blockSize = 256;
    
    rmsNormKernel<T><<<num_rows, blockSize>>>(out, in, weight, eps, num_rows, row_size);
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              llaisysDataType_t type, size_t num_rows, size_t row_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_<float>(out, in, weight, eps, num_rows, row_size);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_<bf16_t_cuda>(out, in, weight, eps, num_rows, row_size);
    case LLAISYS_DTYPE_F16:
        return rms_norm_<fp16_t_cuda>(out, in, weight, eps, num_rows, row_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
