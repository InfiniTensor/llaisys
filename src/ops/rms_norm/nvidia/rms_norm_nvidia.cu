#include "rms_norm_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void rmsNormKernel(T *out, const T *in, const T *weight, float eps, size_t hidden_dim) {
    // Each block processes one row
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;

    // Pointer to the start of this row
    const T *in_row = in + row * hidden_dim;
    T *out_row = out + row * hidden_dim;

    // Compute sum of squares using shared memory
    extern __shared__ float shared_sum[];

    float local_sum = 0.0f;
    for (size_t i = tid; i < hidden_dim; i += blockDim.x) {
        float val = to_float_cuda(in_row[i]);
        local_sum += val * val;
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    // Compute RMS
    float rms = sqrtf(shared_sum[0] / hidden_dim + eps);

    // Normalize and apply weight
    for (size_t i = tid; i < hidden_dim; i += blockDim.x) {
        float val = to_float_cuda(in_row[i]);
        float w = to_float_cuda(weight[i]);
        float result = (val / rms) * w;
        out_row[i] = from_float_cuda<T>(result);
    }
}

template <typename T>
void rms_norm_(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *weight_bytes,
               float eps, size_t batch_size, size_t hidden_dim) {
    auto out = reinterpret_cast<T *>(out_bytes);
    auto in = reinterpret_cast<const T *>(in_bytes);
    auto weight = reinterpret_cast<const T *>(weight_bytes);

    const int blockSize = 256;
    size_t sharedMemSize = blockSize * sizeof(float);

    rmsNormKernel<T><<<batch_size, blockSize, sharedMemSize>>>(out, in, weight, eps, hidden_dim);
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, float eps, size_t batch_size, size_t hidden_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_<float>(out, in, weight, eps, batch_size, hidden_dim);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_<bf16_t_cuda>(out, in, weight, eps, batch_size, hidden_dim);
    case LLAISYS_DTYPE_F16:
        return rms_norm_<fp16_t_cuda>(out, in, weight, eps, batch_size, hidden_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
