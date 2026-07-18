#include "rms_norm_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <type_traits>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, float eps, size_t D) {
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;

    extern __shared__ float shared[];
    float local_sum = 0.0f;

    for (size_t col = tid; col < D; col += blockDim.x) {
        float value = utils::cast_device<float>(in[row * D + col]);
        local_sum += value * value;
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(shared[0] / static_cast<float>(D) + eps);
    for (size_t col = tid; col < D; col += blockDim.x) {
        if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
            float x = utils::cast_device<float>(in[row * D + col]);
            float w = utils::cast_device<float>(weight[col]);
            out[row * D + col] = utils::cast_device<T>(x * inv_rms * w);
        } else {
            out[row * D + col] = (in[row * D + col] * inv_rms) * weight[col];
        }
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              llaisysDataType_t dtype, size_t N, size_t D) {
    const int block_size = 256;
    const int num_blocks = static_cast<int>(N);
    const size_t shared_mem = block_size * sizeof(float);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<<<num_blocks, block_size, shared_mem>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            eps, D);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<<<num_blocks, block_size, shared_mem>>>(
            reinterpret_cast<fp16_t *>(out),
            reinterpret_cast<const fp16_t *>(in),
            reinterpret_cast<const fp16_t *>(weight),
            eps, D);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<<<num_blocks, block_size, shared_mem>>>(
            reinterpret_cast<bf16_t *>(out),
            reinterpret_cast<const bf16_t *>(in),
            reinterpret_cast<const bf16_t *>(weight),
            eps, D);
        break;
    default:
        std::fprintf(stderr, "[ERROR] Unsupported data type for CUDA rms_norm: %d\n", dtype);
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] rms_norm kernel launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
