#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight, size_t N, size_t D) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t numel = N * D;
    if (out_idx < numel) {
        size_t row = out_idx / D;
        size_t col = out_idx % D;
        size_t weight_row = static_cast<size_t>(index[row]);
        out[out_idx] = weight[weight_row * D + col];
    }
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t weight_dtype,
               size_t N, size_t D) {
    const int block_size = 256;
    const size_t numel = N * D;
    const int num_blocks = static_cast<int>((numel + block_size - 1) / block_size);

    switch (weight_dtype) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const float *>(weight),
            N, D);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<fp16_t *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const fp16_t *>(weight),
            N, D);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<bf16_t *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const bf16_t *>(weight),
            N, D);
        break;
    default:
        std::fprintf(stderr, "[ERROR] Unsupported data type for CUDA embedding: %d\n", weight_dtype);
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] embedding kernel launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
