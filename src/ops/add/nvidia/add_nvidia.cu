#include "add_nvidia.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <type_traits>
#include <cstdio>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
            float fa = utils::cast_device<float>(a[idx]);
            float fb = utils::cast_device<float>(b[idx]);
            c[idx] = utils::cast_device<T>(fa + fb);
        } else {
            c[idx] = a[idx] + b[idx];
        }
    }
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<fp16_t *>(c),
            reinterpret_cast<const fp16_t *>(a),
            reinterpret_cast<const fp16_t *>(b),
            numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<bf16_t *>(c),
            reinterpret_cast<const bf16_t *>(a),
            reinterpret_cast<const bf16_t *>(b),
            numel);
        break;
    default:
        std::fprintf(stderr, "[ERROR] Unsupported data type for CUDA add: %d\n", type);
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] add kernel launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
