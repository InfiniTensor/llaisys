#include "argmax_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <type_traits>

namespace llaisys::ops::nvidia {

struct argmax_pair_t {
    float value;
    int64_t index;
};

template <typename T>
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t size) {
    size_t tid = threadIdx.x;
    float local_max = -FLT_MAX;
    int64_t local_idx = 0;

    for (size_t i = tid; i < size; i += blockDim.x * gridDim.x) {
        float value;
        if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
            value = utils::cast_device<float>(vals[i]);
        } else {
            value = vals[i];
        }
        if (value > local_max) {
            local_max = value;
            local_idx = static_cast<int64_t>(i);
        }
    }

    extern __shared__ argmax_pair_t shared[];
    shared[tid] = {local_max, local_idx};
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && shared[tid + stride].value > shared[tid].value) {
            shared[tid] = shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_idx[0] = shared[0].index;
        max_val[0] = utils::cast_device<T>(shared[0].value);
    }
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t size) {
    const int block_size = 256;
    const int num_blocks = 1;
    const size_t shared_mem = block_size * sizeof(argmax_pair_t);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<<<num_blocks, block_size, shared_mem>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals),
            size);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel<<<num_blocks, block_size, shared_mem>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<fp16_t *>(max_val),
            reinterpret_cast<const fp16_t *>(vals),
            size);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel<<<num_blocks, block_size, shared_mem>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<bf16_t *>(max_val),
            reinterpret_cast<const bf16_t *>(vals),
            size);
        break;
    default:
        std::fprintf(stderr, "[ERROR] Unsupported data type for CUDA argmax: %d\n", type);
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] argmax kernel launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
