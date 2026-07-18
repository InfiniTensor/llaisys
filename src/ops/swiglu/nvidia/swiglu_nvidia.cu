#include "swiglu_nvidia.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <type_traits>
#include <cstdio>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
            float g = utils::cast_device<float>(gate[idx]);
            float u = utils::cast_device<float>(up[idx]);
            float silu_gate = g / (1.0f + expf(-g));
            out[idx] = utils::cast_device<T>(u * silu_gate);
        } else {
            T g = gate[idx];
            T u = up[idx];
            T silu_gate = g / (static_cast<T>(1) + expf(-g));
            out[idx] = u * silu_gate;
        }
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<fp16_t *>(out),
            reinterpret_cast<const fp16_t *>(gate),
            reinterpret_cast<const fp16_t *>(up),
            numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<bf16_t *>(out),
            reinterpret_cast<const bf16_t *>(gate),
            reinterpret_cast<const bf16_t *>(up),
            numel);
        break;
    default:
        std::fprintf(stderr, "[ERROR] Unsupported data type for CUDA swiglu: %d\n", type);
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] swiglu kernel launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
