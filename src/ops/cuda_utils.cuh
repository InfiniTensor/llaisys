#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include "llaisys.h"

#define CUDA_KERNEL_CHECK()                                                       \
    do {                                                                          \
        cudaError_t err = cudaGetLastError();                                     \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "[CUDA KERNEL ERROR] %s at %s:%d\n",                  \
                    cudaGetErrorString(err), __FILE__, __LINE__);                  \
            throw std::runtime_error(cudaGetErrorString(err));                     \
        }                                                                         \
    } while (0)

__device__ __forceinline__ float bf16_to_f32(uint16_t v) {
    __nv_bfloat16 bf;
    memcpy(&bf, &v, sizeof(uint16_t));
    return __bfloat162float(bf);
}

__device__ __forceinline__ uint16_t f32_to_bf16(float v) {
    __nv_bfloat16 bf = __float2bfloat16(v);
    uint16_t r;
    memcpy(&r, &bf, sizeof(uint16_t));
    return r;
}

__device__ __forceinline__ float fp16_to_f32(uint16_t v) {
    __half h;
    memcpy(&h, &v, sizeof(uint16_t));
    return __half2float(h);
}

__device__ __forceinline__ uint16_t f32_to_fp16(float v) {
    __half h = __float2half(v);
    uint16_t r;
    memcpy(&r, &h, sizeof(uint16_t));
    return r;
}

__device__ __forceinline__ float load_as_f32(const void *ptr, size_t idx, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return reinterpret_cast<const float *>(ptr)[idx];
    case LLAISYS_DTYPE_BF16:
        return bf16_to_f32(reinterpret_cast<const uint16_t *>(ptr)[idx]);
    case LLAISYS_DTYPE_F16:
        return fp16_to_f32(reinterpret_cast<const uint16_t *>(ptr)[idx]);
    default:
        return 0.0f;
    }
}

__device__ __forceinline__ void store_from_f32(void *ptr, size_t idx, float val, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        reinterpret_cast<float *>(ptr)[idx] = val;
        break;
    case LLAISYS_DTYPE_BF16:
        reinterpret_cast<uint16_t *>(ptr)[idx] = f32_to_bf16(val);
        break;
    case LLAISYS_DTYPE_F16:
        reinterpret_cast<uint16_t *>(ptr)[idx] = f32_to_fp16(val);
        break;
    default:
        break;
    }
}

inline size_t cuda_dsize(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return 4;
    case LLAISYS_DTYPE_BF16: return 2;
    case LLAISYS_DTYPE_F16: return 2;
    case LLAISYS_DTYPE_I64: return 8;
    default: return 0;
    }
}

constexpr int CUDA_BLOCK_SIZE = 256;

inline int cuda_grid_size(size_t n, int block_size = CUDA_BLOCK_SIZE) {
    return static_cast<int>((n + block_size - 1) / block_size);
}
