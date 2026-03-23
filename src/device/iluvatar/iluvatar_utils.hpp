#pragma once

#include "../../utils/types.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace llaisys::device::iluvatar {
inline void cuda_check(cudaError_t err) {
    if (err == cudaSuccess) {
        return;
    }
    if (err == cudaErrorCudartUnloading || err == cudaErrorContextIsDestroyed) {
        return;
    }
    throw std::runtime_error(cudaGetErrorString(err));
}

template <typename T>
struct ScalarOps;

template <>
struct ScalarOps<float> {
    __device__ static inline float load(const float *ptr) {
        return *ptr;
    }
    __device__ static inline void store(float *ptr, float v) {
        *ptr = v;
    }
};

template <>
struct ScalarOps<llaisys::fp16_t> {
    __device__ static inline float load(const llaisys::fp16_t *ptr) {
        return __half2float(*reinterpret_cast<const __half *>(ptr));
    }
    __device__ static inline void store(llaisys::fp16_t *ptr, float v) {
        *reinterpret_cast<__half *>(ptr) = __float2half(v);
    }
};

template <>
struct ScalarOps<llaisys::bf16_t> {
    __device__ static inline float load(const llaisys::bf16_t *ptr) {
        return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(ptr));
    }
    __device__ static inline void store(llaisys::bf16_t *ptr, float v) {
        *reinterpret_cast<__nv_bfloat16 *>(ptr) = __float2bfloat16(v);
    }
};
} // namespace llaisys::device::iluvatar
