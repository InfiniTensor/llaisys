#pragma once

#include "llaisys.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llaisys::ops::cuda {

__device__ __forceinline__ float load_as_float_f32(const void *ptr, size_t idx) {
    return reinterpret_cast<const float *>(ptr)[idx];
}

__device__ __forceinline__ float load_as_float_f16(const void *ptr, size_t idx) {
    return __half2float(reinterpret_cast<const __half *>(ptr)[idx]);
}

__device__ __forceinline__ float load_as_float_bf16(const void *ptr, size_t idx) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(ptr)[idx]);
}

__device__ __forceinline__ float load_as_float(const void *ptr, size_t idx, llaisysDataType_t dtype) {
    if (dtype == LLAISYS_DTYPE_F32) {
        return load_as_float_f32(ptr, idx);
    }
    if (dtype == LLAISYS_DTYPE_F16) {
        return load_as_float_f16(ptr, idx);
    }
    return load_as_float_bf16(ptr, idx);
}

__device__ __forceinline__ void store_from_float_f32(void *ptr, size_t idx, float val) {
    reinterpret_cast<float *>(ptr)[idx] = val;
}

__device__ __forceinline__ void store_from_float_f16(void *ptr, size_t idx, float val) {
    reinterpret_cast<__half *>(ptr)[idx] = __float2half_rn(val);
}

__device__ __forceinline__ void store_from_float_bf16(void *ptr, size_t idx, float val) {
    reinterpret_cast<__nv_bfloat16 *>(ptr)[idx] = __float2bfloat16(val);
}

__device__ __forceinline__ void store_from_float(void *ptr, size_t idx, float val, llaisysDataType_t dtype) {
    if (dtype == LLAISYS_DTYPE_F32) {
        store_from_float_f32(ptr, idx, val);
        return;
    }
    if (dtype == LLAISYS_DTYPE_F16) {
        store_from_float_f16(ptr, idx, val);
        return;
    }
    store_from_float_bf16(ptr, idx, val);
}

} // namespace llaisys::ops::cuda
