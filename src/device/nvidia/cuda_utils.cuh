#pragma once

#include "../../../include/llaisys.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <sstream>
#include <stdexcept>

namespace llaisys::device::nvidia {

inline void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
    if (status != cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA 调用失败: " << expr << " at " << file << ":" << line << " -> " << cudaGetErrorString(status);
        throw std::runtime_error(ss.str());
    }
}

inline void check_cublas(cublasStatus_t status, const char *expr, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "cuBLAS 调用失败: " << expr << " at " << file << ":" << line << " -> status=" << static_cast<int>(status);
        throw std::runtime_error(ss.str());
    }
}

#define CUDA_CHECK(EXPR) ::llaisys::device::nvidia::check_cuda((EXPR), #EXPR, __FILE__, __LINE__)
#define CUBLAS_CHECK(EXPR) ::llaisys::device::nvidia::check_cublas((EXPR), #EXPR, __FILE__, __LINE__)

inline cudaMemcpyKind to_cuda_memcpy_kind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    case LLAISYS_MEMCPY_H2H:
    default:
        return cudaMemcpyHostToHost;
    }
}

template <typename T>
__device__ __forceinline__ float to_float_device(T value);

template <>
__device__ __forceinline__ float to_float_device<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float to_float_device<half>(half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float_device<nv_bfloat16>(nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float_device(float value);

template <>
__device__ __forceinline__ float from_float_device<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ half from_float_device<half>(float value) {
    return __float2half(value);
}

template <>
__device__ __forceinline__ nv_bfloat16 from_float_device<nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

inline cudaDataType_t to_cuda_dtype(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;
    default:
        throw std::runtime_error("不支持的 CUDA dtype");
    }
}

} // namespace llaisys::device::nvidia
