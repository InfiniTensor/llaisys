#pragma once

#include "../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace llaisys::device::nvidia::cuda_utils {

inline void checkCuda(cudaError_t status, const char *expr, const char *file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    std::cerr << "[ERROR] CUDA call failed: " << expr << " -> " << cudaGetErrorString(status)
              << " at " << file << ":" << line << std::endl;
    throw std::runtime_error("CUDA error");
}

inline const char *cublasStatusName(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    default:
        return "CUBLAS_STATUS_UNKNOWN";
    }
}

inline void checkCublas(cublasStatus_t status, const char *expr, const char *file, int line) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }
    std::cerr << "[ERROR] cuBLAS call failed: " << expr << " -> " << cublasStatusName(status)
              << " at " << file << ":" << line << std::endl;
    throw std::runtime_error("cuBLAS error");
}

#define LLAISYS_CUDA_CHECK(EXPR__) ::llaisys::device::nvidia::cuda_utils::checkCuda((EXPR__), #EXPR__, __FILE__, __LINE__)
#define LLAISYS_CUBLAS_CHECK(EXPR__) ::llaisys::device::nvidia::cuda_utils::checkCublas((EXPR__), #EXPR__, __FILE__, __LINE__)

inline cudaMemcpyKind memcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("Unsupported memcpy kind");
    }
}

inline cudaDataType_t cublasDataType(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

inline cublasComputeType_t cublasComputeType(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        return CUBLAS_COMPUTE_32F;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float toFloat<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float toFloat<llaisys::fp16_t>(llaisys::fp16_t value) {
    union {
        __half half_value;
        __half_raw raw;
    } bits{};
    bits.raw.x = value._v;
    return __half2float(bits.half_value);
}

template <>
__device__ __forceinline__ float toFloat<llaisys::bf16_t>(llaisys::bf16_t value) {
    union {
        __nv_bfloat16 bf16_value;
        __nv_bfloat16_raw raw;
    } bits{};
    bits.raw.x = value._v;
    return __bfloat162float(bits.bf16_value);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ float fromFloat<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ llaisys::fp16_t fromFloat<llaisys::fp16_t>(float value) {
    union {
        __half half_value;
        __half_raw raw;
    } bits{};
    bits.half_value = __float2half_rn(value);
    return llaisys::fp16_t{bits.raw.x};
}

template <>
__device__ __forceinline__ llaisys::bf16_t fromFloat<llaisys::bf16_t>(float value) {
    union {
        __nv_bfloat16 bf16_value;
        __nv_bfloat16_raw raw;
    } bits{};
    bits.bf16_value = __float2bfloat16(value);
    return llaisys::bf16_t{bits.raw.x};
}

} // namespace llaisys::device::nvidia::cuda_utils
