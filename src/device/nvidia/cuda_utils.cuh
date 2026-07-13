#pragma once

#include "../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <iostream>
#include <stdexcept>

namespace llaisys::device::nvidia {

#define CUDA_CHECK(EXPR__)                                                                    \
    do {                                                                                      \
        cudaError_t err__ = (EXPR__);                                                         \
        if (err__ != cudaSuccess) {                                                           \
            std::cerr << "[ERROR] CUDA call failed: " << cudaGetErrorString(err__) << " ("   \
                      << static_cast<int>(err__) << ")" << EXCEPTION_LOCATION_MSG << std::endl; \
            throw std::runtime_error(cudaGetErrorString(err__));                              \
        }                                                                                     \
    } while (0)

#define NCCL_CHECK(EXPR__)                                                                        \
    do {                                                                                           \
        ncclResult_t err__ = (EXPR__);                                                             \
        if (err__ != ncclSuccess) {                                                                \
            std::cerr << "[ERROR] NCCL call failed: " << ncclGetErrorString(err__) << " ("        \
                      << static_cast<int>(err__) << ")" << EXCEPTION_LOCATION_MSG << std::endl;    \
            throw std::runtime_error(ncclGetErrorString(err__));                                   \
        }                                                                                          \
    } while (0)

inline cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
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
        CHECK_ARGUMENT(false, "invalid memcpy kind");
        return cudaMemcpyDefault;
    }
}

inline cudaStream_t toCudaStream(llaisysStream_t stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

template <typename T>
__device__ inline float scalarToFloat(T value);

template <>
__device__ inline float scalarToFloat<float>(float value) {
    return value;
}

template <>
__device__ inline float scalarToFloat<llaisys::fp16_t>(llaisys::fp16_t value) {
    union {
        uint16_t u;
        __half h;
    } caster{value._v};
    return __half2float(caster.h);
}

template <>
__device__ inline float scalarToFloat<llaisys::bf16_t>(llaisys::bf16_t value) {
    union {
        uint16_t u;
        __nv_bfloat16 b;
    } caster{value._v};
    return __bfloat162float(caster.b);
}

template <typename T>
__device__ inline T floatToScalar(float value);

template <>
__device__ inline float floatToScalar<float>(float value) {
    return value;
}

template <>
__device__ inline llaisys::fp16_t floatToScalar<llaisys::fp16_t>(float value) {
    union {
        uint16_t u;
        __half h;
    } caster;
    caster.h = __float2half_rn(value);
    return llaisys::fp16_t{caster.u};
}

template <>
__device__ inline llaisys::bf16_t floatToScalar<llaisys::bf16_t>(float value) {
    union {
        uint16_t u;
        __nv_bfloat16 b;
    } caster;
    caster.b = __float2bfloat16(value);
    return llaisys::bf16_t{caster.u};
}

} // namespace llaisys::device::nvidia
