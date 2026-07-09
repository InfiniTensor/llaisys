#pragma once

#include "../../../include/llaisys.h"

#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>
#include <mcblas/mcblas.h>
#include <mcr/mc_runtime.h>

#include <sstream>
#include <stdexcept>

namespace llaisys::device::metax {

inline void check_metax(mcError_t status, const char *expr, const char *file, int line) {
    if (status != mcSuccess) {
        std::stringstream ss;
        ss << "MetaX Runtime 调用失败: " << expr << " at " << file << ":" << line
           << " -> " << mcGetErrorString(status);
        throw std::runtime_error(ss.str());
    }
}

inline void check_mcblas(mcblasStatus_t status, const char *expr, const char *file, int line) {
    if (status != MCBLAS_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "mcBLAS 调用失败: " << expr << " at " << file << ":" << line
           << " -> " << mcblasGetStatusString(status);
        throw std::runtime_error(ss.str());
    }
}

#define METAX_CHECK(EXPR) ::llaisys::device::metax::check_metax((EXPR), #EXPR, __FILE__, __LINE__)
#define MCBLAS_CHECK(EXPR) ::llaisys::device::metax::check_mcblas((EXPR), #EXPR, __FILE__, __LINE__)

inline mcMemcpyKind to_mc_memcpy_kind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2D:
        return mcMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return mcMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return mcMemcpyDeviceToDevice;
    case LLAISYS_MEMCPY_H2H:
    default:
        return mcMemcpyHostToHost;
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
__device__ __forceinline__ float to_float_device<maca_bfloat16>(maca_bfloat16 value) {
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
__device__ __forceinline__ maca_bfloat16 from_float_device<maca_bfloat16>(float value) {
    return __float2bfloat16(value);
}

inline macaDataType to_maca_dtype(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return MACA_R_32F;
    case LLAISYS_DTYPE_F16:
        return MACA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return MACA_R_16BF;
    default:
        throw std::runtime_error("不支持的 MetaX dtype");
    }
}

} // namespace llaisys::device::metax
