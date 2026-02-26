#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Define the custom types directly in CUDA code
// These must match the definitions in utils/types.hpp
struct CustomFloat16 {
    uint16_t _v;
};
typedef struct CustomFloat16 fp16_t_cuda;

struct CustomBFloat16 {
    uint16_t _v;
};
typedef struct CustomBFloat16 bf16_t_cuda;

namespace llaisys::ops::iluvatar {

// Device functions for type conversion from custom types to float
__device__ inline float to_float_cuda(float val) { return val; }

__device__ inline float to_float_cuda(fp16_t_cuda val) {
    __half h = __ushort_as_half(val._v);
    return __half2float(h);
}

__device__ inline float to_float_cuda(bf16_t_cuda val) {
    uint32_t u = static_cast<uint32_t>(val._v) << 16;
    return __uint_as_float(u);
}

// Device functions for type conversion from float to custom types
__device__ inline float from_float_cuda(float val, float*) { return val; }

__device__ inline fp16_t_cuda from_float_cuda(float val, fp16_t_cuda*) {
    __half h = __float2half(val);
    fp16_t_cuda result;
    result._v = __half_as_ushort(h);
    return result;
}

__device__ inline bf16_t_cuda from_float_cuda(float val, bf16_t_cuda*) {
    uint32_t u = __float_as_uint(val);
    bf16_t_cuda result;
    result._v = static_cast<uint16_t>(u >> 16);
    return result;
}

// Template helper for converting to any type from float
template<typename T>
__device__ inline T from_float_cuda(float val) {
    return from_float_cuda(val, static_cast<T*>(nullptr));
}

} // namespace llaisys::ops::iluvatar
