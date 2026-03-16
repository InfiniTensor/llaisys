#pragma once

#include "../core/llaisys_core.hpp"
#include "../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace llaisys::ops::nvidia {

inline void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA API failed");
    }
}

inline cudaStream_t current_stream() {
    return reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
}

template <typename T>
__device__ inline float to_float(T v);

template <>
__device__ inline float to_float<float>(float v) {
    return v;
}

template <>
__device__ inline float to_float<llaisys::fp16_t>(llaisys::fp16_t v) {
    return __half2float(__ushort_as_half(v._v));
}

template <>
__device__ inline float to_float<llaisys::bf16_t>(llaisys::bf16_t v) {
    const uint32_t bits = static_cast<uint32_t>(v._v) << 16;
    return __uint_as_float(bits);
}

template <typename T>
__device__ inline T from_float(float v);

template <>
__device__ inline float from_float<float>(float v) {
    return v;
}

template <>
__device__ inline llaisys::fp16_t from_float<llaisys::fp16_t>(float v) {
    llaisys::fp16_t out;
    out._v = __half_as_ushort(__float2half_rn(v));
    return out;
}

template <>
__device__ inline llaisys::bf16_t from_float<llaisys::bf16_t>(float v) {
    llaisys::bf16_t out;
    const uint32_t bits = __float_as_uint(v);
    const uint32_t rounding_bias = 0x00007FFFu + ((bits >> 16) & 1u);
    out._v = static_cast<uint16_t>((bits + rounding_bias) >> 16);
    return out;
}

inline int num_threads_1d() {
    return 256;
}

inline int num_blocks_1d(size_t n, int threads = 256) {
    return static_cast<int>((n + threads - 1) / threads);
}

} // namespace llaisys::ops::nvidia
