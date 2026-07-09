#include "rope_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace llaisys::ops::nvidia {
namespace detail {
inline void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}

__device__ inline float f16_to_f32(llaisys::fp16_t val) {
    uint16_t h = val._v;
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    uint32_t f32;
    if (exponent == 31) {
        f32 = mantissa != 0 ? (sign | 0x7F800000 | (mantissa << 13)) : (sign | 0x7F800000);
    } else if (exponent == 0) {
        if (mantissa == 0) {
            f32 = sign;
        } else {
            exponent = -14;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f32 = sign | ((exponent + 127) << 23) | (mantissa << 13);
        }
    } else {
        f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    union {
        uint32_t u;
        float f;
    } tmp;
    tmp.u = f32;
    return tmp.f;
}

__device__ inline llaisys::fp16_t f32_to_f16(float val) {
    union {
        uint32_t u;
        float f;
    } tmp;
    tmp.f = val;
    uint32_t f32 = tmp.u;
    uint16_t sign = (f32 >> 16) & 0x8000;
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127;
    uint32_t mantissa = f32 & 0x7FFFFF;
    if (exponent >= 16) {
        if (exponent == 128 && mantissa != 0) {
            return llaisys::fp16_t{static_cast<uint16_t>(sign | 0x7E00)};
        }
        return llaisys::fp16_t{static_cast<uint16_t>(sign | 0x7C00)};
    } else if (exponent >= -14) {
        return llaisys::fp16_t{static_cast<uint16_t>(sign | ((exponent + 15) << 10) | (mantissa >> 13))};
    } else if (exponent >= -24) {
        mantissa |= 0x800000;
        mantissa >>= (-14 - exponent);
        return llaisys::fp16_t{static_cast<uint16_t>(sign | (mantissa >> 13))};
    }
    return llaisys::fp16_t{static_cast<uint16_t>(sign)};
}

__device__ inline float bf16_to_f32(llaisys::bf16_t val) {
    uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;
    union {
        uint32_t u;
        float f;
    } tmp;
    tmp.u = bits32;
    return tmp.f;
}

__device__ inline llaisys::bf16_t f32_to_bf16(float val) {
    union {
        uint32_t u;
        float f;
    } tmp;
    tmp.f = val;
    uint32_t bits32 = tmp.u;
    const uint32_t rounding_bias = 0x00007FFF + ((bits32 >> 16) & 1);
    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);
    return llaisys::bf16_t{bf16_bits};
}

template <typename T>
__device__ inline float to_float(T v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        return f16_to_f32(v);
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        return bf16_to_f32(v);
    } else {
        return static_cast<float>(v);
    }
}

template <typename T>
__device__ inline T from_float(float v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        return f32_to_f16(v);
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        return f32_to_bf16(v);
    } else {
        return static_cast<T>(v);
    }
}
}

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seq_len, size_t n_heads,
                            size_t head_dim) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t half = head_dim / 2;
    size_t total = seq_len * n_heads * half;
    if (idx >= total) {
        return;
    }
    size_t tmp = idx;
    size_t j = tmp % half;
    tmp /= half;
    size_t h = tmp % n_heads;
    size_t s = tmp / n_heads;
    int64_t p = pos_ids[s];
    float angle = static_cast<float>(p) / powf(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
    float cosine = cosf(angle);
    float sine = sinf(angle);
    size_t base = (s * n_heads + h) * head_dim;
    T a = in[base + j];
    T b = in[base + j + half];
    float af = detail::to_float(a);
    float bf = detail::to_float(b);
    out[base + j] = detail::from_float<T>(af * cosine - bf * sine);
    out[base + j + half] = detail::from_float<T>(af * sine + bf * cosine);
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t type, size_t seq_len,
          size_t n_heads, size_t head_dim) {
    size_t half = head_dim / 2;
    size_t total = seq_len * n_heads * half;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            theta, seq_len, n_heads, head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            theta, seq_len, n_heads, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            theta, seq_len, n_heads, head_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    detail::checkCuda(cudaGetLastError());
    detail::checkCuda(cudaStreamSynchronize(stream));
}
}
