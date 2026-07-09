#pragma once

#include "llaisys.h"
#include "../utils.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace llaisys::ops::nvidia::detail {

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
