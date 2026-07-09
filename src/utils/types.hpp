#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace llaisys {
struct CustomFloat16 {
    uint16_t _v;
};
typedef struct CustomFloat16 fp16_t;

struct CustomBFloat16 {
    uint16_t _v;
};
typedef struct CustomBFloat16 bf16_t;

struct CustomFloat8 {
    uint8_t _v;
};
typedef struct CustomFloat8 f8_t;

struct CustomComplex16 {
    fp16_t re;
    fp16_t im;
};
typedef struct CustomComplex16 cp16_t;

struct CustomComplex32 {
    fp16_t re;
    fp16_t im;
};
typedef struct CustomComplex32 cp32_t;

struct CustomComplex64 {
    float re;
    float im;
};
typedef struct CustomComplex64 cp64_t;

struct CustomComplex128 {
    double re;
    double im;
};
typedef struct CustomComplex128 cp128_t;

namespace utils {

size_t dsize(llaisysDataType_t dtype);
const char *dtype_to_str(llaisysDataType_t dtype);

float _f16_to_f32(fp16_t val);
fp16_t _f32_to_f16(float val);

float _bf16_to_f32(bf16_t val);
bf16_t _f32_to_bf16(float val);

// Vectorized conversions (AVX2 + F16C) for bulk casting
void fp16_to_fp32_vec(const uint16_t* src, float* dst, size_t n);
void bf16_to_fp32_vec(const uint16_t* src, float* dst, size_t n);
void fp32_to_fp16_vec(const float* src, uint16_t* dst, size_t n);
void fp32_to_bf16_vec(const float* src, uint16_t* dst, size_t n);

template <typename TypeTo, typename TypeFrom>
TypeTo cast(TypeFrom val) {
    if constexpr (std::is_same<TypeTo, TypeFrom>::value) {
        return val;
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && std::is_same<TypeFrom, float>::value) {
        return _f32_to_f16(val);
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return _f32_to_f16(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && std::is_same<TypeTo, float>::value) {
        return _f16_to_f32(val);
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(_f16_to_f32(val));
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && std::is_same<TypeFrom, float>::value) {
        return _f32_to_bf16(val);
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return _f32_to_bf16(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && std::is_same<TypeTo, float>::value) {
        return _bf16_to_f32(val);
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(_bf16_to_f32(val));
    } else {
        return static_cast<TypeTo>(val);
    }
}

} // namespace utils
} // namespace llaisys
