#include "llaisys.h"

#include <cstring>
#include <cstdlib>

#include <type_traits>
#include <new>
#include <iostream>
#include <stdexcept>

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
inline size_t dsize(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return sizeof(char);
    case LLAISYS_DTYPE_BOOL:
        return sizeof(char);
    case LLAISYS_DTYPE_I8:
        return sizeof(int8_t);
    case LLAISYS_DTYPE_I16:
        return sizeof(int16_t);
    case LLAISYS_DTYPE_I32:
        return sizeof(int32_t);
    case LLAISYS_DTYPE_I64:
        return sizeof(int64_t);
    case LLAISYS_DTYPE_U8:
        return sizeof(uint8_t);
    case LLAISYS_DTYPE_U16:
        return sizeof(uint16_t);
    case LLAISYS_DTYPE_U32:
        return sizeof(uint32_t);
    case LLAISYS_DTYPE_U64:
        return sizeof(uint64_t);
    case LLAISYS_DTYPE_F8:
        return sizeof(f8_t);
    case LLAISYS_DTYPE_F16:
        return sizeof(fp16_t);
    case LLAISYS_DTYPE_BF16:
        return sizeof(bf16_t);
    case LLAISYS_DTYPE_F32:
        return sizeof(float);
    case LLAISYS_DTYPE_F64:
        return sizeof(double);
    case LLAISYS_DTYPE_C16:
        return sizeof(cp16_t);
    case LLAISYS_DTYPE_C32:
        return sizeof(cp32_t);
    case LLAISYS_DTYPE_C64:
        return sizeof(cp64_t);
    case LLAISYS_DTYPE_C128:
        return sizeof(cp128_t);
    case LLAISYS_DTYPE_INVALID:
    default:
        throw std::invalid_argument("Unsupported or invalid data type.");
    }
}

inline const char *dtype_to_str(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return "byte";
    case LLAISYS_DTYPE_BOOL:
        return "bool";
    case LLAISYS_DTYPE_I8:
        return "int8";
    case LLAISYS_DTYPE_I16:
        return "int16";
    case LLAISYS_DTYPE_I32:
        return "int32";
    case LLAISYS_DTYPE_I64:
        return "int64";
    case LLAISYS_DTYPE_U8:
        return "uint8";
    case LLAISYS_DTYPE_U16:
        return "uint16";
    case LLAISYS_DTYPE_U32:
        return "uint32";
    case LLAISYS_DTYPE_U64:
        return "uint64";
    case LLAISYS_DTYPE_F8:
        return "float8";
    case LLAISYS_DTYPE_F16:
        return "float16";
    case LLAISYS_DTYPE_BF16:
        return "bfloat16";
    case LLAISYS_DTYPE_F32:
        return "float32";
    case LLAISYS_DTYPE_F64:
        return "float64";
    case LLAISYS_DTYPE_C16:
        return "complex16";
    case LLAISYS_DTYPE_C32:
        return "complex32";
    case LLAISYS_DTYPE_C64:
        return "complex64";
    case LLAISYS_DTYPE_C128:
        return "complex128";
    case LLAISYS_DTYPE_INVALID:
    default:
        throw std::invalid_argument("Unsupported or invalid data type.");
    }
}

float _f16_to_f32(fp16_t val);
fp16_t _f32_to_f16(float val);

float _bf16_to_f32(bf16_t val);
bf16_t _f32_to_bf16(float val);

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