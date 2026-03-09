#include "llaisys.h"

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
        return 1; // usually 8-bit float (custom)
    case LLAISYS_DTYPE_F16:
        return 2; // 16-bit float
    case LLAISYS_DTYPE_BF16:
        return 2; // bfloat16
    case LLAISYS_DTYPE_F32:
        return sizeof(float);
    case LLAISYS_DTYPE_F64:
        return sizeof(double);
    case LLAISYS_DTYPE_C16:
        return 2; // 2 bytes complex (not standard)
    case LLAISYS_DTYPE_C32:
        return 4; // 4 bytes complex
    case LLAISYS_DTYPE_C64:
        return 8; // 8 bytes complex
    case LLAISYS_DTYPE_C128:
        return 16; // 16 bytes complex
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

// -------- AVX2 vectorized conversions (operate on 8 elements at once) --------
// These helper functions are intended for use in SIMD paths and are
// declared here with descriptive names. Actual implementations should
// use AVX2 intrinsics to process eight values in parallel.
//
// - f16x8_to_f32x8 : convert eight fp16_t values to float
// - f32x8_to_f16x8 : convert eight float values to fp16_t
// - bf16x8_to_f32x8: convert eight bf16_t values to float
// - f32x8_to_bf16x8: convert eight float values to bf16_t
//
// Note: names chosen for clarity; no code is provided in this header.
// --------------------------------------------------------------------------

__m256  f16x8_to_f32x8(__m128i packed_fp16);
__m128i f32x8_to_f16x8(__m256 packed_f32);
__m256  bf16x8_to_f32x8(__m128i packed_bf16);
__m128i f32x8_to_bf16x8(__m256 packed_f32);

} // namespace utils
} // namespace llaisys
