#include "llaisys.h"

#include <iostream>
#include <stdexcept>
#include <cstring>

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#define LLAISYS_HOST_DEVICE __host__ __device__
#include <cuda_fp16.h>
#else
#define LLAISYS_HOST_DEVICE
#endif

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

// Host-only functions (implemented in types.cpp)
float _f16_to_f32(fp16_t val);
fp16_t _f32_to_f16(float val);
float _bf16_to_f32(bf16_t val);
bf16_t _f32_to_bf16(float val);

// Device-compatible conversion functions
LLAISYS_HOST_DEVICE inline float _f16_to_f32_device(fp16_t val) {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
    // Use CUDA half arithmetic on device
    half h = __ushort_as_half(val._v);
    return __half2float(h);
#else
    // Fall back to host implementation
    return _f16_to_f32(val);
#endif
}

LLAISYS_HOST_DEVICE inline fp16_t _f32_to_f16_device(float val) {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
    // Use CUDA half arithmetic on device
    half h = __float2half(val);
    return fp16_t{__half_as_ushort(h)};
#else
    // Fall back to host implementation
    return _f32_to_f16(val);
#endif
}

LLAISYS_HOST_DEVICE inline float _bf16_to_f32_device(bf16_t val) {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
    // bf16 is stored as upper 16 bits of float32
    uint32_t bits = static_cast<uint32_t>(val._v) << 16;
    return *reinterpret_cast<float*>(&bits);
#else
    // Fall back to host implementation
    return _bf16_to_f32(val);
#endif
}

LLAISYS_HOST_DEVICE inline bf16_t _f32_to_bf16_device(float val) {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
    // Convert float to bf16 by taking upper 16 bits (with rounding)
    uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return bf16_t{static_cast<uint16_t>((bits + rounding_bias) >> 16)};
#else
    // Fall back to host implementation
    return _f32_to_bf16(val);
#endif
}

// Device-compatible cast function
template <typename TypeTo, typename TypeFrom>
LLAISYS_HOST_DEVICE TypeTo cast_device(TypeFrom val) {
    if constexpr (std::is_same<TypeTo, TypeFrom>::value) {
        return val;
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && std::is_same<TypeFrom, float>::value) {
        return _f32_to_f16_device(val);
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return _f32_to_f16_device(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && std::is_same<TypeTo, float>::value) {
        return _f16_to_f32_device(val);
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(_f16_to_f32_device(val));
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && std::is_same<TypeFrom, float>::value) {
        return _f32_to_bf16_device(val);
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return _f32_to_bf16_device(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && std::is_same<TypeTo, float>::value) {
        return _bf16_to_f32_device(val);
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(_bf16_to_f32_device(val));
    } else {
        return static_cast<TypeTo>(val);
    }
}

// Original host-only cast function (for backward compatibility)
template <typename TypeTo, typename TypeFrom>
TypeTo cast(TypeFrom val) {
    return cast_device<TypeTo, TypeFrom>(val);
}

} // namespace utils
} // namespace llaisys
