#include "llaisys.h"

#include <iostream>
#include <stdexcept>

namespace llaisys {
// 自定义半精度浮点类型 fp16_t (Float16) 和 bf16_t (BFloat16)。
// 限制：由于它们本质上是结构体包裹的整数，不能直接进行数学运算（如 a + b）。
// 必须先转换为 float 才能计算。
struct CustomFloat16 {
    uint16_t _v;
};
typedef struct CustomFloat16 fp16_t;

struct CustomBFloat16 {
    uint16_t _v;
};
typedef struct CustomBFloat16 bf16_t;

namespace utils {
// dsize(dtype)：
// 功能：返回指定数据类型占用的字节数。用途：在分配张量内存（malloc）或计算偏移量时使用。
// 例如，LLAISYS_DTYPE_F16 返回 2，LLAISYS_DTYPE_F32 返回 4
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
// dtype_to_str(dtype)：
// 功能：返回数据类型的字符串名称。
// 用途：主要用于调试打印（Debug）和错误信息的生成。
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
// 声明了 16 位浮点数与标准 32 位 float 之间的转换逻辑。
float _f16_to_f32(fp16_t val);
fp16_t _f32_to_f16(float val);

float _bf16_to_f32(bf16_t val);
bf16_t _f32_to_bf16(float val);

// 它提供了一个统一的接口 utils::cast<目标类型>(源数据)。
template <typename TypeTo, typename TypeFrom>
TypeTo cast(TypeFrom val) {
    // 使用 C++17 的 if constexpr 进行编译期分支选择
    if constexpr (std::is_same<TypeTo, TypeFrom>::value) {
        return val; // 类型相同，直接返回
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && std::is_same<TypeFrom, float>::value) {
        return _f32_to_f16(val);// float -> fp16
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
        return static_cast<TypeTo>(val); // 普通类型（如 int -> float）使用原生转换
    }
}

} // namespace utils
} // namespace llaisys
