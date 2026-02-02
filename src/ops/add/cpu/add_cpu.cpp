#include "add_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void add_(T *c, const T *a, const T *b, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        // 使用 C++17 的 if constexpr 进行编译期分支判断
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            // 针对半精度浮点数 (BF16, FP16) 的特殊处理：
            // CPU 通常没有针对 bf16 或 fp16 的原生算术指令（或者 C++ 标准库不支持直接相加）。
            // 因此，代码显式地将它们转换为 float 进行计算，然后再转回去。
            // 这既保证了精度，也解决了编译问题。
            c[i] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i]));
        } else {
            // 针对标准类型 (如 float) 的处理：
            // 直接使用原生加法运算符
            c[i] = a[i] + b[i];
        }
    }
}

namespace llaisys::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        // 将 std::byte* 强制转换为 float*，并调用模板函数
        return add_(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), numel);
    case LLAISYS_DTYPE_BF16:
        // 处理 BFloat16
        return add_(reinterpret_cast<llaisys::bf16_t *>(c), reinterpret_cast<const llaisys::bf16_t *>(a),
                    reinterpret_cast<const llaisys::bf16_t *>(b), numel);
    case LLAISYS_DTYPE_F16:
        // 处理 Float16
        return add_(reinterpret_cast<llaisys::fp16_t *>(c), reinterpret_cast<const llaisys::fp16_t *>(a),
                    reinterpret_cast<const llaisys::fp16_t *>(b), numel);
    default:
        // 如果遇到不支持的类型（如 int32），抛出异常
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
