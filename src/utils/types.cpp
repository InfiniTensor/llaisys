#include "types.hpp"

#include <cstring>
// 核心功能是
// 实现 32位单精度浮点数 (float) 与两种 16位浮点数格式（fp16_t即IEEE754 Half-precision，
// 和 bf16_t 即 Brain Floating Point）之间的底层位级转换。
namespace llaisys::utils {
    
// 这个函数将 16 位的 fp16 扩展为 32 位的 float。
// FP16: 1位符号位 | 5位指数位 (Bias=15) | 10位尾数位
// FP32: 1位符号位 | 8位指数位 (Bias=127) | 23位尾数位
float _f16_to_f32(fp16_t val) {
    uint16_t h = val._v;
    uint32_t sign = (h & 0x8000) << 16; // 提取第15位符号位，移动到第31位
    int32_t exponent = (h >> 10) & 0x1F; // 提取5位指数 (0x1F = 31)
    uint32_t mantissa = h & 0x3FF; // 提取10位尾数 (0x3FF = 1023) || 0x3FF 就是全0+3FF 做 与运算得到10位尾数

    uint32_t f32;
    if (exponent == 31) {// 情况 A：指数为全 1 (exponent == 31) —— 特殊值
        if (mantissa != 0) {// NaN (非数): 如果尾数不为 0。
            // 通过 (mantissa << 13) 将原有的 10 位尾数左移，填入 FP32 尾数的高位。
            f32 = sign | 0x7F800000 | (mantissa << 13);
        } else { // Infinity (无穷大): 如果尾数为 0
            // 0x7F800000 是 FP32 中指数全为 1 的表示方式。
            f32 = sign | 0x7F800000;
        }
    } else if (exponent == 0) { // 情况 B：指数为 0 (exponent == 0) —— 零或非规范化数
        if (mantissa == 0) { // Zero: 如果尾数也是 0，直接返回带符号的 0
            f32 = sign;
        } else { // Subnormal (非规范化数): 。FP16 的非规范化数非常小，没有隐含的起始位 "1"。
            // 非规范数: 指数位 E 全为 0,尾数位 M 不全为 0.
            // 实际指数：e=1-bias=1-127=-126，尾数没有隐含的 1，而是 0.M（二进制小数）
            exponent = -14;
            while ((mantissa & 0x400) == 0) { // 使用 while 循环通过左移尾数找到第一位尾数1.
                // 一旦 mantissa <<= 1 把某个 1 推到了第 11 位，(mantissa & 0x400) 就不再是 0，循环停止。
                mantissa <<= 1;
                exponent--; // 指数相应减小，保证数值大小不变
            }
            // 最后按照 FP32 的规范重新打包
            mantissa &= 0x3FF;
            f32 = sign | ((exponent + 127) << 23) | (mantissa << 13);
        }
    } else {
        // 重偏置 (Re-biasing): FP16 的指数偏移量是 15，而 FP32 是 127。所以新的指数 = 原指数 - 15 + 127。
        // 对齐: 尾数从 10 位扩展到 23 位，所以左移 13 位。
        f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    // memcpy 确保了将构造好的 32 位整数位模式原封不动地交给 float 变量。
    memcpy(&result, &f32, sizeof(result));
    return result;
}


// 实现了将 32位单精度浮点数 (Float32) 转换为 16位半精度浮点数
fp16_t _f32_to_f16(float val) {
    uint32_t f32;
    memcpy(&f32, &val, sizeof(f32));               // 将 float 的二进制位读取为整数
    uint16_t sign = (f32 >> 16) & 0x8000;          // 提取符号位并移动到第15位（FP16的符号位位置）
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127; // 提取8位指数并减去偏移量127，得到真实的指数值
    uint32_t mantissa = f32 & 0x7FFFFF;            // 提取23位尾数

    if (exponent >= 16) { // 指数过大 (exponent >= 16) —— 溢出或特殊值，FP16 的最大指数是 15
        // NaN-如果 Float32 本身是 NaN（指数为 128 且尾数不为 0），则返回 FP16 的 NaN (0x7E00)。
        if (exponent == 128 && mantissa != 0) {
            return fp16_t{static_cast<uint16_t>(sign | 0x7E00)};
        }
        // Infinity (无穷大)
        // 如果数太大超出了 FP16 范围，则返回 FP16 的正/负无穷大 (0x7C00)。
        return fp16_t{static_cast<uint16_t>(sign | 0x7C00)};
    } else if (exponent >= -14) { //  规范化数 (Normalized)
        // 重偏置：将真实指数加上 FP16 的偏移量 15。
        // 截断尾数：Float32 有 23 位尾数，而FP16只有10位。代码通过 >> 13 直接截断了低 13 位。
        // 注意：这里采用的是“向零舍入”，即直接截断，没有进行复杂的舍入处理。
        return fp16_t{(uint16_t)(sign | ((exponent + 15) << 10) | (mantissa >> 13))};
    } else if (exponent >= -24) {// 非规范化数 (Subnormal)
        // 如果指数在 -15 到 -24 之间，这个数太小了，无法用 FP16 的规范化格式表示，但可以用非规范化格式表示
        mantissa |= 0x800000;  // 补回 Float32 隐藏的领先位 1
        mantissa >>= (-14 - exponent); // 通过右移尾数来补偿过小的指数
        return fp16_t{(uint16_t)(sign | (mantissa >> 13))};
    } else { // 完全无法表示 (else) —— 归零
        // 如果指数小于 -24，这个数甚至比 FP16 能表示的最小非规范化数还要小。直接返回一个带有原始符号的 0。
        return fp16_t{(uint16_t)sign};
    }
}

// BF16: 1位符号 | 8位指数 | 7位尾数
// FP32: 1位符号 | 8位指数 | 23位尾数
float _bf16_to_f32(bf16_t val) { // 这个过程非常直接，本质上是“补零”。
    // 由于BF16的符号位和指数位与FP32的高16位完全对齐，我们只需要将这16位数据左移16位。
    // 这样，FP32 的低 16 位（尾数的后半部分）就自动补为 0
    uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;

    float out;
    std::memcpy(&out, &bits32, sizeof(out));
    return out;
}

bf16_t _f32_to_bf16(float val) {
    uint32_t bits32;
    std::memcpy(&bits32, &val, sizeof(bits32));

    const uint32_t rounding_bias = 0x00007FFF + // 0111 1111 1111 1111
                                   ((bits32 >> 16) & 1);

    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);

    return bf16_t{bf16_bits};
}
} // namespace llaisys::utils
