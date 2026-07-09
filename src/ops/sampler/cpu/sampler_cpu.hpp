#pragma once

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {

// 随机采样器接口
// logits: 最后一层输出的得分数组指针
// vocab_size: 词表大小 (比如 151936)
// temperature: 温度参数 (默认通常是 0.8 到 1.0)
// top_p: 核采样阈值 (默认通常是 0.9)
// top_k: 截断数量 (0 表示不使用 Top-K 限制)
int32_t sample(const float *logits, size_t vocab_size, float temperature, float top_p, size_t top_k);

} // namespace llaisys::ops::cpu