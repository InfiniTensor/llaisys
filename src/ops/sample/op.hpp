#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// logits: 最后一层的输出 [vocab_size]
// next_token_id: 输出张量 [1]，存放最终采样得到的 Token ID (类型为 I32/I64)
// temperature: 温度参数，默认为 1.0 (0.0 等价于 argmax)
// top_k: 采样候选数，默认为 0 (不限制)
// top_p: 核采样阈值，默认为 1.0 (不限制)
void sample(tensor_t next_token_id, tensor_t logits, float temperature = 1.0f, int top_k = 0, float top_p = 1.0f);
} // namespace llaisys::ops