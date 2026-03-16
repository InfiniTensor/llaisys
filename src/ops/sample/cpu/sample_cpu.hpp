#pragma once

#include "../../../tensor/tensor.hpp"
#include <cstddef>

namespace llaisys::ops::cpu {
// 核心 CPU 采样逻辑，输入为统一的单精度 float 数组
void sample_f32(int64_t* next_token_id, const float* logits, size_t vocab_size, 
                float temperature, int top_k, float top_p);
} // namespace llaisys::ops::cpu