#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {

// 声明 Embedding 的 CPU 实现函数
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t num_tokens, size_t vocab_size, size_t hidden_size,
               llaisysDataType_t dtype);

} // namespace llaisys::ops::cpu