#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void sample(std::byte *out_idx, const std::byte *logits, llaisysDataType_t dtype, size_t numel, int top_k, float top_p, float temperature, int64_t seed);
} //namespace llaisys::ops::cpu