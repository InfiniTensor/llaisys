#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
int64_t sample(const std::byte *logits, llaisysDataType_t type, size_t numel, int top_k, float top_p, float temperature);
}
