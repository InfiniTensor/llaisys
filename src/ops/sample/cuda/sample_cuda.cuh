#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cuda {
void sample(std::byte *out_idx, const std::byte *logits, llaisysDataType_t type, size_t numel,
            float temperature, int top_k, float top_p);
}
