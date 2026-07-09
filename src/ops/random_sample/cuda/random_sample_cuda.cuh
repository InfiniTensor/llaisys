#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {
void random_sample(std::byte *sample_idx, std::byte *sample_val, const std::byte *logits, llaisysDataType_t type,
                   size_t vocab_size, float temperature, int top_k, float top_p, int64_t seed);
}
