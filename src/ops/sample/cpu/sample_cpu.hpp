#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {

void sample(
    std::byte *sampled_token,      // Output: int64 token
    const std::byte *logits,       // Input: logits array
    llaisysDataType_t logits_dtype,
    size_t vocab_size,
    float temperature,
    int top_k,
    float top_p
);

} // namespace llaisys::ops::cpu
