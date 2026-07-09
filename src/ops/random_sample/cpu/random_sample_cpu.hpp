#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {
void random_sample(
    std::byte *out_idx,
    const std::byte *logits,
    llaisysDataType_t type,
    size_t numel,
    float temperature,
    size_t top_k,
    float top_p,
    uint64_t seed);
}
