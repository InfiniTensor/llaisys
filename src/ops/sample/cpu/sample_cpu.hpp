#pragma once

#include "llaisys.h"
#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {

void sample(std::byte *out, const std::byte *logits, size_t numel,
            int top_k, float top_p, float temperature,
            llaisysDataType_t dtype);

void sample_set_seed(uint64_t seed);

} // namespace llaisys::ops::cpu
