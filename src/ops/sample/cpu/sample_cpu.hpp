#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {
int64_t sample(const std::byte *logits, size_t numel, llaisysDataType_t dtype, float temperature, int top_k, float top_p, uint64_t seed);
}
