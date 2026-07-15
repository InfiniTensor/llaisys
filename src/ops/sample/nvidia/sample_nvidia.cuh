#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::nvidia {
int64_t sample(const std::byte *logits, llaisysDataType_t dtype, size_t numel, int top_k, float top_p, float temperature);
}
