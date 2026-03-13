#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, float theta, size_t seq_len, size_t num_heads, size_t head_dim);
}
