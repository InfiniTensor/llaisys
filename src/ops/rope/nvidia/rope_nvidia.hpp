#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t type, size_t seq_len,
          size_t n_heads, size_t head_dim);
}
