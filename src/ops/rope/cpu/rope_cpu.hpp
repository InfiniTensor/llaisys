#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids, float theta, llaisysDataType_t type, size_t seq_len, size_t n_heads, size_t head_dim);
}