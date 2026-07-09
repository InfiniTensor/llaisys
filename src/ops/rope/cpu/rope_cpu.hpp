#pragma once
#include "../../../core/llaisys_core.hpp"

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t dtype, size_t seq_len, size_t n_heads, size_t head_dim, float theta);
}
