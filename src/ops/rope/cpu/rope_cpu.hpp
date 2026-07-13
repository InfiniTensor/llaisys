#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte* out, 
          const std::byte* in, 
          const std::byte* pos_ids, 
          llaisysDataType_t dtype, 
          size_t num_tokens, 
          size_t n_heads, 
          size_t head_dim,
          float theta);
} // namespace llaisys::ops::cpu